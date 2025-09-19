# --- (opcional) desactivar oneDNN para resultados CPU más reproducibles ---
# Debe definirse ANTES de importar keras/tensorflow.
import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

os.environ["KERAS_BACKEND"] = "tensorflow"  # backend estable para Spektral

import numpy as np
import scipy.sparse as sp
import random
import keras
import tensorflow as tf
from spektral.data import Graph, Dataset
from spektral.data.loaders import DisjointLoader

from settings import BATCH_SIZE, LOSS, ADDITIONAL
import dataset
from keras_model import NeuralKnotNetKeras


# ---------- Semillas para reproducibilidad ----------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ---------- Dataset wrapper para listas de Graph ----------
class GraphListDataset(Dataset):
    def __init__(self, graphs, **kwargs):
        self._graphs = graphs
        super().__init__(**kwargs)
    def read(self):
        return self._graphs


# ---------- util: robusto a tensor/array/int ----------
def _to_numpy(x, dtype=np.float32, shape=None):
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.array(x, dtype=dtype)
    if shape is not None:
        x = x.reshape(shape)
    return x


# ---------- PyG -> Spektral (inyectando featureEncoding a cada nodo si ADDITIONAL=True) ----------
def pyg_graph_to_spektral(data):
    """
    Convierte torch_geometric.data.Data -> spektral.data.Graph.
    - x: [N, F]
    - a: matriz dispersa NxN construida desde edge_index
    - e: atributos de arista [E, Fe] (si existen)
    - y: etiqueta (shape (1,))
    - Si ADDITIONAL=True y existe data.featureEncoding [1, D], se concatena a cada nodo.
    """
    # Nodos
    x = _to_numpy(getattr(data, "x", [[0.0]]))      # [N, F]
    n = x.shape[0]

    # Pegar featureEncoding a cada nodo (si procede)
    add = getattr(data, "featureEncoding", None)
    if ADDITIONAL and add is not None:
        add_np = _to_numpy(add)                     # [1, D]
        add_rep = np.repeat(add_np, n, axis=0)      # [N, D]
        x = np.concatenate([x, add_rep], axis=1)    # [N, F+D]

    # Aristas
    ei = _to_numpy(getattr(data, "edge_index", [[0], [0]]), dtype=np.int64)  # [2, E]
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index con forma inesperada: {ei.shape}")
    E = ei.shape[1]
    a = sp.csr_matrix((np.ones(E, dtype=np.float32), (ei[0], ei[1])), shape=(n, n))

    # Atributos de arista
    e = None
    if getattr(data, "edge_attr", None) is not None:
        e = _to_numpy(data.edge_attr)               # típico [E, Fe]

    # Etiqueta: forma (1,) para que el batch sea (B, 1)
    y = _to_numpy(getattr(data, "y", 0.0), dtype=np.float32).reshape(1)

    return Graph(x=x, a=a, e=e, y=y)


def pyg_dataset_to_spektral_graphs(pyg_dataset):
    return [pyg_graph_to_spektral(g) for g in pyg_dataset]


# ---------- Cargar conjuntos desde tu pipeline PyG ----------
train_loader, test_loader, deg = dataset.load()
print("Dataset consists of ", len(train_loader.dataset), " train samples and ", len(test_loader.dataset), " test samples")

# Pasar a listas PyG
train_pyg = list(train_loader.dataset)
test_pyg  = list(test_loader.dataset)

# Convertir a Spektral
train_graphs = pyg_dataset_to_spektral_graphs(train_pyg)
test_graphs  = pyg_dataset_to_spektral_graphs(test_pyg)

# Envolver en Dataset de Spektral
train_ds = GraphListDataset(train_graphs)
val_ds   = GraphListDataset(test_graphs)

# Loaders Spektral (disjoint batching). Mantengo shuffle=False aquí,
# y barajo más abajo con tf.data.shuffle() para evitar el warning.
train_loader_keras = DisjointLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader_keras   = DisjointLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


# ---------- (Opcional) set grande de generalización ----------
big_loader = None
try:
    big = dataset.fetchDataset("datasets/meanAbove71WithGraph.csv")["dataPoint"]
    big_graphs = pyg_dataset_to_spektral_graphs(list(big))
    big_ds = GraphListDataset(big_graphs)
    big_loader = DisjointLoader(big_ds, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print("Aviso: no se pudo cargar el set grande de generalización:", e)


# ---------- Datasets tf.data con barajado y prefetch ----------
AUTOTUNE = tf.data.AUTOTUNE
train_tfds = (
    train_loader_keras
    .load()
    .shuffle(4096, reshuffle_each_iteration=True)  # barajado seguro
    .prefetch(AUTOTUNE)
)
val_tfds = (
    val_loader_keras
    .load()
    .prefetch(AUTOTUNE)
)


# ---------- Modelo Keras ----------
model = NeuralKnotNetKeras(
    channels=96,
    n_layers=6,
    dropout=0.35,
    l2=5e-5,  # L2 regularization en Dense
)

# Pérdida (alineado con LOSS de settings)
if LOSS == "abs":
    loss_fn = keras.losses.MeanAbsoluteError()
elif LOSS == "squared":
    loss_fn = keras.losses.MeanSquaredError()
else:
    loss_fn = keras.losses.Huber()

# Optimizador y LR schedule
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-4,
    weight_decay=3e-4,  # un poco más de wd
    clipnorm=1.0,       # grad clipping
)

# Callbacks: ReduceLROnPlateau + EarlyStopping + Checkpoint
ckpt_path = "best_model_keras.weights.h5"
cbs = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True,
        save_weights_only=True, verbose=1
    ),
]

model.compile(optimizer=optimizer, loss=loss_fn)

# ---------- Entrenamiento ----------
EPOCHS = 300
history = model.fit(
    train_tfds,
    steps_per_epoch=train_loader_keras.steps_per_epoch,
    validation_data=val_tfds,
    validation_steps=val_loader_keras.steps_per_epoch,
    epochs=EPOCHS,
    verbose=1,
    callbacks=cbs,
)

# ---------- Evaluación práctica: MAE/RMSE + CSV de predicciones ----------
def evaluate_and_dump(name, loader, model):
    import math
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    ys_all, ps_all = [], []

    # Itera el tf.data.Dataset de forma finita:
    ds = loader.load().take(loader.steps_per_epoch)

    for (x, a, e, i), y in ds:
        # Predicción como tensor -> numpy
        p = model((x, a, e, i), training=False).numpy()  # (B, 1)
        y = y.numpy().reshape(-1, 1)

        # Asegurar forma (B,1)
        if p.ndim == 1:
            p = p.reshape(-1, 1)

        err = np.abs(p - y)
        mae_sum += float(err.sum())
        mse_sum += float(((p - y) ** 2).sum())
        n += y.shape[0]

        ys_all.append(y)
        ps_all.append(p)

    mae = mae_sum / max(n, 1)
    rmse = np.sqrt(mse_sum / max(n, 1))
    print(f"{name} -> MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    ys_all = np.vstack(ys_all).ravel()
    ps_all = np.vstack(ps_all).ravel()
    np.savetxt(f"{name}_preds.csv",
               np.c_[ys_all, ps_all],
               delimiter=",", header="y_true,y_pred", comments="")
    print(f"Guardado {name}_preds.csv")

# Val split
evaluate_and_dump("val", val_loader_keras, model)

# Set grande (si existe)
if big_loader is not None:
    evaluate_and_dump("big", big_loader, model)
    res = model.evaluate(
        big_loader.load().take(big_loader.steps_per_epoch),
        steps=big_loader.steps_per_epoch,
        verbose=0
    )
    print("Val (generalización/grande) - Keras loss:", res)