# keras_train.py
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Backend estable para Spektral

import numpy as np
import scipy.sparse as sp
import keras
from spektral.data import Graph, Dataset
from spektral.data.loaders import DisjointLoader

from settings import BATCH_SIZE, LOSS, ADDITIONAL
import dataset
from keras_model import NeuralKnotNetKeras

# ---------- util: robusto a tensor/array/int ----------
def _to_numpy(x, dtype=np.float32, shape=None):
    try:
        if hasattr(x, "detach"):  # tensor torch
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.array(x, dtype=dtype)
    if shape is not None:
        x = x.reshape(shape)
    return x

# ---------- PyG -> Spektral (inyectando featureEncoding en x si ADDITIONAL=True) ----------
def pyg_graph_to_spektral(data):
    """
    Convierte torch_geometric.data.Data -> spektral.data.Graph.
    Si ADDITIONAL=True y existe data.featureEncoding (shape [1, D]),
    se concatena ese vector a cada nodo de x (repitiéndolo N veces).
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
        e = _to_numpy(data.edge_attr)               # típico [E, 2]

    # Etiqueta: **forma (1,)** (no (1,1)) para que el batch sea (B, 1)
    y = _to_numpy(getattr(data, "y", 0.0), dtype=np.float32).reshape(1)

    return Graph(x=x, a=a, e=e, y=y)

def pyg_dataset_to_spektral_graphs(pyg_dataset):
    return [pyg_graph_to_spektral(g) for g in pyg_dataset]

# ---------- Dataset wrapper para listas de Graph ----------
class GraphListDataset(Dataset):
    def __init__(self, graphs, **kwargs):
        self._graphs = graphs
        super().__init__(**kwargs)
    def read(self):
        return self._graphs

# ---------- Cargar conjuntos desde tu pipeline ----------
train_loader, test_loader, deg = dataset.load()
train_pyg = list(train_loader.dataset)
test_pyg  = list(test_loader.dataset)

# (Opcional) set grande de generalización
big_loader = None
try:
    big = dataset.fetchDataset("datasets/meanAbove71WithGraph.csv")["dataPoint"]
    big_graphs = pyg_dataset_to_spektral_graphs(list(big))
    big_ds = GraphListDataset(big_graphs)
    big_loader = DisjointLoader(big_ds, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print("Aviso: no se pudo cargar el set grande de generalización:", e)

# Convertir a Spektral
train_graphs = pyg_dataset_to_spektral_graphs(train_pyg)
test_graphs  = pyg_dataset_to_spektral_graphs(test_pyg)

# Envolver en Dataset de Spektral
train_ds = GraphListDataset(train_graphs)
val_ds   = GraphListDataset(test_graphs)

# Loaders Spektral (disjoint batching)
train_loader_keras = DisjointLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader_keras   = DisjointLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# edge_channels para CrystalConv (normalmente 2 en tu repo)
edge_channels = 0
if len(train_graphs) and train_graphs[0].e is not None:
    edge_channels = train_graphs[0].e.shape[1]

# ---------- Modelo Keras ----------
model = NeuralKnotNetKeras(
    channels=64,
    edge_channels=edge_channels,
    n_layers=4,
    use_additional=False,
    additional_dim=0,
)

# Pérdida igual que en PyTorch
loss_fn = keras.losses.MeanAbsoluteError() if LOSS == "abs" else keras.losses.MeanSquaredError()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn)

# ---------- Entrenamiento ----------
EPOCHS = 200  # Épocas
history = model.fit(
    train_loader_keras.load(),
    steps_per_epoch=train_loader_keras.steps_per_epoch,
    validation_data=val_loader_keras.load(),
    validation_steps=val_loader_keras.steps_per_epoch,
    epochs=EPOCHS,
    verbose=1,
)

# ---------- Evaluación en set grande (si existe) ----------
if big_loader is not None:
    res = model.evaluate(
        big_loader.load(),
        steps=big_loader.steps_per_epoch,
        verbose=0,
    )
    print("Val (generalización/grande):", res)

