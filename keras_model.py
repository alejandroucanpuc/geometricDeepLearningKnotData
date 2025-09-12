# keras_model.py
import keras
from keras import layers
from spektral.layers import CrystalConv, GlobalSumPool  # Convolución con e (edge feats)

class NeuralKnotNetKeras(keras.Model):
    """
    GNN con CrystalConv (usa atributos de arista) + GlobalSumPool + MLP final.
    Inyectamos featureEncoding en x en el pre-proceso, así que no entra como input aparte.
    """
    def __init__(self, channels=64, edge_channels=2, n_layers=4,
                 use_additional=False, additional_dim=0):
        super().__init__()
        # NOTA: ya no usamos 'use_additional': add va dentro de x
        self.pre = layers.Dense(channels, activation="relu")  # proyección inicial de nodos
        self.convs = [CrystalConv(activation="relu") for _ in range(n_layers)]
        self.pool = GlobalSumPool()
        self.mlp = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)  # regresión (para binario: Dense(1, activation="sigmoid"))
        ])

    def call(self, inputs):
        # inputs (disjoint): (x, a, e, i)
        x, a, e, i = inputs
        x = self.pre(x)
        for conv in self.convs:
            x = conv([x, a, e])
        g = self.pool([x, i])  # [batch, channels]
        return self.mlp(g)     # [batch, 1]
