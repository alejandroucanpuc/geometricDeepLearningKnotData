import keras
from keras import layers
from keras import regularizers
from spektral.layers import (
    CrystalConv,
    GlobalSumPool,
    GlobalAvgPool,
    GlobalMaxPool,
)


class ResidualCrystalBlock(keras.layers.Layer):
    """
    Bloque residual: BN -> ReLU -> CrystalConv -> Dropout -> (proyección residual) -> suma.
    """
    def __init__(self, channels, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.dropout = dropout
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
        # Mantengo activación fuera para BN->ReLU->Conv
        self.conv = CrystalConv(channels=channels, activation=None)
        self.drop = layers.Dropout(dropout)
        self.proj = None  # se crea en build si hace falta

    def build(self, input_shape):
        # inputs: (x, a, e)
        x_shape = input_shape[0]
        in_ch = x_shape[-1]
        if in_ch != self.channels:
            self.proj = layers.Dense(self.channels, use_bias=False)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, a, e = inputs
        h = self.bn(x, training=training)
        h = self.act(h)
        h = self.conv([h, a, e])
        h = self.drop(h, training=training)
        shortcut = x if self.proj is None else self.proj(x)
        return h + shortcut


class NeuralKnotNetKeras(keras.Model):
    """
    GNN con CrystalConv + bloques residuales + Global[Sum|Avg|Max]Pool concatenados + MLP final.
    Si ADDITIONAL=True, el preproceso concatena data.featureEncoding a cada nodo (ver keras_train.py).
    """
    def __init__(self, channels=96, n_layers=6, dropout=0.35, l2=5e-5):
        super().__init__()
        reg = regularizers.l2(l2)

        # Capa de proyección inicial
        self.pre = layers.Dense(channels, activation="relu", kernel_regularizer=reg)

        # Bloques residuales
        self.blocks = [
            ResidualCrystalBlock(channels=channels, dropout=dropout)
            for _ in range(n_layers)
        ]

        # Poolings globales
        self.pool_sum = GlobalSumPool()
        self.pool_avg = GlobalAvgPool()
        self.pool_max = GlobalMaxPool()
        self.concat = layers.Concatenate()

        # MLP final con L2 + Dropout
        self.mlp = keras.Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(128, activation="relu", kernel_regularizer=reg),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu", kernel_regularizer=reg),
            layers.Dropout(dropout),
            layers.Dense(1, kernel_regularizer=reg),  # regresión
        ])

    def call(self, inputs, training=False):
        # inputs (disjoint): (x, a, e, i)
        x, a, e, i = inputs
        x = self.pre(x)
        for blk in self.blocks:
            x = blk([x, a, e], training=training)

        g_sum = self.pool_sum([x, i])
        g_avg = self.pool_avg([x, i])
        g_max = self.pool_max([x, i])
        g = self.concat([g_sum, g_avg, g_max])  # concat sum/avg/max

        return self.mlp(g, training=training)  # [batch, 1]
