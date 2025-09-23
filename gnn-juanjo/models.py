# Libraries
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Sigmoid, Tanh
from torch_geometric.nn import (BatchNorm, PNAConv, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.nn.conv import GATConv, TransformerConv
import numpy as np
from settings import *

# Model PNAKnotNet
class PNAKnotNet(torch.nn.Module):
    """
    PNA Knot Neural Network model.
    """
    def __init__(self,deg):
        super().__init__()
        self.channels=20 # Cantidad de canales que se van a usar en las capas ocultas de convolución.

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList() # Lista de capas de convolución.
        self.batch_norms = ModuleList() # Lista de capas de normalización por lotes.

        # Define the PNA layers
        conv = PNAConv(in_channels=1, out_channels=self.channels,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=-1, towers=4, pre_layers=1, post_layers=1,
                            divide_input=False)
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(self.channels))
        for _ in range(4):
            conv = PNAConv(in_channels=self.channels, out_channels=self.channels,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=-1, towers=4, pre_layers=1, post_layers=1,
                            divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.channels))
        
        mlpInputSize = self.channels*3

        
        if ADDITIONAL:
            mlpInputSize+=10
        self.mlp = Sequential(
            Linear(mlpInputSize, 25),
            BatchNorm(25), Tanh(), # 3 because of the maximum, average, and summation channels
            Linear(25, 25), Tanh(),
            Linear(25, 25), Tanh(),
            Linear(25, 25), ReLU(),
            Linear(25, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, additionalFeatures = None):
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = torch.tanh(
                    batch_norm(
                            conv(x, edge_index, edge_attr)
                    )
                )
        summation = global_add_pool(x, batch)
        average = global_mean_pool(x, batch)
        maximum = global_max_pool(x, batch)
        
        x = torch.cat([maximum, average, summation], dim=1)
        
        if ADDITIONAL:
            x = torch.cat([x, additionalFeatures], dim=1)
        x = self.mlp(x)
        return x
    


# Model GATKnotNet
class GATKnotNet(torch.nn.Module):
    def __init__(self, deg):
        super().__init__()
        self.channels=20

        self.convs = ModuleList()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()


        # Define the GAT layers
        self.heads = 1
        conv = GATConv(in_channels=1, out_channels=self.channels,
                            aggr=aggregators, #fill_value=scalers, #deg=deg,
                            edge_dim=-1, heads=self.heads,# pre_layers=1, post_layers=1,
        )
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(self.channels))
        
        for _ in range(25):
            conv = GATConv(in_channels=self.channels, out_channels=self.channels,
                            aggr=aggregators, #fill_value=scalers, #deg=deg,
                            edge_dim=-1, heads=self.heads,# pre_layers=1, post_layers=1,
            )
            for _ in range(1):
                self.convs.append(conv)
                self.batch_norms.append(BatchNorm(self.channels))

        mlpInputSize = self.channels*4
        
        if ADDITIONAL:
            mlpInputSize+=10
        self.mlp = Sequential(
            Linear(mlpInputSize, 25),
            BatchNorm(25), Tanh(), # 3 because of the maximum,average,summation channels
            Linear(25, 25), Tanh(),
            Linear(25, 25), Tanh(),
            Linear(25, 25), ReLU(),
            Linear(25, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, additionalFeatures=None):
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = torch.tanh(
                    batch_norm(
                            conv(x, edge_index, edge_attr)
                    )
                )

        summation = global_add_pool(x, batch)
        average = global_mean_pool(x, batch)
        maximum = global_max_pool(x, batch)
        
        x = torch.cat([maximum, average, summation], dim = 1)

        if ADDITIONAL:
            x = torch.cat([x, additionalFeatures], dim = 1)
        x = self.mlp(x)
        return x
    

# Model TransKnotNet
class TransKnotNet(torch.nn.Module):
    def __init__(self,deg):
        super().__init__()
        self.convs = ModuleList()

        # Define the Transformer layers
        conv = TransformerConv(in_channels=1, out_channels=1)
        self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr, batch):
        
        for conv in self.convs:
            x = conv(x=x, edge_index = edge_index, edge_attr = edge_attr)

        return x