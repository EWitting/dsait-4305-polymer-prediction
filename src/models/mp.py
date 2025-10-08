import torch 
import torch.nn as nn

from torch_geometric.nn import MessagePassing

class BasicMP(MessagePassing):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        heads, 
        dropout, 
        edge_dim, 
        residual=True
    ):
        super().__init__(aggr='add')