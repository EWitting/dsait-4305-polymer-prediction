import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, LayerNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

# Example from the PyTorch Geometric README
class SimpleGCN(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: list | int, 
        out_channels: int,
        activation: nn.Module,
        improved: bool = False,
        use_layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Handle single int or list of hidden channels
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        
        self.hidden_channels = hidden_channels
        self.use_layer_norm = use_layer_norm
        self.act = activation
        
        # Dropout module (handles training mode automatically)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels[0], improved=improved))
        if use_layer_norm:
            self.norms.append(LayerNorm(hidden_channels[0]))
        
        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1], improved=improved))
            if use_layer_norm:
                self.norms.append(LayerNorm(hidden_channels[i+1]))
        
        self.global_pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_channels[-1], out_channels)

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_layer_norm:
                x = self.norms[i](x)
            x = self.act(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        x = self.global_pool(x, data.batch)
        x = self.fc(x)
        return x