import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

# Example from the PyTorch Geometric README
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.global_pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.global_pool(x, data.batch)
        x = self.fc(x)
        return x