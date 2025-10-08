import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.nn import GATv2Conv, GATConv, GAT
from torch_geometric.nn import global_mean_pool, ASAPooling
from torch_geometric.data import Data

class GATv2(torch.nn.Module):
    """GAT using GATv2 from https://arxiv.org/pdf/2105.14491. 
    I assume that we want to predict 5 regression targets.
    """
    def __init__(
        self, 
        in_channel: int,
        hidden_channels: list | int = list([128]),
        hidden_layers: list | int = list([16, 32]),   
        act: nn.Module = nn.ELU(),
        heads: int = 8, 
        edge_dim: int | None = None, 
        dropout: float = 0.1, 
        residual: bool = True
    ):
        super().__init__()
        self.attention_layers = nn.ModuleList()
        self.attention_layers.extend(
            [GATv2Conv(
                in_channels=in_channel,
                out_channels=hidden_channels[0],
                heads=heads,
                residual=residual,
                dropout=dropout, 
                edge_dim=edge_dim
            ), 
            nn.Linear(heads * hidden_channels[0], hidden_channels[0])] # project back to non-headed
        )
        
        for in_dim, out_dim in zip(hidden_channels[:-1], hidden_channels[1:]):
            self.attention_layers.extend(
                [GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    residual=residual,
                    dropout=dropout, 
                    edge_dim=edge_dim
                ), 
                nn.Linear(heads * out_dim, out_dim)]
            )
    
        # Hierachal pooling here #TODO
        self.pooling1 = ASAPooling(in_channels=hidden_channels[-1], add_self_loops=True)
        self.global_pool = global_mean_pool
        
        self.fcs = nn.Sequential(nn.Linear(hidden_channels[-1], hidden_layers[0]), nn.ReLU())
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.fcs.extend(
                [nn.Linear(in_dim, out_dim), nn.ReLU()]
            ) 
        self.fcs.append(nn.Linear(hidden_layers[-1], 5))
        self.act = act
    
    def forward(self, data: Data) -> Tensor:
        x = data.x
        for block_idx in range(0, len(self.attention_layers), 2):
            attn, lin = self.attention_layers[block_idx:block_idx+2]
            x_headed = attn(x, data.edge_index)
            x = self.act(lin(x_headed))
            
        pool_out = self.pooling1(x, data.edge_index, batch=data.batch)
        x = self.global_pool(pool_out[0], pool_out[3])
        out = self.fcs(x)
        
        return out
        
    