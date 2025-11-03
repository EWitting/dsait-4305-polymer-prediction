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
        residual: bool = True, 
        use_descs: bool = False
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
    
        self.global_pool = global_mean_pool
        
        if use_descs:
            self.fcs = nn.Sequential(nn.Linear(hidden_channels[-1] + 217, hidden_layers[0]), nn.ReLU())
        else:
            self.fcs = nn.Sequential(nn.Linear(hidden_channels[-1], hidden_layers[0]), nn.ReLU())
            
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.fcs.extend(
                [nn.Linear(in_dim, out_dim), nn.ReLU()]
            ) 
        self.fcs.append(nn.Linear(hidden_layers[-1], 5))
        self.act = act
        self.use_descs = use_descs
    
    def forward(self, data: Data, return_attn: bool = False):
        """Forward pass.

        If `return_attn=True` the method returns a tuple (out, attn_list)
        where `attn_list` is a list of (edge_index, attn_weights) collected
        from each GAT layer that supports returning attention weights.
        This is a minimal, backward-compatible extension used for
        post-hoc explainability.
        """
        if self.use_descs:
            data, descs = data
        else:
            data = data
            
        x = data.x
        attn_collected = []

        for block_idx in range(0, len(self.attention_layers), 2):
            attn, lin = self.attention_layers[block_idx:block_idx+2]
            if return_attn:
                try:
                    res = attn(x, data.edge_index, return_attention_weights=True)
                    # res often is (out, (edge_index, attn_weights))
                    if isinstance(res, tuple) and len(res) >= 2:
                        x_headed = res[0]
                        second = res[1]
                        if isinstance(second, tuple) and len(second) >= 2:
                            edge_idx, attn_weights = second[0], second[1]
                        else:
                            # (out, attn_weights) — reuse data.edge_index
                            edge_idx, attn_weights = data.edge_index, second
                    else:
                        # Unexpected structure; fall back to normal call
                        x_headed = res
                        edge_idx, attn_weights = data.edge_index, None
                except TypeError:
                    # return_attention_weights not in this PyG version — call normally
                    x_headed = attn(x, data.edge_index)
                    edge_idx, attn_weights = data.edge_index, getattr(attn, 'alpha', None)
                except Exception:
                    # Any other issue, fall back gracefully
                    x_headed = attn(x, data.edge_index)
                    edge_idx, attn_weights = data.edge_index, getattr(attn, 'alpha', None)
            else:
                x_headed = attn(x, data.edge_index)
                edge_idx, attn_weights = None, None

            x = self.act(lin(x_headed))

            if return_attn and attn_weights is not None:
                # Normalize/standardize stored format: ensure tensors on cpu
                try:
                    attn_t = attn_weights.detach().cpu()
                except Exception:
                    attn_t = torch.as_tensor(attn_weights).detach().cpu()
                # If edge_idx is a tuple (as some PyG versions return), take first element
                if isinstance(edge_idx, (tuple, list)):
                    edge_idx = edge_idx[0]
                attn_collected.append((edge_idx.detach().cpu() if hasattr(edge_idx, 'detach') else edge_idx, attn_t))

        x = self.global_pool(x[0], x[3])
        
        if self.use_descs:
            x = torch.cat([x, descs], dim=-1)
        out = self.fcs(x)

        if return_attn:
            return out, attn_collected

        return out
        
    