import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, ASAPooling
from torch_geometric.data import Data
from torch_geometric.nn.pool import SAGPooling


class GATv2(torch.nn.Module):
    """GAT using GATv2 from https://arxiv.org/pdf/2105.14491.
    I assume that we want to predict 5 regression targets.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int = 5,
        hidden_channels: list[int] = [128],
        hidden_layers: list[int] = [16, 32],
        act: nn.Module = nn.ELU(),
        heads: int = 8,
        edge_dim: int | None = None,
        dropout: float = 0.1,
        residual: bool = True,
        use_descs: bool = False,
    ):
        super().__init__()
        # --- Encoder Part ---
        self.attention_layers = nn.ModuleList()
        self.attention_layers.extend(
            [
                GATv2Conv(
                    in_channels=in_channel,
                    out_channels=hidden_channels[0],
                    heads=heads,
                    residual=residual,
                    dropout=dropout,
                    edge_dim=edge_dim,
                ),
                nn.Linear(heads * hidden_channels[0], hidden_channels[0]),
            ]
        )
        for in_dim, out_dim in zip(hidden_channels[:-1], hidden_channels[1:]):
            self.attention_layers.extend(
                [
                    GATv2Conv(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        heads=heads,
                        residual=residual,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    ),
                    nn.Linear(heads * out_dim, out_dim),
                ]
            )

        self.global_pool = global_mean_pool

        if use_descs:
            self.fcs = nn.Sequential(
                nn.Linear(hidden_channels[-1] + 217, hidden_layers[0]), nn.ReLU()
            )
        else:
            self.fcs = nn.Sequential(
                nn.Linear(hidden_channels[-1], hidden_layers[0]), nn.ReLU()
            )

        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.fcs.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.fcs.append(nn.Linear(hidden_layers[-1], out_channel))  # Use out_channel
        self.act = act
        self.use_descs = use_descs

    def encode(self, data: Data, return_attn: bool = False):
        """
        Extracts node-level embeddings from the graph.
        This is the "backbone" or "encoder" part.

        If `return_attn=True`, returns (x, attn_collected)
        Otherwise, returns x (node embeddings).
        """
        # Assumes 'data' is a standard PyG Data/Batch object
        x = data.x.float()
        attn_collected = []

        for block_idx in range(0, len(self.attention_layers), 2):
            attn, lin = self.attention_layers[block_idx : block_idx + 2]

            if return_attn:
                try:
                    res = attn(x, data.edge_index, return_attention_weights=True)
                    if isinstance(res, tuple) and len(res) >= 2:
                        x_headed = res[0]
                        second = res[1]
                        if isinstance(second, tuple) and len(second) >= 2:
                            edge_idx, attn_weights = second[0], second[1]
                        else:
                            edge_idx, attn_weights = data.edge_index, second
                    else:
                        x_headed = res
                        edge_idx, attn_weights = data.edge_index, None
                except TypeError:
                    x_headed = attn(x, data.edge_index)
                    edge_idx, attn_weights = data.edge_index, getattr(
                        attn, "alpha", None
                    )
                except Exception:
                    x_headed = attn(x, data.edge_index)
                    edge_idx, attn_weights = data.edge_index, getattr(
                        attn, "alpha", None
                    )
            else:
                x_headed = attn(x, data.edge_index)
                edge_idx, attn_weights = None, None

            x = self.act(lin(x_headed))  # Apply activation

            if return_attn and attn_weights is not None:
                try:
                    attn_t = attn_weights.detach().cpu()
                except Exception:
                    attn_t = torch.as_tensor(attn_weights).detach().cpu()
                if isinstance(edge_idx, (tuple, list)):
                    edge_idx = edge_idx[0]
                attn_collected.append(
                    (
                        (
                            edge_idx.detach().cpu()
                            if hasattr(edge_idx, "detach")
                            else edge_idx
                        ),
                        attn_t,
                    )
                )

        if return_attn:
            return x, attn_collected

        return x

    def forward(self, data: Data, return_attn: bool = False):
        """
        Forward pass for the *downstream task* (prediction).
        Handles `use_descs` tuple, calls encode(), and applies pooling/head.
        """
        descs = None
        attn_collected = []

        # Handle the (data, descs) tuple if 'use_descs' is True
        if self.use_descs:
            if not isinstance(data, (tuple, list)) or len(data) != 2:
                # This error check is good practice for the downstream task
                raise ValueError(
                    "GATv2.use_descs=True, but data was not a (Data, descs) tuple."
                )
            data, descs = data

        # --- Call the new encode method ---
        if return_attn:
            x, attn_collected = self.encode(data, return_attn=True)
        else:
            x = self.encode(data, return_attn=False)
        # -----------------------------------

        # Global pooling
        x = self.global_pool(x, data.batch)

        # Concatenate graph-level descriptors
        if self.use_descs:
            if descs is None:
                raise ValueError("GATv2.use_descs=True, but 'descs' is None.")
            x = torch.cat([x, descs], dim=-1)

        # Pass through the final prediction head
        out = self.fcs(x)

        if return_attn:
            return out, attn_collected

        return out
