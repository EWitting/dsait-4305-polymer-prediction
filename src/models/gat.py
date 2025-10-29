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
        self.act = act

        # --- Pooling Part ---
        from torch_geometric.nn.pool import global_mean_pool

        # replaced with global mean pooling for now
        self.pooling1 = global_mean_pool

        # --- Prediction Head Part ---
        self.fcs = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_layers[0]), nn.ReLU()
        )
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.fcs.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.fcs.append(nn.Linear(hidden_layers[-1], out_channel))

    def encode(self, data: Data) -> Tensor:
        """Encodes node features into rich embeddings."""
        x = data.x
        for block_idx in range(0, len(self.attention_layers), 2):
            attn, lin = self.attention_layers[block_idx : block_idx + 2]
            x_headed = attn(x, data.edge_index)
            x = self.act(lin(x_headed))
        return x

    def forward(self, data: Data) -> Tensor:
        """The full end-to-end forward pass for supervised training."""
        # 1. Get node embeddings
        node_embeddings = self.encode(data)

        # 2. Pool node embeddings to get a graph embedding
        #    self.pooling1 is global_mean_pool, so its output
        #    is the final graph_embedding tensor.
        graph_embedding = self.pooling1(node_embeddings, batch=data.batch)

        # 3. Predict final values from the graph embedding
        out = self.fcs(graph_embedding)
        return out
