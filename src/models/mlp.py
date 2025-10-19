import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

class SimpleMLP(torch.nn.Module):
    """
    Simple multi-layer perceptron that mirrors the API/style of SimpleGCN.

    - Operates on node features (data.x), applies several Linear layers + activation
      per node, then applies global_mean_pool(data.batch) and a final linear head.
    - Constructor and forward are aligned with [SimpleGCN](http://_vscodecontentref_/4) to make swapping models easy.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list | int,
        out_channels: int,
        activation: nn.Module,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Normalize single int to list
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        self.hidden_channels = hidden_channels
        self.use_layer_norm = use_layer_norm
        self.act = activation

        # Dropout module (takes care of train/eval automatically)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Build Linear layers and optional LayerNorm(s)
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        # First layer
        self.linears.append(nn.Linear(in_channels, hidden_channels[0]))
        if use_layer_norm:
            self.norms.append(nn.LayerNorm(hidden_channels[0]))

        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            self.linears.append(nn.Linear(hidden_channels[i], hidden_channels[i + 1]))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels[i + 1]))

        # Pooling and final head
        self.global_pool = global_mean_pool
        self.fc = nn.Linear(hidden_channels[-1], out_channels)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass.

        Expects:
        - data.x: node feature tensor of shape [num_nodes, in_channels]
        - data.batch: assignment vector mapping nodes -> example in batch (required by global_mean_pool)

        Returns:
        - Tensor of shape [batch_size, out_channels]
        """
        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Input `data` must have `x` (node features).")
        if not hasattr(data, "batch") or data.batch is None:
            raise ValueError("Input `data` must have `batch` for global pooling.")

        x = data.x  # [num_nodes, in_channels]

        for i, lin in enumerate(self.linears):
            x = lin(x)
            if self.use_layer_norm:
                x = self.norms[i](x)
            x = self.act(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Pool to graph-level and final head
        x = self.global_pool(x, data.batch)
        x = self.fc(x)
        return x