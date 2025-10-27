import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data


class NodeMLP(torch.nn.Module):
    """
    MLP for node/graph data that always uses global pooling.

    - Operates on node features (data.x), applies several Linear layers + activation
      per node, then applies global_mean_pool(data.batch) and a final linear head.
    - Always uses global pooling and expects graph data structure.
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

        # Always use global pooling for graph data
        self.global_pool = global_mean_pool
        self.fc = nn.Linear(hidden_channels[-1], out_channels)

    def forward(self, data) -> Tensor:
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


class GraphMLP(torch.nn.Module):
    """
    Graph Linear Model for flat tensor input that never uses pooling.

    - Takes flat tensor input of shape [batch_size, in_channels]
    - Applies several Linear layers + activation
    - Never uses pooling
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

        # Final output layer (no pooling)
        self.fc = nn.Linear(hidden_channels[-1], out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, in_channels]

        Returns:
            Tensor of shape [batch_size, out_channels]
        """
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if self.use_layer_norm:
                x = self.norms[i](x)
            x = self.act(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Final output layer (no pooling)
        x = self.fc(x)
        return x