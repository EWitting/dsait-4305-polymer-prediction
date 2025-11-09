import torch

from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, ASAPooling
from torch_geometric.data import Data

class HybridResidualGraphNetwork(torch.nn.Module):
    """Customizable class that consists of a sequence of repeated blocks along a residual stream.
    Each block consists of a sequence of convolution layers, followed by a sequence of attention layers.
    The pooling strategy, activation, and final MLP head dimensions are customizable.
    """
    def __init__(
            self,
            in_channels,
            hidden_dim,
            fc_hidden_dims,
            out_channels,   

            num_blocks,
            block, # Block factory, function without arguments that creates a block
            
            pooling,
            activation,
            dropout=None):
        super().__init__()

        # initial fc layer to expand to hidden dim
        self.fc_embed = torch.nn.Linear(in_channels, hidden_dim)

        # sequence of idential blocks with hidden dim
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(block())

        # pool layer        
        self.pooling = pooling

        # fc network to produce output
        self.head_fcs = torch.nn.ModuleList()
        head_dimensions = [hidden_dim] + fc_hidden_dims + [out_channels]
        for in_dim, out_dim in zip(head_dimensions[:-1],head_dimensions[1:]):
            self.head_fcs.append(torch.nn.Linear(in_dim,out_dim))

        # These will be used for every block and layer
        self.activation = activation
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout = torch.nn.Dropout(self.dropout)


    def forward(self, data: Data, return_attn: bool = False):
        """Forward pass. If `return_attn=True` the method will return
        a tuple `(out, attn_list)` where `attn_list` collects attention
        coefficients from attention layers inside each block.
        """
        x, edge_index = data.x, data.edge_index
        x = self.fc_embed(x)

        attn_collected = []
        # Apply blocks along the residual stream, no dropout etc. handled in the block
        for block in self.blocks:
            # Request attention collection from blocks only when return_attn=True
            try:
                res = block(x, edge_index, return_attn=return_attn)
            except TypeError:
                # Older blocks may not accept return_attn kwarg; call positional
                res = block(x, edge_index)

            if isinstance(res, tuple) and len(res) == 2:
                block_out, block_attn = res
                x = x + block_out
                if block_attn:
                    attn_collected.extend(block_attn)
            else:
                x = x + res

        # Pool the graph and apply the final fc layers with dropout and activation
        x = self.pooling(x, data.batch)
        for fc in self.head_fcs[:-1]:
            x = fc(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.activation(x)

        out = self.head_fcs[-1](x) # no dropout or activation on last linear layer
        # Return attention only if explicitly requested (backwards-compatible)
        if return_attn:
            return out, attn_collected

        return out

class HybridResidualGraphBlock(torch.nn.Module):
    def __init__(
        self,
        conv_layer,
        num_conv_layers,
        attention_layer,
        num_attention_layers,
        attention_heads,
        hidden_dim,       
        activation, 
        use_layer_norm,
        dropout,
    ):
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv_layers.append(conv_layer())
        
        self.attention_layers = torch.nn.ModuleList()
        self.attention_projections = torch.nn.ModuleList()
        for i in range(num_attention_layers):
            self.attention_layers.append(attention_layer())
            self.attention_projections.append(torch.nn.Linear(attention_heads * hidden_dim, hidden_dim))

        self.activation = activation
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(hidden_dim)
            self.layer_norm_attn = torch.nn.LayerNorm(hidden_dim*attention_heads)
        else:
            self.layer_norm = None
            self.layer_norm_attn = None
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, edge_index, return_attn: bool = False):
        attn_collected = []
        for conv_layer in self.conv_layers:
            # conv_layer is a factory that returns a conv module when called
            
            x = conv_layer(x, edge_index)

            if self.use_layer_norm:
                x = self.layer_norm(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.activation(x)

        for attention_layer, attention_projection in zip(self.attention_layers, self.attention_projections):
            # Try to call attention layer and collect attention weights when available
            attn_weights = None
            edge_idx = None
            try:
                res = attention_layer(x, edge_index, return_attention_weights=True)
                if isinstance(res, tuple) and len(res) >= 2:
                    x_headed = res[0]
                    second = res[1]
                    if isinstance(second, tuple) and len(second) >= 2:
                        edge_idx, attn_weights = second[0], second[1]
                    else:
                        attn_weights = second
                        edge_idx = edge_index
                else:
                    x_headed = res
            except TypeError:
                # conv signature doesn't accept return_attention_weights
                try:
                    x_headed = attention_layer(x, edge_index)
                    attn_weights = getattr(attention_layer, 'alpha', None)
                    edge_idx = edge_index
                except Exception:
                    x_headed = attention_layer(x, edge_index)
            except Exception:
                x_headed = attention_layer(x, edge_index)

            if self.use_layer_norm:
                # layer_norm_attn expects hidden_dim * attention_heads shape
                try:
                    x_headed = self.layer_norm_attn(x_headed)
                except Exception:
                    pass
            if self.dropout is not None:
                x_headed = self.dropout(x_headed)

            x = self.activation(attention_projection(x_headed))

            if attn_weights is not None:
                # Normalize stored format and move to cpu
                try:
                    attn_t = attn_weights.detach().cpu()
                except Exception:
                    attn_t = torch.as_tensor(attn_weights).detach().cpu()
                if isinstance(edge_idx, (tuple, list)):
                    edge_idx = edge_idx[0]
                attn_collected.append((edge_idx.detach().cpu() if hasattr(edge_idx, 'detach') else edge_idx, attn_t))

        if return_attn:
            return x, attn_collected

        if attn_collected:
            return x, attn_collected

        return x
