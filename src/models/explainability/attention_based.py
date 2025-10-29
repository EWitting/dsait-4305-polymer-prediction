"""Attention-based post-hoc explainer helpers.

This module provides a lightweight explainer that extracts attention
coefficients from GAT/GATv2 layers (when available) and converts them
to node-level importances suitable for visualization.

Strategy:
- Run the model once while capturing the inputs to each GAT/GATv2 conv
  module (using forward pre-hooks). This gives the features and
  edge_index at the exact module invocation point.
- Re-run each captured conv module separately with
  `return_attention_weights=True` where supported to obtain edge-level
  attention coefficients. If the conv does not support that keyword,
  we attempt a few fallbacks but will gracefully skip modules we
  cannot probe.
- Aggregate edge attention into node importances by summing attention
  incoming to each destination node and pooling across heads.

The explainer is intentionally defensive — it should work across a
range of torch_geometric versions and both `GATConv` and `GATv2Conv`.
"""

from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data


def _register_capture_hooks(model: nn.Module):
    """Register forward-pre hooks on all GATConv/GATv2Conv modules.

    Returns a tuple (hooks, captured) where `captured` is a list that
    will be appended with tuples (module, inputs) when the hooks run.
    """
    captured: List[Tuple[nn.Module, Tuple]] = []
    hooks = []

    def make_hook(m):
        def hook(module, inputs):
            # inputs is a tuple; typically (x, edge_index, *args)
            captured.append((module, inputs))
        return hook

    for _, m in model.named_modules():
        if isinstance(m, (GATConv, GATv2Conv)):
            hooks.append(m.register_forward_pre_hook(make_hook(m)))

    return hooks, captured


def _remove_hooks(hooks):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


def _call_conv_for_attention(conv, *inputs):
    """Call a conv module attempting to retrieve attention weights.

    Returns a tuple (edge_index, attn) or (None, None) on failure.
    `attn` should be a Tensor of shape [num_edges, heads].
    """
    # Try the modern API: return_attention_weights=True
    try:
        res = conv(*inputs, return_attention_weights=True)
        # Many PyG versions return (out, (edge_index, attn_weights))
        if isinstance(res, tuple) and len(res) >= 2:
            second = res[1]
            # second may be (edge_index, attn) or attn itself
            if isinstance(second, tuple) and len(second) >= 2:
                edge_idx, attn = second[0], second[1]
            else:
                # some versions return (out, attn)
                attn = second
                # attempt to reuse inputs[1] as edge_index
                edge_idx = inputs[1] if len(inputs) > 1 else None
            return edge_idx, attn
    except TypeError:
        # return_attention_weights not supported in this conv signature
        pass
    except Exception:
        # any other error -> fallback below
        pass

    # Fallback: call conv normally and inspect known attributes
    try:
        _ = conv(*inputs)
    except Exception:
        # If we can't run it standalone, give up for this module
        return None, None

    # Inspect common attribute names for stored attention coefficients
    for attr in ("alpha", "attn", "_alpha", "_attn", "edge_attention"):
        att = getattr(conv, attr, None)
        if isinstance(att, torch.Tensor):
            # no edge_index information — try to use inputs[1]
            edge_idx = inputs[1] if len(inputs) > 1 else None
            return edge_idx, att

    return None, None


def _edge_attention_to_node_importance(edge_index: torch.Tensor, attn: torch.Tensor, num_nodes: int, head_pool: str = "mean") -> torch.Tensor:
    """Aggregate edge-level attention to node-level importances.

    edge_index: Tensor[2, E]
    attn: Tensor[E, H] or [E]
    Returns: Tensor[num_nodes]
    """
    if edge_index is None or attn is None:
        return torch.zeros(num_nodes)

    # Normalize shapes: attn -> [E, H]
    if attn.dim() == 1:
        attn = attn.unsqueeze(1)

    # Pool across heads
    if head_pool == "mean":
        edge_scores = attn.mean(dim=1)
    elif head_pool == "sum":
        edge_scores = attn.sum(dim=1)
    elif head_pool == "max":
        edge_scores, _ = attn.max(dim=1)
    else:
        raise ValueError("head_pool must be one of 'mean','sum','max'")

    # edge_index[1] is the destination node for each attention coefficient
    targets = edge_index[1].to(edge_scores.device)
    node_scores = torch.zeros(num_nodes, device=edge_scores.device)
    node_scores = node_scores.scatter_add(0, targets, edge_scores)

    # Move to CPU and return
    return node_scores.detach().cpu()


def attention_node_importance(model: nn.Module, data: Data, head_pool: str = "mean", normalize: bool = True) -> Dict[str, torch.Tensor]:
    """Compute node importance scores from attention layers in `model` for `data`.

    Returns a dict with:
    - "per_module": list of (module_name, node_scores Tensor[num_nodes])
    - "aggregated": Tensor[num_nodes] (sum across modules)
    """
    device = next(model.parameters()).device if any(p.numel() for p in model.parameters()) else torch.device("cpu")
    model = model.eval()
    data = data.clone()
    data = data.to(device)

    # 1) Capture inputs to conv modules during a forward pass
    hooks, captured = _register_capture_hooks(model)
    try:
        # Run the model once to populate captured inputs
        with torch.no_grad():
            _ = model(data)
    finally:
        _remove_hooks(hooks)

    num_nodes = data.num_nodes
    per_module: List[Tuple[str, torch.Tensor]] = []

    for module, inputs in captured:
        # inputs usually: (x, edge_index, *args)
        # We'll only forward the first two
        x_in = inputs[0]
        edge_index_in = inputs[1] if len(inputs) > 1 else None

        # Convert to device
        try:
            x_dev = x_in.to(device)
        except Exception:
            x_dev = x_in
        try:
            ei_dev = edge_index_in.to(device) if edge_index_in is not None else None
        except Exception:
            ei_dev = edge_index_in

        edge_idx, attn = _call_conv_for_attention(module, x_dev, ei_dev)
        if attn is None or edge_idx is None:
            # Skip modules where we couldn't extract attention
            continue

        # Ensure edge_idx is Tensor[2, E]
        if isinstance(edge_idx, tuple) or isinstance(edge_idx, list):
            edge_idx = edge_idx[0]

        node_scores = _edge_attention_to_node_importance(edge_idx, attn, num_nodes, head_pool=head_pool)
        per_module.append((module.__class__.__name__, node_scores))

    # Aggregate across modules
    if per_module:
        agg = sum(t for _, t in per_module)
    else:
        agg = torch.zeros(num_nodes)

    if normalize:
        def _norm(v: torch.Tensor):
            v = v - float(v.min())
            if float(v.max()) > 0:
                v = v / float(v.max())
            return v

        per_module = [(n, _norm(t)) for n, t in per_module]
        agg = _norm(agg)

    return {"per_module": per_module, "aggregated": agg}


def visualize_node_importances(data: Data, scores: torch.Tensor, title: Optional[str] = None):
    """Simple networkx/matplotlib visualization for node importances.

    Accepts a torch_geometric `Data` object with `edge_index` and `num_nodes`.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        raise RuntimeError("Matplotlib and networkx required for visualization")

    edge_index = data.edge_index.cpu()
    edges = edge_index.t().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)
    plt.figure()
    sc = nx.draw_networkx_nodes(G, pos, node_color=scores.numpy(), cmap="viridis", node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.colorbar(sc, label="importance")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
