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

# Visualization / utility imports used by functions below
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from torch_geometric.utils import to_networkx


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


def visualize_node_importance(
	data: Data,
	node_imp,
	title: Optional[str] = None,
	smiles: Optional[str] = None,
	atom_symbols: Optional[list] = None,
):
	"""Visualize per-node importances on the molecular graph.

	Parameters
	- data: torch_geometric.data.Data graph for a single molecule/graph
	- node_imp: Tensor or array-like of shape [num_nodes] with per-node importance scores
	- title: optional plot title
	- smiles: optional SMILES string to attempt RDKit-based 2D coords and atom labels
	- atom_symbols: optional list of atom symbol strings to use as labels

	The function is robust: it tries RDKit for coordinates/labels if SMILES is provided
	(or if `data` has `.smiles`), otherwise falls back to a spring layout. Node importances
	are normalized and displayed as node colors with a colorbar.
	"""
	# Convert node_imp to numpy
	try:
		import numpy as _np

		if torch.is_tensor(node_imp):
			node_vals = node_imp.detach().cpu().numpy()
		else:
			node_vals = _np.asarray(node_imp)
	except Exception:
		# Last resort: try to coerce
		node_vals = node_imp

	# Build networkx graph (undirected for visualization)
	G = to_networkx(data, to_undirected=True)

	# Attempt RDKit positions and labels if smiles provided or available on data
	pos = None
	labels = None
	rdkit_smiles = smiles
	if rdkit_smiles is None and hasattr(data, "smiles"):
		try:
			rdkit_smiles = data.smiles
		except Exception:
			rdkit_smiles = None

	if rdkit_smiles is not None:
		try:
			from rdkit import Chem
			from rdkit.Chem import AllChem

			mol = Chem.MolFromSmiles(str(rdkit_smiles))
			if mol is not None:
				AllChem.Compute2DCoords(mol)
				conf = mol.GetConformer()
				rdkit_pos = {atom.GetIdx(): (conf.GetAtomPosition(atom.GetIdx()).x, conf.GetAtomPosition(atom.GetIdx()).y) for atom in mol.GetAtoms()}
				rdkit_labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
				if len(rdkit_pos) == G.number_of_nodes():
					pos = rdkit_pos
					labels = rdkit_labels
		except Exception:
			pos = None
			labels = None

	if pos is None:
		pos = nx.spring_layout(G)

	# Determine labels: priority: explicit atom_symbols arg -> data.atom_symbols -> rdkit labels -> numeric indices
	if atom_symbols is not None:
		labels = {i: str(s) for i, s in enumerate(atom_symbols)}
	elif labels is None:
		if hasattr(data, "atom_symbols"):
			try:
				sym_list = list(data.atom_symbols)
				if len(sym_list) == G.number_of_nodes():
					labels = {i: str(sym_list[i]) for i in range(len(sym_list))}
			except Exception:
				labels = None

	if labels is None:
		labels = {i: str(i) for i in range(G.number_of_nodes())}

	# Plot
	fig, ax = plt.subplots(figsize=(10, 6))
	# ensure node_vals length matches
	try:
		node_plot_vals = node_vals
		if len(node_plot_vals) != G.number_of_nodes():
			# fallback: try to trim or pad with zeros
			import numpy as _np

			if len(node_plot_vals) > G.number_of_nodes():
				node_plot_vals = node_plot_vals[: G.number_of_nodes()]
			else:
				pad = _np.zeros(G.number_of_nodes() - len(node_plot_vals))
				node_plot_vals = _np.concatenate([node_plot_vals, pad])
	except Exception:
		node_plot_vals = None

	cmap = mpl.cm.plasma
	if node_plot_vals is None:
		# draw uncoloured graph if we can't get values
		nx.draw(G, pos=pos, with_labels=False, ax=ax)
	else:
		nc = nx.draw_networkx_nodes(G, pos=pos, node_color=node_plot_vals, cmap=cmap, ax=ax)
		nx.draw_networkx_edges(G, pos=pos, ax=ax)
		# overlay labels
		nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax, font_size=8)
		# colorbar
		norm = mpl.colors.Normalize(vmin=float(node_plot_vals.min()), vmax=float(node_plot_vals.max()))
		mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
		try:
			mappable.set_array(node_plot_vals)
		except Exception:
			# Matplotlib older versions may require different handling
			mappable.set_array([float(x) for x in node_plot_vals])
		fig.colorbar(mappable, ax=ax)

	if title:
		ax.set_title(title)
	plt.tight_layout()
	plt.show()
    

def extract_node_importance_from_forward(model: nn.Module, data: Data, head_pool: str = "mean", normalize: bool = True):
    """Try to extract attention returned from a model forward pass and
    convert it to node-level importances.

    Returns the same structure as `attention_node_importance`: a dict with
    - "per_layer": list of (layer_name, Tensor[num_nodes])
    - "aggregated": Tensor[num_nodes]

    If the model forward does not return attention, returns None.
    """
    # Attempt to call model(...) with a standard attention-returning kwarg
    attn_list = None
    try:
        res = model(data, return_attn=True)
        if isinstance(res, tuple) and len(res) == 2:
            _, attn_list = res
    except TypeError:
        # Some models expect forward(...) explicitly
        try:
            res = model.forward(data, return_attn=True)
            if isinstance(res, tuple) and len(res) == 2:
                _, attn_list = res
        except Exception:
            attn_list = None
    except Exception:
        attn_list = None

    if not attn_list:
        return None

    # Ensure data is not moved by the caller
    num_nodes = int(data.num_nodes)

    def _norm(t: torch.Tensor) -> torch.Tensor:
        t = t - float(t.min())
        if float(t.max()) > 0:
            t = t / float(t.max())
        return t

    per_layer = []
    agg = None
    for idx, (edge_idx, attn_weights) in enumerate(attn_list):
        # resolve edge_index (use data.edge_index if None)
        if edge_idx is None:
            edge_index = data.edge_index
        else:
            # some APIs return (edge_index, ...)
            if isinstance(edge_idx, (tuple, list)):
                edge_index = edge_idx[0]
            else:
                edge_index = edge_idx

        # Normalize attn shapes to [E, H] or [E,]
        a = attn_weights
        # If attn is on GPU, move edge_index to same device in helper below

        node_scores = _edge_attention_to_node_importance(edge_index, a, num_nodes, head_pool=head_pool)
        if normalize:
            node_scores = _norm(node_scores)

        layer_name = f"forward_attn_layer_{idx}"
        per_layer.append((layer_name, node_scores))
        agg = node_scores if agg is None else agg + node_scores

    if agg is None:
        agg = torch.zeros(num_nodes)
    if normalize:
        agg = _norm(agg)

    return {"per_layer": per_layer, "aggregated": agg}
