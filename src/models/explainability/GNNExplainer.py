"""Lightweight wrapper for a GNNExplainer-style interface.

The file provides a minimal, robust helper that will use PyG's
GNNExplainer if available; otherwise it falls back to a simple
gradient-based node saliency as a pragmatic default for this repo.

Designed to work with the project's models which accept a
torch_geometric.data.Data object and return a graph-level tensor.
"""
from typing import Optional, Dict, Any

import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch_geometric.utils import to_networkx


def explain_with_gnnexplainer(
	model: torch.nn.Module,
	data: Data,
	node_idx: Optional[int] = None,
	target: Optional[int] = None,
	epochs: int = 100,
	lr: float = 0.01,
) -> Dict[str, Any]:
	"""Explain a model prediction using a GNNExplainer when available.

	Fallback: gradient-based node saliency (absolute-sum of gradients
	of the chosen output w.r.t. node features).

	Returns a dictionary with keys `node_mask` (Tensor[num_nodes]) and
	`edge_mask` (or None when unavailable).
	"""

	# Try to use PyG's GNNExplainer if installed (different versions expose
	# it in different modules). If it's not available, we'll compute a
	# lightweight gradient-based saliency as a fallback.
	PyGGNNExplainer = None
	try:
		# new-style location
		from torch_geometric.explain import GNNExplainer as PyGGNNExplainer
	except Exception:
		try:
			# older style location
			from torch_geometric.nn.models import GNNExplainer as PyGGNNExplainer
		except Exception:
			PyGGNNExplainer = None

	if PyGGNNExplainer is not None:
		# Different torch_geometric releases expose different signatures for
		# GNNExplainer. Try to use it, but on any unexpected API we'll quietly
		# fall back to the gradient-based saliency below.
		try:
			explainer = PyGGNNExplainer(model)

			# Try node vs graph explanation API with graceful fallbacks
			if node_idx is None and hasattr(explainer, "explain_graph"):
				result = explainer.explain_graph(data)
			elif node_idx is not None and hasattr(explainer, "explain_node"):
				result = explainer.explain_node(node_idx, data)
			else:
				# Try alternate call signatures used by other PyG versions
				if node_idx is None and hasattr(explainer, "explain_graph"):
					result = explainer.explain_graph(data.x, data.edge_index)
				elif node_idx is not None and hasattr(explainer, "explain_node"):
					result = explainer.explain_node(node_idx, data.x, data.edge_index)
				else:
					# If we can't find a usable API, raise to trigger fallback
					raise RuntimeError("GNNExplainer missing expected API methods")

			# `result` shape differs by implementation. Normalize to node_mask/edge_mask.
			if isinstance(result, tuple) and len(result) >= 1:
				# many versions return (node_mask, edge_mask) or ((node_mask, edge_mask), ...)
				if len(result) >= 2:
					node_mask, edge_mask = result[0], result[1]
				else:
					node_mask = result[0]
					edge_mask = None
			else:
				# unknown structure, return raw result under node_mask
				node_mask, edge_mask = result, None

			return {"node_mask": node_mask, "edge_mask": edge_mask}
		except Exception:
			# Ignore and fall back to gradient-saliency implementation below
			pass

	# Fallback: gradient-based node saliency.
	device = next(model.parameters()).device
	model.eval()

	x = data.x.clone().detach().to(device).requires_grad_(True)
	dcopy = data.clone()
	dcopy.x = x

	out = model(dcopy)
	if out is None:
		raise RuntimeError("Model returned None on provided Data input.")

	# Choose a scalar score to backprop through. If model returns [B, C],
	# use `target` when provided, otherwise sum everything.
	if out.dim() == 1:
		score = out.sum()
	else:
		if target is None:
			score = out.sum()
		else:
			score = out[:, target].sum()

	model.zero_grad()
	score.backward()

	grads = x.grad  # [num_nodes, in_channels]
	node_mask = grads.abs().sum(dim=1).detach().cpu()

	return {"node_mask": node_mask, "edge_mask": None}

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
