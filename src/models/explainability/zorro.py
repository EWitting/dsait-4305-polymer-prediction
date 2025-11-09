"""Simple, model-agnostic Zorro-like explainer (perturbation-based).

This file implements a minimal Zorro-style node importance routine suited
for quick experiments and tests: for each node we mask its features (zero)
and measure the change in the model's graph-level score. The magnitude of
the change is used as the node importance.

This is intentionally simple and slow (O(num_nodes) forward passes). It is
meant as a readable baseline and is compatible with the repo's models.
"""
from typing import Optional

import torch
from torch_geometric.data import Data


def zorro_node_importance(
	model: torch.nn.Module,
	data: Data,
	target: Optional[int] = None,
	device: Optional[torch.device] = None,
) -> torch.Tensor:
	"""Compute node importances by per-node feature masking.

	For each node i we zero its feature vector, run the model, and compute
	the absolute drop in the graph-level target score for the graph that
	contains node i. Returns a Tensor[num_nodes] on the CPU.
	"""

	if not hasattr(data, "x") or data.x is None:
		raise ValueError("data.x is required for zorro_node_importance")
	if not hasattr(data, "batch") or data.batch is None:
		raise ValueError("data.batch is required for per-graph aggregation")

	device = device or next(model.parameters()).device
	model.eval()

	# Work on a clone so we don't mutate caller data
	data = data.clone()
	data.x = data.x.to(device)
	data.batch = data.batch.to(device)

	with torch.no_grad():
		base_out = model(data)

	if base_out is None:
		raise RuntimeError("Model returned None on provided Data input.")

	# compute baseline per-graph scalar score
	if base_out.dim() == 1:
		base_score = base_out
	else:
		base_score = base_out[:, target] if target is not None else base_out.sum(dim=1)

	num_nodes = data.x.shape[0]
	importances = torch.zeros(num_nodes, dtype=torch.float32, device="cpu")

	# Iterate nodes and measure impact of zeroing their features.
	for i in range(num_nodes):
		dcopy = data.clone()
		x_masked = dcopy.x.clone()
		x_masked[i] = 0.0
		dcopy.x = x_masked

		with torch.no_grad():
			out = model(dcopy)

		if out.dim() == 1:
			out_score = out
		else:
			out_score = out[:, target] if target is not None else out.sum(dim=1)

		g_idx = int(dcopy.batch[i].item())
		# magnitude of change for the containing graph
		importance = float((base_score[g_idx] - out_score[g_idx]).abs().item())
		importances[i] = importance

	# Normalize to [0,1] for convenience (if all zeros, leave as zero)
	maxv = float(importances.max().item())
	if maxv > 0:
		importances = importances / maxv

	return importances

