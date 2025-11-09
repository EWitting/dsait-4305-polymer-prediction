import torch
import random


def drop_nodes(data, p=0.2):
    num_nodes = data.num_nodes
    num_to_drop = int(num_nodes * p)
    drop_mask = torch.ones(num_nodes, dtype=torch.bool)

    # Randomly select nodes to drop
    drop_indices = torch.randperm(num_nodes)[:num_to_drop]
    drop_mask[drop_indices] = False

    # Subgraph the graph to keep only the non-dropped nodes
    return data.subgraph(drop_mask)


def mask_features(data, p=0.3):
    num_nodes, _ = data.x.shape
    num_to_mask = int(num_nodes * p)

    # Create a copy to avoid modifying the original
    new_data = data.clone()

    # Randomly select nodes to mask
    mask_indices = torch.randperm(num_nodes)[:num_to_mask]

    # Set features to zero (or a learned [MASK] token)
    new_data.x[mask_indices] = 0.0
    return new_data


# returns two augmented versions of the input graph data (one with dropped nodes, one with masked features)
def get_augmentations(data, drop_p: float, mask_p: float):
    aug1 = drop_nodes(data, p=drop_p)
    aug2 = mask_features(data, p=mask_p)
    return aug1, aug2
