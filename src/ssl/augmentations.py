import torch
import random
from torch_geometric.data import Data


def drop_nodes(data: Data, p: float) -> Data:
    """
    Randomly drops nodes (and their edges) from the graph.
    """
    if p == 0.0:
        return data

    num_nodes = data.num_nodes
    num_to_keep = int(num_nodes * (1 - p))

    # Indices of nodes to keep
    keep_indices = torch.randperm(num_nodes)[:num_to_keep]

    # The subgraph method automatically re-indexes edges
    return data.subgraph(keep_indices)


def mask_features(data: Data, p: float) -> Data:
    """
    Randomly masks node features by setting them to zero.
    """
    if p == 0.0:
        return data

    num_nodes, _ = data.x.shape
    num_to_mask = int(num_nodes * p)

    # Create a copy to avoid modifying the original
    new_data = data.clone()

    # Randomly select nodes to mask
    mask_indices = torch.randperm(num_nodes)[:num_to_mask]

    # Set features to zero
    new_data.x[mask_indices] = 0.0
    return new_data


def drop_edges(data: Data, p: float) -> Data:
    """
    Randomly drops edges from the graph.
    """
    if p == 0.0:
        return data

    num_edges = data.num_edges
    num_to_keep = int(num_edges * (1 - p))

    # Indices of edges to keep
    keep_indices = torch.randperm(num_edges)[:num_to_keep]

    new_data = data.clone()
    new_data.edge_index = data.edge_index[:, keep_indices]

    # Also filter edge_attr if it exists
    if data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr[keep_indices]

    return new_data


def augment(
    data: Data, drop_node_p: float, mask_feat_p: float, drop_edge_p: float
) -> Data:
    """
    Applies a sequence of augmentations.
    The randomness is handled within each function.
    """
    # We apply augmentations in sequence
    # The order can be a new hyperparameter, but this is a solid default

    data_aug = data
    data_aug = drop_nodes(data_aug, drop_node_p)
    data_aug = drop_edges(data_aug, drop_edge_p)
    data_aug = mask_features(data_aug, mask_feat_p)

    return data_aug


def get_augmentations(
    data: Data, drop_node_p: float, mask_feat_p: float, drop_edge_p: float
):
    """
    Returns two *different*, *randomly* augmented versions of the input graph.

    Now, both aug1 and aug2 are created by the same 'augment' process,
    which is a more robust way to create positive pairs.
    """
    aug1 = augment(data, drop_node_p, mask_feat_p, drop_edge_p)
    aug2 = augment(data, drop_node_p, mask_feat_p, drop_edge_p)

    # We need to pass all three probabilities from the config now.
    # Your Lightning model's 'training_step' will need to be updated.

    return aug1, aug2
