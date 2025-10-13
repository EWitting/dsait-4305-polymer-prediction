# This file is more a placeholder where the content was taken from what Miguel implemented in jupyternotebook
import warnings
from rdkit import Chem
from torch_geometric.utils import from_smiles
from sklearn.base import BaseEstimator, TransformerMixin
import torch


# Finds an atom that is bonded to the "*" (connection points)
def _neighbor_idx(mol, atom_idx):
    neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
    if len(neighbors) != 1:
        raise ValueError(
            f"Star atom at index {atom_idx} does not have exactly one neighbor."
        )
    return neighbors[0]


def smiles_to_oligomer(monomer_smiles: str, length: int, cap_ends: bool = True) -> str:
    """
    Builds an oligomer SMILES string by connecting monomer units at '*' with single bonds.

    Args:
        monomer_smiles (str): The SMILES string of the monomer, containing two '*' for head/tail connection.
        length (int): The number of monomer units in the final oligomer.
        cap_ends (bool): If True, removes the terminal '*' atoms from the final structure.

    Returns:
        str: The SMILES string of the resulting oligomer.
    """
    if monomer_smiles.count("*") != 2:
        raise ValueError(
            "Monomer SMILES must have exactly two connection points denoted by '*'."
        )

    # Creates an RDKit molecule object
    monomer_mol = Chem.MolFromSmiles(monomer_smiles)
    if monomer_mol is None:
        raise ValueError(f"Invalid monomer SMILES string provided: {monomer_smiles}")

    # Find the indices of the star atoms, which mark connection points (there should be 2)
    star_atoms = [
        atom.GetIdx() for atom in monomer_mol.GetAtoms() if atom.GetSymbol() == "*"
    ]

    # Initialize the oligomer with the first monomer unit
    oligomer_mol = Chem.Mol(monomer_mol)
    # The tail of the growing chain is the index of the second star atom
    open_end_idx = star_atoms[1]

    # Iteratively add new monomer units for the desired length
    for _ in range(length - 1):
        new_monomer_mol = Chem.Mol(monomer_mol)
        new_star_atoms = [
            atom.GetIdx()
            for atom in new_monomer_mol.GetAtoms()
            if atom.GetSymbol() == "*"
        ]

        # Combine the existing oligomer with the new monomer
        combined_mol = Chem.CombineMols(oligomer_mol, new_monomer_mol)
        rw_mol = Chem.RWMol(combined_mol)

        # Calculate the index offset for atoms from the new monomer
        offset = oligomer_mol.GetNumAtoms()

        # Get indices of the connection points in the combined molecule
        old_tail_star_idx = open_end_idx
        new_head_star_idx = new_star_atoms[0] + offset
        new_tail_star_idx = new_star_atoms[1] + offset

        # Connect the neighbor of the old tail to the neighbor of the new head
        neighbor1_idx = _neighbor_idx(rw_mol, old_tail_star_idx)
        neighbor2_idx = _neighbor_idx(rw_mol, new_head_star_idx)
        rw_mol.AddBond(neighbor1_idx, neighbor2_idx, Chem.BondType.SINGLE)

        # Remove the two star atoms that have just been connected
        atoms_to_remove = sorted([old_tail_star_idx, new_head_star_idx], reverse=True)
        for idx in atoms_to_remove:
            rw_mol.RemoveAtom(idx)

        # The new molecule becomes the current oligomer for the next iteration
        oligomer_mol = rw_mol.GetMol()

        # Update the index of the new open tail, adjusting for the removed atoms
        final_tail_idx = new_tail_star_idx
        for idx in atoms_to_remove:
            if final_tail_idx > idx:
                final_tail_idx -= 1
        open_end_idx = final_tail_idx

    # If requested, remove the two stars at both ends of the polymers.
    if cap_ends:
        rw_mol = Chem.RWMol(oligomer_mol)
        final_star_atoms = [
            atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "*"
        ]
        for idx in sorted(final_star_atoms, reverse=True):
            rw_mol.RemoveAtom(idx)
        oligomer_mol = rw_mol.GetMol()

    # Clean up the molecule's valences and properties
    Chem.SanitizeMol(oligomer_mol)
    return Chem.MolToSmiles(oligomer_mol)


# Converts a list of SMILES string into a graph structure
class OligomerPreprocessor:
    def __init__(self, oligomer_len: int = 1):
        self.oligomer_len = oligomer_len

    def fit(self, X, y=None):
        return self

    def transform(self, smiles_list: list[str]):
        graphs = []
        # for each one of the SMILES strings in the input
        for smiles in smiles_list:
            processed_smiles = smiles

            if self.oligomer_len > 1 and smiles.count("*") == 2:
                try:
                    # apply polymerization to create a longer SMILES string.
                    processed_smiles = smiles_to_oligomer(smiles, self.oligomer_len)
                except Exception as e:
                    warnings.warn(
                        f"Could not generate oligomer for '{smiles}': {e}. Using original monomer."
                    )

            # Convert the final SMILES string to a graph Data object using RDKit
            try:
                graph = from_smiles(processed_smiles)
                graph.x = graph.x.float()
                graphs.append(graph)
            except Exception as e:
                warnings.warn(
                    f"Could not create graph for SMILES '{processed_smiles}': {e}. Skipping."
                )

        return graphs

    def fit_transform(self, smiles_list: list[str], y=None):
        self.fit(smiles_list, y)
        return self.transform(smiles_list)


# We need to normalize the node features (edge features are bond types which are one-hot encoded
# so no need for normalization there I believe)
class GraphFeatureNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    # Calcualtes the mean and std in
    def fit(self, graph_list: list, y=None):
        # Concatenate all node features from all graphs into one big tensor
        all_node_features = torch.cat([data.x for data in graph_list], dim=0)

        # Calculate mean and std
        self.mean = torch.mean(all_node_features, dim=0)
        self.std = torch.std(all_node_features, dim=0)

        # To avoid division by zero for features with no variance
        self.std[self.std == 0] = 1.0

        return self

    # Applies standardization using the mean and std calculated
    def transform(self, graph_list: list):
        if self.mean is None or self.std is None:
            raise RuntimeError("You must fit the normalizer before transforming data.")

        transformed_graphs = []
        for data in graph_list:
            # Create a copy to avoid modifying the original graph object
            new_data = data.clone()
            new_data.x = (data.x - self.mean) / self.std
            transformed_graphs.append(new_data)

        return transformed_graphs
