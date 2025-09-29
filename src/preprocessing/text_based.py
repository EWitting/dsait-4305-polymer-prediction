from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import itertools

class TextBasedPreprocessor:
    """Converts SMILES into graph that PyG accepts.

    Does not parse the SMILES string into proper molecular graph.
    Just reads the string as a chain of characters.
    So it encodes also characters like parentheses etc. as nodes. 
    Each connected to only the one before and after
    
    """
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, smiles: list[str]) -> None:
        """Fit the one hot encoder on the training data."""
        print(f'Fitting one hot encoder on {len(smiles)} SMILES strings...')
        character_list = list(itertools.chain(*smiles)) # Concat then split all characters: ["abc","ab"] -> ["a","b","c","a","b"]
        self.one_hot_encoder.fit(np.array(character_list).reshape(-1, 1)) # Assign each unique character an index
        print(f'Done fitting one hot encoder, number of categories: {len(self.one_hot_encoder.get_feature_names_out())}.')

    def fit_transform(self, smiles: list[str]) -> list[Data]:
        """Fit the one hot encoder on the training data and transform it."""
        self.fit(smiles)
        return self.transform(smiles)

    def transform(self, smiles: list[str]) -> list[Data]:
        """Transform the SMILES string into a PyG Data object."""
        print(f'Transforming {len(smiles)} SMILES strings...')
        return [self._transform_single(smile) for smile in tqdm(smiles)]

    def _transform_single(self, smiles: str) -> Data:
        """Transform a single SMILES string into a PyG Data object."""

        # Turn each character into a one hot encoded vector
        x = self.one_hot_encoder.transform(np.array(list(smiles)).reshape(-1, 1)) # Shape (num_characters, encoding_size)
        x = torch.tensor(x, dtype=torch.float32)

        # Create edge list where each character is connected to the next one (and back because undirected)
        node_indices = range(len(smiles))
        edge_pairs = [(a,b) for a,b in zip(node_indices[:-1], node_indices[1:])] # Forward edges
        edge_pairs.extend([(b,a) for a,b in edge_pairs]) # Backward edges
        edge_index = torch.tensor(edge_pairs).T.contiguous() # PyG format, see https://pytorch-geometric.readthedocs.io/en/2.5.1/get_started/introduction.html 

        return Data(x=x, edge_index=edge_index, num_nodes=len(smiles))

