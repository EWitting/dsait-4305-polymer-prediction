import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
import numpy as np


class RDKitDescriptorsPreprocessor:
    """
    Preprocessor that converts SMILES strings to graph-level molecular descriptors using RDKit.
    
    Computes all available molecular descriptors from RDKit.Descriptors module
    and returns flat tensors suitable for non-graph ML models.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._descriptor_names = None
        self._computed_dim = None
    
    def _initialize_descriptors(self):
        """Initialize the descriptor names if not already done."""
        if self._descriptor_names is None:
            self._descriptor_names = [
                (name, func) for name, func in Descriptors._descList
            ]
    
    def _compute_descriptors(self, smiles_list, desc="Computing descriptors"):
        """
        Compute descriptors for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            desc: Description for progress bar
            
        Returns:
            numpy array of descriptor matrices
        """
        self._initialize_descriptors()
        num_descriptors = len(self._descriptor_names)
        
        descriptor_matrix = []
        for smiles in tqdm(smiles_list, desc=desc):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    warnings.warn(f"Could not parse SMILES: {smiles}")
                    descriptor_matrix.append(np.zeros(num_descriptors))
                    continue
                
                descriptors = [func(mol) for name, func in self._descriptor_names]
                descriptor_matrix.append(descriptors)
            except Exception as e:
                warnings.warn(f"Error computing descriptors for {smiles}: {e}")
                descriptor_matrix.append(np.zeros(num_descriptors))
        
        descriptor_matrix = np.array(descriptor_matrix)
        
        # Handle infinity and NaN values
        descriptor_matrix = np.where(np.isinf(descriptor_matrix), np.nan, descriptor_matrix)
        descriptor_matrix = np.nan_to_num(descriptor_matrix, nan=0.0)
        
        return descriptor_matrix
    
    def fit(self, smiles_list):
        """
        Fit the scaler on computed descriptors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            self
        """
        print(f'Computing descriptors and fitting scaler on {len(smiles_list)} molecules...')
        
        descriptor_matrix = self._compute_descriptors(smiles_list, desc="Computing descriptors")
        self._computed_dim = descriptor_matrix.shape[1]
        
        # Fit the scaler
        self.scaler.fit(descriptor_matrix)
        print(f'Fitted scaler on {self._computed_dim} descriptors')
        
        return self
    
    def fit_transform(self, smiles_list):
        """Fit and transform SMILES to descriptors."""
        print(f'Computing descriptors and fitting scaler on {len(smiles_list)} molecules...')
        
        descriptor_matrix = self._compute_descriptors(smiles_list, desc="Computing descriptors")
        self._computed_dim = descriptor_matrix.shape[1]
        
        # Fit the scaler on the same data we'll transform
        self.scaler.fit(descriptor_matrix)
        print(f'Fitted scaler on {self._computed_dim} descriptors')
        
        # Transform using the same descriptor_matrix
        scaled_matrix = self.scaler.transform(descriptor_matrix)
        
        return torch.tensor(scaled_matrix, dtype=torch.float32)
    
    def transform(self, smiles_list):
        """
        Transform SMILES strings to scaled descriptor tensors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of torch.Tensor (one per molecule)
        """
        print(f'Transforming {len(smiles_list)} SMILES strings to descriptors...')
        
        descriptor_matrix = self._compute_descriptors(smiles_list, desc="Transforming to descriptors")
        
        # Scale the descriptors
        scaled_matrix = self.scaler.transform(descriptor_matrix)
        
        return torch.tensor(scaled_matrix, dtype=torch.float32)
    
    def get_descriptor_names(self):
        """Return the names of all descriptors being computed."""
        self._initialize_descriptors()
        return [name for name, func in self._descriptor_names]
