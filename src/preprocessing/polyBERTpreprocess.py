import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import lightning.pytorch as pl
import numpy as np
import pandas as pd

class PolyBERTPreprocessor:
    """Simple preprocessor using PolyBERT tokenizer."""
    
    def __init__(self, model_name: str = "kuelumbus/polybert", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def fit(self, smiles_list):
        """No fitting needed for pretrained tokenizer."""
        pass
    
    def fit_transform(self, smiles_list):
        """Fit and transform SMILES strings."""
        return self.transform(smiles_list)
    
    def transform(self, smiles_list):
        """Transform SMILES strings using PolyBERT tokenizer."""
        # Convert to list of str if it's a NumPy array or Series
        if isinstance(smiles_list, (np.ndarray, pd.Series)):
            smiles_list = smiles_list.tolist()
        smiles_list = [str(s) for s in smiles_list]

        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }