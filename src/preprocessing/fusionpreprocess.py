import torch
from torch_geometric.data import Batch
from transformers import AutoTokenizer
from src.preprocessing.polyBERTpreprocess import PolyBERTPreprocessor 
from src.preprocessing.oligomer import OligomerPreprocessor      
import numpy as np
import pandas as pd

class FusionPreprocessor:
    """
    Preprocessor for the FusionPolyBERTGraph model.
    Combines PolyBERT SMILES tokenization with graph featurization.
    """

    def __init__(
        self,
        polybert_model: str = "kuelumbus/polybert",
        max_length: int = 256,
        graph_preprocessor: OligomerPreprocessor = None
    ):
        # PolyBERT tokenizer
        self.polybert_preprocessor = PolyBERTPreprocessor(model_name=polybert_model, max_length=max_length)
        
        # Graph preprocessor (can reuse your existing one)
        self.graph_preprocessor = graph_preprocessor or OligomerPreprocessor()

    def fit(self, smiles_list):
        """
        Fit the graph preprocessor if needed (e.g., atom feature scaling).
        PolyBERT does not require fitting.
        """
        self.graph_preprocessor.fit(smiles_list)

    def transform(self, smiles_list):
        """
        Transform SMILES strings into both PolyBERT inputs and graph objects.
        """
        if isinstance(smiles_list, (np.ndarray, pd.Series)):
            smiles_list = smiles_list.tolist()
        smiles_list = [str(s) for s in smiles_list]
        # Tokenize for PolyBERT
        polybert_encodings = self.polybert_preprocessor.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.polybert_preprocessor.max_length,
            return_tensors='pt'
        )

        # Generate graph representations
        graph_data_list = self.graph_preprocessor.transform(smiles_list)

        return {
            "input_ids": polybert_encodings["input_ids"],
            "attention_mask": polybert_encodings["attention_mask"],
            "graph_data": graph_data_list
        }

    def fit_transform(self, smiles_list):
        """
        Convenience method to fit and transform in one go.
        """
        self.fit(smiles_list)
        return self.transform(smiles_list)
    
def collate_fusion_fn(batch):
        """
        Custom collate function for FusionPolyBERTGraph model.
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        # Combine graphs into a PyG batch
        graph_data_list = [item["graph_data"] for item in batch]
        graph_batch = Batch.from_data_list(graph_data_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_data": graph_batch,
            "labels": labels
        }