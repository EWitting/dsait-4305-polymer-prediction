import torch
import torch.nn as nn
from torch import Tensor
import lightning.pytorch as pl
from transformers import AutoModel
from torch_geometric.nn import GCNConv, global_mean_pool
from src.loss.masked_loss import WeightedMAELoss  # your custom loss


class FusionPolyBERTGraph(pl.LightningModule):
    """
    Fusion model combining PolyBERT (text/SMILES) embeddings with a graph encoder.
    """

    def __init__(
        self,
        model_name: str = "kuelumbus/polybert",
        num_node_features: int = 9,
        num_labels: int = 5,
        graph_model_type: str = "gcn",
        graph_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        freeze_polybert: bool = True,
        property_ranges: Tensor = None,
        num_samples_per_property: Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # === PolyBERT encoder ===
        self.polybert = AutoModel.from_pretrained(model_name)
        if freeze_polybert:
            for param in self.polybert.parameters():
                param.requires_grad = False
        polybert_dim = self.polybert.config.hidden_size

        # === Graph encoder ===

        self.graph_encoder = nn.Sequential(
            GCNConv(num_node_features, graph_dim),
            nn.ReLU(),
            GCNConv(graph_dim, graph_dim),
            nn.ReLU(),
        )

        # === Fusion head ===
        self.fusion_head = nn.Sequential(
            nn.Linear(polybert_dim + graph_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_labels)
        )

        # === Loss function ===
        assert property_ranges is not None and num_samples_per_property is not None
        self.loss_fn = WeightedMAELoss(property_ranges, num_samples_per_property)

        self.learning_rate = learning_rate

    # === Forward Pass ===
    def forward(self, input_ids: Tensor, attention_mask: Tensor, graph_data) -> Tensor:
        # PolyBERT embedding
        polybert_outputs = self.polybert(input_ids=input_ids, attention_mask=attention_mask)
        polybert_emb = polybert_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Graph embedding
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        for layer in self.graph_encoder:
            if isinstance(layer, (GCNConv)):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        graph_emb = global_mean_pool(x, batch)

        # Fusion
        combined = torch.cat([polybert_emb, graph_emb], dim=1)
        preds = self.fusion_head(combined)
        return preds

    # === Training / Validation / Test ===
    def shared_step(self, batch):
        preds = self(batch['input_ids'], batch['attention_mask'], batch['graph_data'])
        loss = self.loss_fn(preds, batch['labels'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # === Optimizer ===
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)