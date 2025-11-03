import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import lightning.pytorch as pl
from src.loss.masked_loss import WeightedMAELoss  # make sure path is correct
from src.utils.running_mean import RunningMean

class PolyBERTForRegression(pl.LightningModule):
    """
    PolyBERT regression model with weighted MAE loss handling NaNs.
    """

    def __init__(
        self,
        model_name: str = "kuelumbus/polybert",
        num_labels: int = 5,
        learning_rate: float = 1e-4,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        property_ranges: Tensor = None,
        num_samples_per_property: Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained PolyBERT
        self.polybert = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.polybert.parameters():
                param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_size = self.polybert.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # Weighted MAE loss
        assert property_ranges is not None and num_samples_per_property is not None, \
            "Must provide property_ranges and num_samples_per_property"
        self.loss_fn = WeightedMAELoss(property_ranges, num_samples_per_property)

        self.learning_rate = learning_rate
        self.val_running_mean = RunningMean()

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        outputs = self.polybert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regression_head(pooled_output)
        return predictions

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        val_outputs = self.trainer.callback_metrics  
        mean_val_loss = val_outputs.get("val_loss")  
        if mean_val_loss is not None:
            self.val_running_mean.update(mean_val_loss.item())
            self.log("val_loss_smoothed", self.val_running_mean.mean, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, labels)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)