import lightning.pytorch as pl
import torch.nn as nn
from hydra.utils import instantiate
from src.ssl.augmentations import get_augmentations
from src.ssl.loss import NTXentLoss
from torch_geometric.nn import global_mean_pool


class ContrastiveModel(pl.LightningModule):

    def __init__(
        self,
        backbone,
        optimizer_cfg,
        scheduler_cfg,
        backbone_output_dim,
        temperature: float = 0.1,
        drop_p: float = 0.2,
        mask_p: float = 0.3,
        hparams=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = backbone

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.contrastive_loss = NTXentLoss(temperature=temperature)

        self.drop_p = drop_p
        self.mask_p = mask_p

        # Store configs, will be instantiated in configure_optimizers
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, data):
        node_embeddings = self.model.encode(data)
        graph_embedding = global_mean_pool(node_embeddings, data.batch)
        # return a single graph embedding
        return graph_embedding

    def training_step(self, batch):
        batch_size = batch.num_graphs

        # Create two augmented views
        aug1, aug2 = get_augmentations(batch, self.drop_p, self.mask_p)

        # Get embeddings from the backbone
        z1 = self.forward(aug1)
        z2 = self.forward(aug2)

        # Get projections from the head
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        # Calculate the contrastive loss
        loss = self.contrastive_loss(p1, p2)
        self.log(
            "ssl_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.num_graphs

        # Create two augmented views
        aug1, aug2 = get_augmentations(batch, self.drop_p, self.mask_p)

        # Get embeddings from the backbone
        z1 = self.forward(aug1)
        z2 = self.forward(aug2)

        # Get projections from the head
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        # Calculate the contrastive loss
        loss = self.contrastive_loss(p1, p2)

        # Log with 'val_' prefix. 'on_step=False' is standard for validation.
        self.log(
            "val_ssl_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_cfg(self.model.parameters())
        if self.scheduler_cfg is not None:
            scheduler = self.scheduler_cfg(optimizer)
            return [optimizer], [scheduler]
        else:
            return [optimizer]
