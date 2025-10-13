print('Importing libraries...')
import os
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.loss.masked_loss import WeightedMAELoss

print('Finished importing libraries.')
print(f'CUDA available: {torch.cuda.is_available()}')

# Suppress warning about worker count, increasing workers lead to crashing on windows
warnings.filterwarnings("ignore", ".*does not have many workers.*")     
# Suppress warning about torch-scatter, which has difficulties installing with uv
warnings.filterwarnings("ignore", ".*can be accelerated via the 'torch-scatter' package*") 


# Create datasets and dataloaders
class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


# Wrap the model in a LightningModule
class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, loss_fn, batch_size, hparams=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        # Save hyperparameters for wandb/tensorboard logging
        if hparams:
            self.save_hyperparameters(hparams)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {'optimizer': self.optimizer, 'scheduler': self.scheduler}
        else:
            return {'optimizer': self.optimizer}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function using Hydra configuration."""
    
    print("="*60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Set random seed if specified
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Random seed set to: {cfg.seed}")
    
    # =================== Data Loading ===================
    print("\nLoading data...")
    data_path = os.path.join(cfg.data.data_dir, cfg.data.train_file)
    raw_train_data = pd.read_csv(data_path)
    
    X = raw_train_data['SMILES'].values
    Y = raw_train_data[cfg.data.label_names].values

    if cfg.normalize_labels == 'minmax':
        Y = (Y - np.nanmin(Y, axis=0)) / (np.nanmax(Y, axis=0) - np.nanmin(Y, axis=0))
    elif cfg.normalize_labels == 'standard':
        Y = (Y - np.nanmean(Y, axis=0)) / np.nanstd(Y, axis=0)

    # Train, Val, Test split
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, 
        test_size=cfg.data.test_size, 
        random_state=cfg.data.random_state
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, 
        test_size=cfg.data.val_size, 
        random_state=cfg.data.random_state
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # =================== Loss Function Setup ===================
    # Compute property statistics for loss function
    property_ranges = torch.tensor([
        np.nanmax(Y[:, i]) - np.nanmin(Y[:, i]) 
        for i in range(Y.shape[1])
    ], dtype=torch.float32)
    
    num_samples_per_property = torch.tensor([
        np.sum(~np.isnan(Y[:, i])) 
        for i in range(Y.shape[1])
    ], dtype=torch.float32)
    
    # Instantiate loss function with computed statistics
    loss_fn = WeightedMAELoss(property_ranges, num_samples_per_property)
    print(f"\nLoss function initialized with property ranges: {property_ranges}")
    
    # =================== Preprocessing ===================
    print(f"\nInstantiating preprocessor: {cfg.preprocessing._target_}")
    preprocessor = instantiate(cfg.preprocessing)
    print("\nPreprocessing data...")
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    
    # Create datasets
    train_dataset = GraphDataset(X_train, Y_train)
    val_dataset = GraphDataset(X_val, Y_val)
    test_dataset = GraphDataset(X_test, Y_test)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    )
    
    # =================== Model Instantiation ===================
    print(f"\nInstantiating model: {cfg.model._target_}")
    model = instantiate(cfg.model)
    print(f"Model architecture:\n{model}")
    
    # =================== Optimizer Instantiation ===================
    print(f"\nInstantiating optimizer: {cfg.optimizer._target_}")
    # Use partial instantiation to pass model parameters
    optimizer = instantiate(cfg.optimizer)(model.parameters())

    # =================== Scheduler Instantiation ===================
    if cfg.scheduler is not None:
        print(f"\nInstantiating scheduler: {cfg.scheduler._target_}")
        scheduler = instantiate(cfg.scheduler)(optimizer)
    else:
        scheduler = None
    
    # =================== Wandb Logger Setup ===================
    print("\nSetting up Weights & Biases logger...")
    # Convert config to dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Add dataset sizes to config
    config_dict['train_size'] = len(train_dataset)
    config_dict['val_size'] = len(val_dataset)
    config_dict['test_size'] = len(test_dataset)
    
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name if cfg.wandb.name else cfg.experiment_name,
        save_dir=cfg.wandb.save_dir,
        offline=cfg.wandb.offline,
        log_model=cfg.wandb.log_model,
        config=config_dict  # Pass full config to wandb
    )
    
    # =================== Lightning Model ===================
    lightning_model = LightningModel(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        batch_size=cfg.data.batch_size,
        hparams=config_dict
    )
    
    # =================== Trainer Setup ===================
    print("\nSetting up PyTorch Lightning Trainer...")
    trainer = pl.Trainer(**cfg.trainer, logger=logger)      # callbacks can be added via config if needed
    
    # =================== Training ===================
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
    
    # =================== Testing ===================
    print("\n" + "="*60)
    print("Starting testing...")
    print("="*60)
    trainer.test(lightning_model, test_dataloader)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
