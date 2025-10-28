print('Importing libraries...')
import os
import numpy as np
import pandas as pd
import random
import warnings
import wandb
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import petname

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
    
class PolyBERTDataset(Dataset):
    def __init__(self, graphs, labels):
        self.input_ids = graphs['input_ids']
        self.attention_mask = graphs['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        
        
class DescGraphDataset(Dataset):
    def __init__(self, graphs, labels, descs=None):
        self.graphs = graphs
        self.descs = descs
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.descs[idx, :], self.labels[idx]


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
            return [self.optimizer], [self.scheduler]
        else:
            return [self.optimizer]


def train_fold(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, 
               preprocessor, property_ranges, num_samples_per_property, 
               experiment_name, fold_idx=None, cv_group_name=None):
    """Train a single fold and return metrics."""
    
    # Preprocess data
    if cfg.data.dataset_type == '3d':
        X_train_processed = X_train
        X_val_processed = X_val
        X_test_processed = X_test
    else:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val) if X_val is not None else None
        X_test_processed = preprocessor.transform(X_test) if X_test is not None else None
    
    # Create datasets
    dataset_type = cfg.data.get('dataset_type', 'graph')

    if dataset_type == 'graph':
        dataset_cls = GraphDataset
    elif dataset_type == '3d':
        dataset_cls = GraphDataset
    elif dataset_type == 'polybert':
        dataset_cls = PolyBERTDataset
    else:
        dataset_cls = TensorDataset
    
    if dataset_type == 'tensor':
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32) if Y_val is not None else None
        Y_test = torch.tensor(Y_test, dtype=torch.float32) if Y_test is not None else None
    
    train_dataset = dataset_cls(X_train_processed, Y_train)
    val_dataset = dataset_cls(X_val_processed, Y_val) if X_val is not None else None
    test_dataset = dataset_cls(X_test_processed, Y_test) if X_test is not None else None
    
    # Create dataloaders
    if dataset_type in ['tensor', 'polybert']:
        dataloader_cls = TorchDataLoader
    else:
        dataloader_cls = DataLoader 
    train_dataloader = dataloader_cls(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    )
    val_dataloader = dataloader_cls(
        val_dataset, batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    ) if val_dataset is not None else None
    test_dataloader = dataloader_cls(
        test_dataset, batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    ) if test_dataset is not None else None
    
    # Instantiate model, optimizer, scheduler
    # If using PolyBERT, pass property_ranges and num_samples_per_property
    if cfg.data.dataset_type.lower() == 'polybert':
        model = instantiate(cfg.model, 
                            property_ranges=property_ranges,
                            num_samples_per_property=num_samples_per_property)
    else:
        model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer)(model.parameters())
    
    # Handle scheduler - it might be None/null in config
    scheduler = None
    if "scheduler" in cfg and cfg.scheduler is not None:
        scheduler_obj = instantiate(cfg.scheduler)
        scheduler = scheduler_obj(optimizer) if callable(scheduler_obj) else scheduler_obj
    
    # Setup wandb logger
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_dict['train_size'] = len(train_dataset)
    config_dict['val_size'] = len(val_dataset) if val_dataset is not None else 0
    config_dict['test_size'] = len(test_dataset) if test_dataset is not None else 0
    if fold_idx is not None:
        config_dict['fold'] = fold_idx
    
    # Generate run name
    run_name = f"{experiment_name}_fold{fold_idx}" if fold_idx is not None else experiment_name
    
    wandb.finish()
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        group=cv_group_name if cv_group_name else None,
        save_dir=cfg.wandb.save_dir,
        offline=cfg.wandb.offline,
        log_model=cfg.wandb.log_model,
        config=config_dict
    )
    
    # Create lightning model and trainer
    loss_fn = WeightedMAELoss(property_ranges, num_samples_per_property)
    if isinstance(model, pl.LightningModule):
        lightning_model = model
    else:
        lightning_model = LightningModel(
            model=model, optimizer=optimizer, scheduler=scheduler,
            loss_fn=loss_fn, batch_size=cfg.data.batch_size, hparams=config_dict
        )
    trainer = pl.Trainer(**cfg.trainer, logger=logger)
    
    # Train and test
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
    
    # Get final metrics
    metrics = {}
    train_results = trainer.callback_metrics
    metrics['train_score'] = train_results.get('train_loss', torch.tensor(0.0)).item() if train_results else 0.0
    
    if val_dataloader is not None:
        val_results = trainer.callback_metrics
        metrics['val_score'] = val_results.get('val_loss', torch.tensor(0.0)).item() if val_results else 0.0
    
    if test_dataloader is not None:
        test_results = trainer.test(lightning_model, test_dataloader)
        metrics['test_score'] = test_results[0].get('test_loss', 0.0) if test_results else 0.0
    
    return metrics


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
    if cfg.data.dataset_type == "3d":
        print('Loading pickle...')
        raw_train_data = pd.read_pickle(data_path)
        graph_descriptors = None
        if cfg.data.compute_desc:
            from src.preprocessing.rdkit_descriptors import RDKitDescriptorsPreprocessor
            graph_descriptors = RDKitDescriptorsPreprocessor().fit_transform(raw_train_data['SMILES'])
        # for now just one graph TODO: multigraph and adding descriptors
        X = raw_train_data[cfg.data.intervals[0]].values
        Y = raw_train_data[cfg.data.label_names].values
    else:
        raw_train_data = pd.read_csv(data_path)
        X = raw_train_data['SMILES'].values
        Y = raw_train_data[cfg.data.label_names].values

    # Normalize labels
    if cfg.normalize_labels == 'minmax':
        Y = (Y - np.nanmin(Y, axis=0)) / (np.nanmax(Y, axis=0) - np.nanmin(Y, axis=0))
    elif cfg.normalize_labels == 'standard':
        Y = (Y - np.nanmean(Y, axis=0)) / np.nanstd(Y, axis=0)

    # Compute property statistics for loss function
    property_ranges = torch.tensor([
        np.nanmax(Y[:, i]) - np.nanmin(Y[:, i]) 
        for i in range(Y.shape[1])
    ], dtype=torch.float32)
    num_samples_per_property = torch.tensor([
        np.sum(~np.isnan(Y[:, i])) 
        for i in range(Y.shape[1])
    ], dtype=torch.float32)
    
    # Split test set
    X_trainval = X
    Y_trainval = Y
    X_test = None
    Y_test = None
    
    if cfg.data.test_split > 0:
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(
            X, Y, test_size=cfg.data.test_split, random_state=cfg.data.random_state
        )
        print(f"Test size: {len(X_test)}")
    
    # Instantiate preprocessor
    print(f"\nInstantiating preprocessor: {cfg.preprocessing._target_}")
    preprocessor = instantiate(cfg.preprocessing)
    
    # Run cross-validation or single train-val-test split
    if cfg.crossval:
        print(f"\n{'='*60}")
        print(f"Cross-Validation Mode ({cfg.data.n_splits} folds)")
        print('='*60)
        
        # Generate CV group name
        experiment_name = cfg.wandb.name if cfg.wandb.name else petname.generate(3, separator='-')
        cv_group_name = f"{experiment_name}_cv"
        cv = KFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.random_state)
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval)):
            print(f"\nFold {fold_idx + 1}/{cfg.data.n_splits}")
            print('-'*60)
            
            X_train = X_trainval[train_idx]
            Y_train = Y_trainval[train_idx]
            X_val = X_trainval[val_idx]
            Y_val = Y_trainval[val_idx]
            
            metrics = train_fold(
                cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                preprocessor, property_ranges, num_samples_per_property,
                experiment_name, fold_idx=fold_idx, cv_group_name=cv_group_name
            )
            fold_metrics.append(metrics)
            print(f"Fold {fold_idx + 1} metrics: {metrics}")
        
        # Aggregate results
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
        
        print(f"\n{'='*60}")
        print("Cross-Validation Results (averaged over folds):")
        print('='*60)
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
        
    else:
        print(f"\n{'='*60}")
        print("Standard Train-Val-Test Split")
        print('='*60)
        
        # Generate experiment name
        experiment_name = cfg.wandb.name if cfg.wandb.name else petname.generate(3, separator='-')
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=cfg.data.val_size, 
            random_state=cfg.data.random_state
        )
        print(f"Train size: {len(X_train)}")
        print(f"Val size: {len(X_val)}")
        
        # Run training (preprocessing happens inside train_fold)
        metrics = train_fold(
            cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test,
            preprocessor, property_ranges, num_samples_per_property,
            experiment_name, fold_idx=None, cv_group_name=None
        )
        
        print(f"\n{'='*60}")
        print("Final Metrics:")
        print('='*60)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print('='*60)


if __name__ == "__main__":
    main()
