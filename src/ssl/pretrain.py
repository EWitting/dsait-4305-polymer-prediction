import os
import numpy as np
import pandas as pd
import random
import wandb

import torch
from torch_geometric.loader import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import petname


class SSLDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# config path relative to the file location so have to go up.
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # set the seed if specified
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Random seed set to: {cfg.seed}")

    data_path = os.path.join(cfg.data.data_dir, cfg.data.train_file)

    if cfg.data.dataset_type == "3d":
        raw_train_data = pd.read_pickle(data_path)
        X = raw_train_data[cfg.data.intervals[0]].values
    else:
        raw_train_data = pd.read_csv(data_path)
        X = raw_train_data["SMILES"].values

    # Create train/val split
    X_train, X_val = train_test_split(
        X, test_size=cfg.data.val_size, random_state=cfg.data.random_state
    )

    print(f"Pre-train size: {len(X_train)}")
    print(f"Pre-train val size: {len(X_val)}")

    # Instantiate preprocessor
    print("\nPreprocessing pre-training data...")
    preprocessor = instantiate(cfg.preprocessing)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    train_dataset = SSLDataset(X_train_processed)
    val_dataset = SSLDataset(X_val_processed)

    dataloader_cls = DataLoader  # Assuming graph data
    train_dataloader = dataloader_cls(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=(
            cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
        ),
    )
    val_dataloader = dataloader_cls(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=(
            cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
        ),
    )

    print(f"\nInstantiating backbone model: {cfg.model._target_}")
    backbone_model = instantiate(cfg.model)

    try:
        backbone_output_dim = cfg.model.hidden_channels[-1]
    except Exception as e:
        print(
            f"Could not automatically determine backbone_output_dim from cfg.model.hidden_channels"
        )
        print(
            "Please ensure your GATv2 model's 'encode' method outputs a known dimension."
        )
        try:
            backbone_output_dim = backbone_model.fcs[0].in_features
            print(f"Inferred backbone_output_dim as: {backbone_output_dim}")
        except:
            raise e

    # =================== Wandb Logger Setup ===================
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_dict["train_size"] = len(train_dataset)
    config_dict["val_size"] = len(val_dataset)

    hparams_dict = config_dict.copy()
    if "loss" in hparams_dict:
        del hparams_dict["loss"]

    # change this depending on which pretext task
    try:
        # Build a descriptive name from the config
        lr = cfg.optimizer.lr
        temp = cfg.ssl_model.temperature
        drop_p = cfg.ssl_model.drop_p
        mask_p = cfg.ssl_model.mask_p

        # This will become the filename
        experiment_name = f"SSL_lr{lr}_temp{temp}_drop{drop_p}_mask{mask_p}_linear"
    except Exception as e:
        print(f"Could not build dynamic name, falling back. Error: {e}")
        experiment_name = "ContrastiveLearning_SSL"

    run_name = f"{experiment_name}_Pretraining"

    wandb.finish()  # Finish any previous runs
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir,
        offline=cfg.wandb.offline,
        config=config_dict,
    )

    if cfg.ssl_model is None:
        return

    print(f"\nInstantiating SSL model: {cfg.ssl_model._target_}")
    lightning_model = instantiate(cfg.ssl_model, hparams=hparams_dict)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{experiment_name}_ssl_checkpoint",
        monitor="val_ssl_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    print("\nSetting up PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback],  # Add the callback
    )

    print("\n" + "=" * 60)
    print("Starting SSL Pre-training...")
    print("=" * 60)
    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    print("\n" + "=" * 60)
    print("Pre-training completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
