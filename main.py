print('Importing libraries...')
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import Dataset
import torch
from src.preprocessing.text_based import TextBasedPreprocessor
from src.models.gcn import SimpleGCN
from src.models.gat import GATv2
from src.loss.masked_loss import MaskedLoss
import warnings
import random

print('Finished importing libraries.')
print(f'CUDA available: {torch.cuda.is_available()}')

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
    def __init__(self, model, optimizer, loss_fn):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss

    def configure_optimizers(self):
        return self.optimizer

if __name__ == "__main__":
    # parsing inputs
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of steps')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-d', '--device', type=str, default='gpu', help='device')
    parser.add_argument('--seed', type=int, default=None, help='seed experiment') 
    parser.add_argument('-m', '--model', type=str, default='gat', help='model type ') 

    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    
    # =================== Parameters ===================
    DATA_DIR = "data/raw"
    PREPROCESSOR = TextBasedPreprocessor() # Anything that transforms SMILES string into PyG graph

    # select model
    match args.model:
        case "gcn":
            MODEL = SimpleGCN(43, 128, 5) # Torch module that takes graph and outputs (batch_size, num_labels=5) 
        case "gat":
            MODEL = GATv2(43)
    
    OPTIMIZER = Adam(MODEL.parameters(), lr=args.lr)
    LOSS_FN = MaskedLoss(L1Loss()) # Mask the loss in case of missing labels
    BATCH_SIZE = args.batch_size
    TRAINER_PARAMS = {
        'max_epochs': args.epochs,
        'accelerator': args.device,
        'logger': TensorBoardLogger(save_dir='lightning_logs'),
        # 'callbacks': [pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)]
    }
    DATALOADER_PARAMS = {
        # 'num_workers': 8,
        # 'persistent_workers': True
    }
    # =================== Parameters ===================


    # Load data as numpy arrays
    raw_train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_names = ['Tg','FFV','Tc','Density','Rg']
    X = raw_train_data['SMILES'].values
    Y = raw_train_data[label_names].values

    # Train, Val, Test split of 70/15/15
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.17, random_state=42)

    # Preprocess data
    # Fit the preprocessor on training data and transform all datasets
    X_train = PREPROCESSOR.fit_transform(X_train)
    X_val = PREPROCESSOR.transform(X_val)
    X_test = PREPROCESSOR.transform(X_test)
    
    train_dataset = GraphDataset(X_train, Y_train)
    val_dataset = GraphDataset(X_val, Y_val)
    test_dataset = GraphDataset(X_test, Y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **DATALOADER_PARAMS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, **DATALOADER_PARAMS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, **DATALOADER_PARAMS)
    warnings.filterwarnings("ignore", ".*does not have many workers.*") # Suppress warning about worker count, increasing workers lead to crashing on windows

    LIGHTNING_MODEL = LightningModel(MODEL, OPTIMIZER, LOSS_FN)

    # Perform training, validation and testing
    trainer = pl.Trainer(**TRAINER_PARAMS)
    trainer.fit(LIGHTNING_MODEL, train_dataloader, val_dataloader)
    trainer.test(LIGHTNING_MODEL, test_dataloader)