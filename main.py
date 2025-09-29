print('Importing libraries...')
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
from src.loss.masked_loss import MaskedLoss
import warnings

print('Finished importing libraries.')

# =================== Parameters ===================
DATA_DIR = "data/raw"
PREPROCESSOR = TextBasedPreprocessor() # Anything that transforms SMILES string into PyG graph
MODEL = SimpleGCN(43, 128, 5) # Torch module that takes graph and outputs (batch_size, num_labels=5) 
OPTIMIZER = Adam(MODEL.parameters(), lr=1e-2)
LOSS_FN = MaskedLoss(L1Loss()) # Mask the loss in case of missing labels
BATCH_SIZE = 32
TRAINER_PARAMS = {
    'max_epochs': 50,
    'accelerator': 'gpu',
    'logger': TensorBoardLogger(save_dir='lightning_logs'),
    'callbacks': [pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)]
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

# Create datasets and dataloaders
class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = torch.tensor(labels, dtype=torch.float32)    
    def __len__(self):
        return len(self.graphs)    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

train_dataset = GraphDataset(X_train, Y_train)
val_dataset = GraphDataset(X_val, Y_val)
test_dataset = GraphDataset(X_test, Y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **DATALOADER_PARAMS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, **DATALOADER_PARAMS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, **DATALOADER_PARAMS)
warnings.filterwarnings("ignore", ".*does not have many workers.*") # Suppress warning about worker count, increasing workers lead to crashing on windows

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

LIGHTNING_MODEL = LightningModel(MODEL, OPTIMIZER, LOSS_FN)

# Perform training, validation and testing
trainer = pl.Trainer(**TRAINER_PARAMS)
trainer.fit(LIGHTNING_MODEL, train_dataloader, val_dataloader)
trainer.test(LIGHTNING_MODEL, test_dataloader)