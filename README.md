# Setup

1. Download uv from [here](https://docs.astral.sh/uv/getting-started/installation/). Run `uv sync` to install the dependencies.
2. Join the competition on Kaggle [here](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025). Download all data and unzip it into data/raw.

# Usage

The project uses **Hydra** for configuration management and **Weights & Biases** for experiment tracking and logging.

## Setup Weights & Biases

Before running training:
1. Create an account at [wandb.ai](https://wandb.ai)
2. Login: `uv run wandb login`

## Running Training

Basic usage with default configuration:
```bash
uv run main.py
```

### Override Configuration Parameters

Hydra allows you to override any configuration parameter from the command line:

```bash
# Change model to GCN
uv run main.py model=gcn

# Change learning rate and batch size
uv run main.py optimizer.lr=0.0001 data.batch_size=64

# Set random seed for reproducibility
uv run main.py seed=42

# Change number of epochs
uv run main.py trainer.max_epochs=100

# Multiple overrides at once
uv run main.py model=gcn optimizer.lr=0.0005 data.batch_size=128 seed=123
```

### Wandb Configuration

```bash
# Custom project and run name
uv run main.py wandb.project=my-polymer-project wandb.name=experiment-1

# Offline mode (sync later with `wandb sync`)
uv run main.py wandb.offline=true

# Disable model logging
uv run main.py wandb.log_model=false
```

## Configuration Structure

The configuration is organized in `conf/` directory:

```
conf/
├── config.yaml          # Main configuration file with defaults
├── model/               # Model architectures
│   ├── gcn.yaml        # Simple GCN model
│   └── gat.yaml        # GATv2 model (default)
├── optimizer/           # Optimizer configurations
│   └── adam.yaml       # Adam optimizer
├── trainer/             # PyTorch Lightning trainer settings
│   └── default.yaml    
├── data/                # Data loading and preprocessing
│   └── default.yaml
├── loss/                # Loss function configurations
│   └── weighted_mae.yaml
└── wandb/              # Weights & Biases settings
    └── default.yaml
```

### Model Configurations

**GCN Model** (`model=gcn`):
- Simple Graph Convolutional Network
- Parameters: `in_channels`, `hidden_channels`, `out_channels`

**GAT Model** (`model=gat`, default):
- Graph Attention Network v2
- Parameters: `in_channel`, `hidden_channels`, `hidden_layers`, `heads`, `dropout`, etc.

### Creating Custom Configurations

You can create custom configuration files for different experiments:

```bash
# Create a new model config
cp conf/model/gat.yaml conf/model/my_gat.yaml
# Edit my_gat.yaml with your parameters
uv run main.py model=my_gat

# Create a new optimizer config
cp conf/optimizer/adam.yaml conf/optimizer/sgd.yaml
# Edit to use SGD instead
uv run main.py optimizer=sgd
```

## Hydra Features

### Multi-run for Hyperparameter Sweeps

Run multiple experiments with different configurations:

```bash
# Sweep over learning rates
uv run main.py -m optimizer.lr=0.001,0.0001,0.00001

# Sweep over models and learning rates
uv run main.py -m model=gcn,gat optimizer.lr=0.001,0.0001

# Sweep with seeds
uv run main.py -m seed=1,2,3,4,5
```

### Outputs Directory

Hydra automatically organizes outputs by date and time:
- Single runs: `outputs/YYYY-MM-DD/HH-MM-SS/`
- Multi-runs: `multirun/YYYY-MM-DD/HH-MM-SS/`

Each run directory contains:
- `.hydra/` folder with the resolved configuration
- Logs and any other outputs from your run

## Configuration Reference

### Key Configuration Groups

**Data** (`data/default.yaml`):
- `data_dir`: Path to raw data
- `batch_size`: Training batch size
- `test_size`: Test set proportion
- `val_size`: Validation set proportion
- `num_workers`: DataLoader workers (0 for Windows)

**Trainer** (`trainer/default.yaml`):
- `max_epochs`: Number of training epochs
- `accelerator`: 'gpu' or 'cpu'
- `devices`: Number of devices or 'auto'
- `log_every_n_steps`: Logging frequency

**Optimizer** (`optimizer/adam.yaml`):
- `lr`: Learning rate
- Any other optimizer-specific parameters

**Wandb** (`wandb/default.yaml`):
- `project`: Wandb project name
- `name`: Run name (auto-generated if null)
- `offline`: Run in offline mode
- `log_model`: Whether to log model checkpoints

## Examples

```bash
# Quick test with fewer epochs
uv run main.py trainer.max_epochs=10

# Reproducible experiment
uv run main.py seed=42 wandb.name=reproducible-run-1

# High-capacity model with smaller batch
uv run main.py model=gat model.hidden_channels=[256,256] data.batch_size=16

# CPU training
uv run main.py trainer.accelerator=cpu

# Full hyperparameter sweep
uv run main.py -m model=gcn,gat optimizer.lr=0.001,0.0001 seed=1,2,3
```

# Data Format

Uses the PyG (PyTorch Geometric) format, which provides its own [Data class for graphs](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.data.Data.html#torch_geometric.data.Data). It's based on edge lists. See `src/preprocessing/text_based.py` for an example.

# Technical Details

## Model Instantiation

Models are instantiated using Hydra's `_target_` directive, which points to the Python class:

```yaml
# conf/model/gat.yaml
_target_: src.models.gat.GATv2
in_channel: 43
hidden_channels: [128]
...
```
