# Setup

1. Download uv from [here](https://docs.astral.sh/uv/getting-started/installation/). Run `uv sync` to install the dependencies.
2. Join the competition on Kaggle [here](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025). Download all data and unzip it into data/raw.

# Usage

Launch tensorboard with `uv run tensorboard --logdir lightning_logs` to view the progress. In another terminal, run `uv run main.py` to run the code. 

# Data Format
Uses the PyG (PyTorch Geometric) format, which provides its own [Data class for graphs](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.data.Data.html#torch_geometric.data.Data). It's based on edge lists. See `src/preprocessing/text_based.py` for an example.