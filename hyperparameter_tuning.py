import os
import gc
import glob
import math
import random
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
import heapq
from typing import Dict, List, Tuple
from inference import model_name, model_instance, device, embed_dim, dropout_rate, dtype
from dataloading import learning_rate, batch_size, train_loader, epochs  
import torch
from model import FeedForwardMLP
from training import train_model
from evaluate import evaluate_model, test_loader, criterion


# Defining Hyperparameters Sweep Function
def sweep_config():
    """Defines the hyperparameter sweep configuration for WandB Sweeps."""

    sweep_config = {
        # Using Bayesian optimization
        "method": "bayes",
         # Metric to optimize
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "hidden_layers": {
                "values": [[64, 32], [128, 64], [256, 128], [64, 32, 16], [128, 64, 32]]
            },
            "dropout_rates": {
                "values": [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.2, 0.3, 0.4]]
            },
            "learning_rate": {
                "values": [0.01, 0.001, 0.0001]
            },
            "epochs": {
                "values": [10, 20, 30]
            },
            "batch_size": {
                "values": [16, 32, 64]
            }
        }
    }
    return sweep_config

# Storing Sweep ID
sweep_id = wandb.sweep(sweep_config(), project="GROUP3-GPT-TRANSFORMER-DEMO") # Pass the output of sweep_config

def sweep_train():
    # Adding error handling within the sweep_train function
    try:
        with wandb.init() as run:
            config = wandb.config
            # Accessing config parameters using config.hidden_layers, config.dropout_rates, etc.
            model = FeedForwardMLP(embed_dim, dropout_rate, device, dtype)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            # Calling train_model with the correct arguments and data loaders
            train_model(model, train_loader, test_loader, epochs=config.epochs)
            # Calling evaluate_model with the correct arguments and data loaders
            evaluate_model(model, test_loader, criterion)
    except Exception as e:
        # Printing the exception
        print(f"Error in sweep_train: {e}")
        # Optionally log the error to WandB
        wandb.log({"sweep_error": str(e)})
        # Re-raise the exception to stop the agent
        raise e

# Deploying Sweep Agent
wandb.agent(sweep_id, function=sweep_train, count=3)