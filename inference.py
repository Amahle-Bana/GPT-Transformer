import os
import gc
import glob
import math
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
import heapq
from typing import Dict, List, Tuple
from torchinfo import summary

# Import the custom model (Ensure model.py exists in the same directory)
from model import DecoderOnlyModel

# Define model parameters
model_name = "GROUP3-GPT-TRANSFORMER-DEMO"
num_layers = 3
num_heads = 8
embed_dim = 512
dropout_rate = 0.1
context_length = 500
with_encoder = False

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the model
model_instance = DecoderOnlyModel(
    num_layers=num_layers,
    num_heads=num_heads,
    embed_dim=embed_dim,
    dropout_rate=dropout_rate,
    with_encoder=with_encoder,
    device=device,
    model_name=model_name
)
model_instance = model_instance.to(device)

if __name__ == "__main__":
    # Define prompt
    prompt = "Hello there, how are you today???!"

    # Generate output
    output, tokens = model_instance.generate(prompt, max_output=50)
    print(output)

    # Prepare input data for model
    context_input = torch.tensor([model_instance.tokenizer.encode(prompt)]).to(model_instance.device)

    # Ensure input shape is compatible
    input_data = context_input.unsqueeze(0)  # Adding batch dimension if needed

    # Display model summary
    summary(model_instance, input_data=input_data, device=model_instance.device)
