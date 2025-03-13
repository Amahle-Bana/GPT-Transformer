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
from inference import model_name, model_instance, device
from dataloading import learning_rate, batch_size, train_loader, epochs  
import torch
from dataloading import LanguageModellingDataset


# Load the test data
with open("cleaned_test_data.txt", "r", encoding="utf-8") as file:
    test_text_src = file.readlines()

# Loading the test data
text_src = test_text_src

dataset = LanguageModellingDataset(
    text_src=text_src, tokenizer=model_instance.tokenizer,
    context_length=model_instance.context_length,
    stride=model_instance.context_length // 2,
)

# Printing the length of the test dataset
print( dataset.__len__() )

# Creating the test loader
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

# Evaluating the model on the test data and logs results to wandb.
def evaluate_model(model, test_loader, criterion):

        # model: The trained PyTorch model.
        # test_loader: DataLoader for the test data.
        # criterion: Loss function.
        # A dictionary containing the test loss and accuracy.

    # Initializing a new WandB run
    wandb.init(project="GROUP3-GPT-TRANSFORMER-DEMO", group="evaluation", job_type="test")

    # Setting model to evaluation mode
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0


    # Disabling gradient calculation
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.type(torch.LongTensor))
            test_loss += loss.item()

            # Getting predicted class
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # Counting correct predictions
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    # Logging results to wandb
    wandb.log({
        "test_loss": avg_test_loss,
        "test_accuracy": accuracy
    })

    # Printing the results
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return {"test_loss": avg_test_loss, "test_accuracy": accuracy}

evaluate_model(model_instance, test_loader, criterion)