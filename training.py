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
from inference import model_name, model_instance
from dataloading import learning_rate, batch_size, train_loader, epochs  


def save_model(model, optimizer, epoch, total_epochs, loss, perplexity, checkpoint_dir: str = "./checkpoints", num_checkpoints=3, save_every=10):
    if ( epoch % save_every  == 0):
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Get all existing checkpoints
        checkpoint_pattern = f"{checkpoint_dir}/{model.model_name}_ckpt_*.pt"
        checkpoint_files = glob.glob(checkpoint_pattern)

        saved_checkpoints: List[Tuple[float, str]] = []
        for checkpoint in checkpoint_files:
            parts = checkpoint.split("_")
            saved_loss = float(parts[-1].replace(".pt", ""))  # Assuming loss is in filename
            saved_checkpoints.append((saved_loss, checkpoint))

        saved_checkpoints.sort()  # Sort by loss (ascending)

        if (len(saved_checkpoints) < num_checkpoints) or (loss < saved_checkpoints[-1][0]):
            checkpoint_path = os.path.join(checkpoint_dir, f"{model.model_name}_ckpt_{epoch}_loss_{loss:.6f}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_epochs': total_epochs,
                'loss': loss,
                'perplexity': perplexity,
            }, checkpoint_path)
            print("Model checkpoint saved.")

            # Remove the worst checkpoint if we exceed num_models
            if len(saved_checkpoints) >= num_checkpoints:
                os.remove(saved_checkpoints[-1][1])

def train_model(model, train_loader, epochs):
    """
    Trains the given PyTorch model.

    Args:
        model: The PyTorch model to train.
        train_loader: A DataLoader for the training dataset.
        epochs: The number of training epochs.
    """
    run = wandb.init(
        project = "Class Demo 2",
        config = {
            "model_name": model.model_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "embed_dim": model.embed_dim,
            "dropout_rate": model.dropout_rate,
            "device": model.device,
        }
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}", leave=True,  ncols=90, colour='#00F00F',  #unit="batch"
        )
        for batch in progress_bar:
            inputs, attn_mask, labels = batch['input_ids'].to(model.device), batch['attention_mask'].to(model.device), batch['labels'].to(model.device)

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs) # Forward pass, output shape: (batch_size, seq_len, target_vocab_size)
            # print("\n", outputs.shape)
            next_word_probs = outputs[:, -1, :]  # (batch_size, target_vocab_size)
            # print(next_word_probs.shape)
            outputs = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_len, target_vocab_size]
            # print("\n", outputs.shape)
            labels = labels.view(-1)  # Reshape to [batch_size * seq_len]

            loss = loss_fn(outputs, labels) # Compute the loss.
            loss.backward() # Backward pass to Compute gradients.
            optimizer.step() # Optmize, i.e Update gradients.

            loss_value = loss.item()
            # print(loss_value)
            running_loss += loss_value

            progress_bar.set_postfix(loss=loss_value)
            # progress_bar.update()

        # progress_bar.close()
        loss = running_loss / len(train_loader)
        perplexity = math.exp(loss)
        run.log({
            "train/loss": loss, "train/perplexity": perplexity
        });
        save_model(model, optimizer, epoch, epochs, loss, perplexity)

    # Return the trained model
    print("\n====================================Finished Training!====================================")
    run.finish()
    model.eval()
    return model


trained_model = train_model(model_instance, train_loader, epochs)

prompt = "Good morning."
output, tokens = trained_model.generate(prompt , max_output=100)
print(output)
# print(tokens)

print(tokens.shape)