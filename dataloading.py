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
from inference import model_instance


epochs = 500
learning_rate = 1e-5
batch_size = 4

class LanguageModellingDataset(Dataset):
    def __init__(self, text_src, tokenizer, context_length=100, stride=50):
        self.tokenizer = tokenizer
        self.text_src = text_src
        self.context_length = context_length
        self.stride = stride  # How much to slide the window

        if isinstance(text_src, str) and os.path.isfile(text_src):
            with open(text_src, 'r') as f:
                self.text_src = f.read()  # Read the whole file as one string
        elif isinstance(text_src, str): # If text_src is already a string
            self.text_src = text_src
        else:
            raise TypeError("text_src must be a string or a path to a text file.")

        self.tokenized_text = self.tokenizer.encode(self.text_src, add_special_tokens=False) # Tokenize once
        self.chunks = self._chunk_tokens()

    def _chunk_tokens(self):
        chunks = []
        for i in range(0, len(self.tokenized_text) - self.context_length, self.stride):
            chunk = self.tokenized_text[i:i + self.context_length]
            chunks.append(chunk)
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk) # Convert to tensor directly
        attention_mask = torch.ones_like(input_ids) # Create attention mask of 1s

        # Shift inputs for Autoregressive target labels
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = self.tokenizer.eos_token_id  # Ignore last token

        # Ensure labels are within the valid vocabulary range
        labels[labels >= self.tokenizer.vocab_size] = self.tokenizer.pad_token_id # self.tokenizer.pad_token_id  # Ignore last token # Ignore last token

        return { "input_ids": input_ids,  "attention_mask": attention_mask,  "labels": labels}




with open("Cleaned_Transformer_Training_Dataset.txt", "r", encoding="utf-8") as file:
    pretraining_text_src = file.read()

text_src = pretraining_text_src

dataset = LanguageModellingDataset(
    text_src=pretraining_text_src, tokenizer=model_instance.tokenizer,
    context_length=model_instance.context_length,
    stride=model_instance.context_length // 2,
)
print( dataset.__len__() )
# print(dataset.texts)
# print(dataset.__getitem__(0)['input_ids'])
# print(dataset.__getitem__(0)['labels'])

# print(dataset.__getitem__(0)['input_ids'])
# print(dataset.__getitem__(0)['labels'])
# print(model_instance.tokenizer.decode(dataset.__getitem__(0)['labels']))

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)