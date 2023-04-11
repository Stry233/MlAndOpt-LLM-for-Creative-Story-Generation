# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console

# Load and process dataset
fileTag = "your_file_tag_here"
dfTrain = pd.read_csv(f'./genV2-{fileTag}-train.csv')
dfTest = pd.read_csv(f'./genV2-{fileTag}-test.csv')
dfAll = pd.concat([dfTest, dfTrain]).sample(frac=1).reset_index(drop=True)

# Define rich console logger
console = Console(record=True)

# Function to display dataframe in ASCII format
def display_df(df):
    console = Console()
    table = Table(Column("source_text (Question)", justify="center"),
                  Column("target_text (Answer)", justify="center"),
                  title="Sample Data", pad_edge=False, box=box.ASCII)

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# Table to log training status
training_logger = Table(Column("Epoch", justify="center"),
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"),
                        title="Training Status", pad_edge=False, box=box.ASCII)

# Set up device for GPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Custom Dataset class
class NextSentenceDataSetClass(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Clean data to ensure it is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  padding="max_length",
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  padding="max_length",
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

# Training function
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:,
