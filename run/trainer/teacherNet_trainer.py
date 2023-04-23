# Import required libraries
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")


# Define preprocess_data function
def preprocess_data(data):
    data["source"] = data["question"].apply(
        lambda x: x.replace("<extra_id_0>", " ").replace("<extra_id_1>", " ")
    )
    return data


# Define tokenize function
def tokenize(batch):
    tokenized_input = tokenizer(
        batch["source"], padding="max_length", truncation=True, max_length=256
    )
    tokenized_label = tokenizer(
        batch["answer"], padding="max_length", truncation=True, max_length=256
    )
    return {
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "labels": tokenized_label["input_ids"],
    }


# Load data from CSV files
train_data_csv = "genV2-original-plutchik-v1-train.csv"
test_data_csv = "genV2-original-plutchik-v1-test.csv"

train_data = pd.read_csv(train_data_csv)
test_data = pd.read_csv(test_data_csv)

# Combine datasets
total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

# Preprocess data
total_data = preprocess_data(total_data)

# Split data into train and test sets
train_data, test_data = train_test_split(total_data, test_size=0.05, random_state=42)

# Save the new train and test data as CSV files
train_data.to_csv("combined_train_data.csv", index=False)
test_data.to_csv("combined_test_data.csv", index=False)

# Load datasets
train_dataset = load_dataset('csv', data_files='combined_train_data.csv')
test_dataset = load_dataset('csv', data_files='combined_test_data.csv')

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    learning_rate=3e-5,
)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=256)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=256)

# Set dataset formats
train_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)
test_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset['train'],
    eval_dataset=test_dataset['train'],
)

# Train the model
trainer.train()
