import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Step 1: Choose a pre-trained teacher run and student run architecture
teacher_model_name = "gpt2-large"
student_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Step 2: Tokenize and batch your input text data
train_file = ""  # todo: manage the data file
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 3: Generate soft labels using the teacher run
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).cuda()
teacher_model.eval()


def generate_soft_labels(batch):
    with torch.no_grad():
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        outputs = teacher_model(input_ids, attention_mask=attention_mask)
        return outputs.logits[:, :-1].detach().cpu()


train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
train_soft_labels = [generate_soft_labels(batch) for batch in train_dataloader]

# Step 4: Train the student run using the soft labels as targets
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)


def compute_loss(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    soft_labels = torch.stack(train_soft_labels.pop(0))
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]

    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits, dim=-1),
        torch.nn.functional.softmax(soft_labels, dim=-1),
        reduction="batchmean"
    )
    return loss


training_args = TrainingArguments(
    output_dir="./distilled_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    compute_loss=compute_loss,
)

trainer.train()
