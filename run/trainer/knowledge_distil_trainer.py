import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Step 1: Choose a pre-trained teacher model and student model architecture
teacher_model_name = "opt-1.3b"
student_model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Step 2: Tokenize and batch your input text data
train_file = ""
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 3: Generate soft labels using the teacher model
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

# Step 4: Train the student model using the soft labels as targets
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)


def compute_loss(model, batch, alpha_ce=0.5, alpha_mlm=0.5, alpha_cos=0.0, temperature=2.0):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    soft_labels = torch.stack(train_soft_labels.pop(0))

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]

    # Temperature-scaled softmax
    student_probs = F.log_softmax(logits / temperature, dim=-1)
    teacher_probs = F.softmax(soft_labels / temperature, dim=-1)

    # Distillation loss (Lce)
    loss_ce = F.kl_div(student_probs, teacher_probs, reduction="batchmean")

    # Masked language modeling loss (Lmlm)
    loss_mlm = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))

    # Cosine embedding loss (Lcos)
    teacher_hidden_states = teacher_model.base_model(input_ids.cuda()).last_hidden_state.detach()
    student_hidden_states = outputs.hidden_states[-1]
    loss_cos = F.cosine_embedding_loss(student_hidden_states, teacher_hidden_states.cpu(),
                                       target=torch.ones(input_ids.size(0)))

    # Linear combination of the losses
    total_loss = alpha_ce * loss_ce + alpha_mlm * loss_mlm + alpha_cos * loss_cos
    return total_loss


training_args = TrainingArguments(
    output_dir="./model",
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
