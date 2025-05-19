from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Load dataset
dataset = load_dataset("text", data_files="train.txt", split="train")

# Load model and tokenizer
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Fix padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./phi2_woocommerce_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=5,
    save_steps=10,
    save_total_limit=1
)

# Causal LM collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
trainer.train()
