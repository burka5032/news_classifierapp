import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert labels to numbers
labels = df["category"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["labels"] = df["category"].map(label2id)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Training setup (FIXED version)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train model
trainer.train()

# Save model
model.save_pretrained("news_ai_model")
tokenizer.save_pretrained("news_ai_model")

print("✅ news_ai_model created successfully!")