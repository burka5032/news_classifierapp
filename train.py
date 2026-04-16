import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert labels
labels = df["category"].unique().tolist()
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

df["labels"] = df["category"].map(label2id)

dataset = Dataset.from_pandas(df)

# Load BERT model
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Training setup
args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("news_ai_model")
tokenizer.save_pretrained("news_ai_model")

print("REAL NEWS AI TRAINED")