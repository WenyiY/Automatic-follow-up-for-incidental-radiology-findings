from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from datasets import Dataset

# Load data
df = pd.read_csv("data/examples/500_labeled_nodules.csv")
df = df[['findings_reduced', 'label_binary']]
# Splits data into 80% training, 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_binary'], random_state=42)

# Load tokenizer & model
model_name = "dmis-lab/biobert-base-cased-v1.1"   # picks the pretrained BioBERT model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess each record
def tokenize(batch):
    return tokenizer(batch['findings_reduced'], padding=True, truncation=True, max_length=256)

# Converts pandas into Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df).map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'has_nodule'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'has_nodule'])

# Training arguments
args = TrainingArguments(
    output_dir="./models/biobert_nodule",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(metrics)
