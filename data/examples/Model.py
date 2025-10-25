import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import re


# Function to preprocess text data
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()    # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()


# Custom dataset class for pulmonary nodule data
class PulmonaryNoduleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Function to compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Custom Trainer to support class weights
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor([1.51, 1.0]).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Main function to execute training
def main():
    # Load and preprocess data
    data = pd.read_csv('500_labeled_nodules.csv')
    data['findings_reduced'] = data['findings_reduced'].apply(preprocess_text)
    data = data.dropna(subset=['findings_reduced', 'label_binary'])

    texts = data['findings_reduced'].tolist()
    labels = data['label_binary'].tolist()

    # Split data into training and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Load BioBERT tokenizer and model
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_dataset = PulmonaryNoduleDataset(train_texts, train_labels, tokenizer, max_len=64)
    test_dataset = PulmonaryNoduleDataset(test_texts, test_labels, tokenizer, max_len=64)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        learning_rate=1e-5,
        dataloader_pin_memory=True if torch.cuda.is_available() else False
    )

    # Create Trainer with early stopping
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained('./biobert_pulmonary_nodule_model')
    tokenizer.save_pretrained('./biobert_pulmonary_nodule_model')

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)


if __name__ == "__main__":
    main()