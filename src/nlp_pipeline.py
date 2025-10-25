#------------- NLP Model trained with bioBert library ----------------#

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, IntervalStrategy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import pandas as pd
from datasets import Dataset
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv("data/500_labeled_nodules.csv")
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

train_dataset = Dataset.from_pandas(train_df.rename(columns={'label_binary': 'labels'})).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df.rename(columns={'label_binary': 'labels'})).map(tokenize, batched=True)
# Then change the set_format line:
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Training arguments
args = TrainingArguments(
    output_dir="./models/biobert_nodule",
    eval_strategy="epoch", 
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    label_names=['labels']
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

# Calculate Precision(TP / (TP + FP)), Recall(TP / (TP + FN)), and F1-score(2 × (Precision × Recall) / (Precision + Recall))

# --- Prediction & Evaluation Combined ---
predictions_output = trainer.predict(test_dataset)
predictions = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids
probs = softmax(predictions_output.predictions, axis=1)[:, 1]

# --- Adjust threshold for better recall (tune this manually) ---
threshold = 0.4
predictions_adj = (probs >= threshold).astype(int)

# --- Compute basic metrics ---
accuracy = accuracy_score(true_labels, predictions_adj)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions_adj, average='macro')
class_metrics = precision_recall_fscore_support(true_labels, predictions_adj, average=None, labels=[0, 1])

# Unpack metrics for both classes
class_0_precision, class_1_precision = class_metrics[0]
class_0_recall, class_1_recall = class_metrics[1]
class_0_f1, class_1_f1 = class_metrics[2]
class_0_support, class_1_support = class_metrics[3]

# --- Print summary ---
print("\n--- Model Performance Metrics ---")
print(f"**Overall Accuracy:** {accuracy:.4f}")
print("---------------------------------")
print(f"**Macro Average Precision:** {precision:.4f}")
print(f"**Macro Average Recall:** {recall:.4f}")
print(f"**Macro Average F1-Score:** {f1:.4f}")
print("---------------------------------")
print("  Class 0 (Negative) Metrics:")
print(f"    Precision: {class_0_precision:.4f}, Recall: {class_0_recall:.4f}, F1: {class_0_f1:.4f}, Support: {class_0_support}")
print("  Class 1 (Positive) Metrics:")
print(f"    Precision: {class_1_precision:.4f}, Recall: {class_1_recall:.4f}, F1: {class_1_f1:.4f}, Support: {class_1_support}")
print("---------------------------------\n")

print(classification_report(true_labels, predictions_adj))
print("---------------------------------\n")

# --- Confusion Matrix ---
cm = confusion_matrix(true_labels, predictions_adj)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Nodules absent', 'Nodules present'],
            yticklabels=['Nodules absent', 'Nodules present'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("test/confusion_matrix.png")
plt.show()
