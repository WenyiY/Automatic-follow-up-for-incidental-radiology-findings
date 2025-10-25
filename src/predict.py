#-- Input the raw file and give the output of pulmonary nodules present(1) and 
# pulmonary nodules absent(0)--#

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
# The path where training model saved the fine-tuned model and tokenizer
MODEL_PATH = "./models/biobert_nodule/checkpoint-153" 
INPUT_CSV_PATH = "./data/findings_nodule.csv"          # The file need to be predicted
OUTPUT_CSV_PATH = "./data/predictions_output.csv"      # Where to save the results
TEXT_COLUMN = 'text'   # The name of the column containing the text

# --- Load Model and Tokenizer ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() # Set model to evaluation mode

# --- Load Data and Prepare Dataset ---
print(f"Loading data from {INPUT_CSV_PATH}...")
new_df = pd.read_csv(INPUT_CSV_PATH)
data_to_predict = new_df.rename(columns={TEXT_COLUMN: 'text'})[['text']]

def tokenize_new_data(batch):
    # Ensure the tokenization matches what was used during training
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

predict_dataset = Dataset.from_pandas(data_to_predict).map(
    tokenize_new_data,
    batched=True,
    remove_columns=data_to_predict.columns.tolist()
)

predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# --- Generate Predictions ---
print("Generating predictions...")
# Use torch DataLoader for batching (similar to Trainer)
from torch.utils.data import DataLoader
data_loader = DataLoader(predict_dataset, batch_size=32) 

all_logits = []
with torch.no_grad(): # Disable gradient calculation for inference
    for batch in data_loader:
        # Move batch tensors to the device the model is on (usually CPU for simple inference)
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu())

# Combine all logits into a single numpy array
raw_logits = torch.cat(all_logits, dim=0).numpy()

# --- Process and Save Results ---
# Convert logits to binary prediction (0 or 1)
binary_predictions = np.argmax(raw_logits, axis=1)

# Convert logits to probabilities for confidence score
probabilities = F.softmax(torch.tensor(raw_logits), dim=1).numpy()
positive_probabilities = probabilities[:, 1] # Probability of Class 1

# Add the new prediction columns to the original DataFrame
new_df['predicted_label'] = binary_predictions
new_df['positive_probability'] = positive_probabilities

# Save the updated DataFrame
new_df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n--- Predictions Complete ---")
print(f"Successfully saved {len(new_df)} predictions to '{OUTPUT_CSV_PATH}'")
print(new_df[[TEXT_COLUMN, 'predicted_label', 'positive_probability']].head())