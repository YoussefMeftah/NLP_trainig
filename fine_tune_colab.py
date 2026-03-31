"""
Google Colab Fine-tuning Script
Fine-tune the XLM-RoBERTa model using data from Google Drive and save results to Google Drive.

Run in Colab:
    !git clone <your_repo_url>
    %cd NLP_training
    !python fine_tune_colab.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys
import shutil
import glob

# ============================================================================
# Configuration - Google Drive Paths
# ============================================================================
DATA_DIR = "/content/drive/MyDrive/data"
MODEL_OUTPUT_DIR = "/content/drive/MyDrive/xlm_roberta_model_finetuned"
OLD_MODEL_DIR = "/content/drive/MyDrive/xlm_roberta_model"

MAX_LENGTH = 128

print("\n" + "=" * 70)
print("GOOGLE COLAB FINE-TUNING")
print("=" * 70)
print(f"Input data:        {DATA_DIR}")
print(f"Output model:      {MODEL_OUTPUT_DIR}")
print("=" * 70 + "\n")

# Load tag and intent mappings
with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)

with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}
ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}

NUM_LABELS_NER = len(TAG_MAP)
NUM_LABELS_INTENT = len(INTENT_MAP)

print(f"NER Classes: {NUM_LABELS_NER} ({TAG_MAP})")
print(f"Intent Classes: {NUM_LABELS_INTENT} ({INTENT_MAP})")

# ============================================================================
# Model Definition (same as training)
# ============================================================================
class XLMRobertaForIntentAndNER(nn.Module):
    def __init__(self, model_name, num_labels_intent, num_labels_ner, dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels_intent),
        )

        self.ner_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels_ner),
        )

    def forward(self, input_ids, attention_mask, ner_labels=None, intent_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]
        intent_logits = self.intent_classifier(cls_output)

        ner_logits = self.ner_classifier(sequence_output)

        loss = None
        if ner_labels is not None and intent_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = loss_fn(ner_logits.view(-1, NUM_LABELS_NER), ner_labels.view(-1))
            intent_loss = loss_fn(intent_logits, intent_labels)
            loss = ner_loss + intent_loss

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "ner_logits": ner_logits,
        }


# ============================================================================
# Evaluation Metrics
# ============================================================================
def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    pred_logits, ner_preds = predictions

    # Intent classification metrics
    intent_preds = np.argmax(pred_logits, axis=1)
    intent_labels_true = labels[:, 0]  # First element is intent label
    intent_acc = accuracy_score(intent_labels_true, intent_preds)

    # NER metrics (ignoring -100 labels)
    ner_labels_true = labels[:, 1:]
    ner_preds_flat = []
    ner_labels_flat = []

    for i in range(len(ner_preds)):
        for j in range(ner_preds[i].shape[0]):
            pred_tag = np.argmax(ner_preds[i][j])
            true_tag = ner_labels_true[i][j]
            if true_tag != -100:  # Ignore padding/special tokens
                ner_preds_flat.append(pred_tag)
                ner_labels_flat.append(true_tag)

    ner_f1 = f1_score(ner_labels_flat, ner_preds_flat, average="weighted", zero_division=0)

    return {
        "intent_accuracy": intent_acc,
        "ner_f1": ner_f1,
    }


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================
print("\n📦 Loading datasets...")
train_dataset = load_from_disk(f"{DATA_DIR}/train")
val_dataset = load_from_disk(f"{DATA_DIR}/val")
test_dataset = load_from_disk(f"{DATA_DIR}/test")

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_data(examples):
    """Prepare data for model input"""
    processed = {
        "input_ids": examples["tokens"],
        "attention_mask": [[1 if token != 0 else 0 for token in seq] for seq in examples["tokens"]],
        "intent_labels": examples["intent"],
        "ner_labels": examples["ner_tags"],
    }
    return processed

train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=["text", "tokens", "ner_tags", "intent"])
val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=["text", "tokens", "ner_tags", "intent"])
test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=["text", "tokens", "ner_tags", "intent"])

# ============================================================================
# Load Existing Model
# ============================================================================
print("\n🔄 Loading existing model for fine-tuning...")
try:
    # Try to load model config from multiple possible locations
    config_path = None
    for potential_path in [
        f"{OLD_MODEL_DIR}/model_config.json",
        f"{OLD_MODEL_DIR}/config.json"
    ]:
        if Path(potential_path).exists():
            config_path = potential_path
            break
    
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("model_name", "xlm-roberta-base")
        num_labels_intent = config.get("num_labels_intent", NUM_LABELS_INTENT)
        num_labels_ner = config.get("num_labels_ner", NUM_LABELS_NER)
    else:
        raise FileNotFoundError("No config file found")

    # Initialize model
    model = XLMRobertaForIntentAndNER(
        model_name=model_name,
        num_labels_intent=num_labels_intent,
        num_labels_ner=num_labels_ner
    )

    # Try to load existing weights from safetensors
    try:
        from safetensors.torch import load_file
        model_path = f"{OLD_MODEL_DIR}/model.safetensors"
        if Path(model_path).exists():
            state_dict = load_file(model_path)
            # Load only encoder weights (dimensions should match)
            encoder_weights = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
            if encoder_weights:
                model.encoder.load_state_dict(encoder_weights, strict=False)
                print(f"✓ Loaded encoder weights from existing model ({len(encoder_weights)} layers)")
                print("⚠️  Classifier heads reinitialized (NER classes changed from 7 → 9)")
            else:
                print("⚠️  No encoder weights found, will train from scratch")
        else:
            print("⚠️  model.safetensors not found, will train from scratch")
    except Exception as e:
        print(f"⚠️  Could not load weights: {e}. Will train from scratch.")

    print("✓ Model initialized successfully")

except Exception as e:
    print(f"⚠️  Error loading existing model: {e}")
    print("Creating new model from scratch...")
    model = XLMRobertaForIntentAndNER(
        model_name="xlm-roberta-base",
        num_labels_intent=NUM_LABELS_INTENT,
        num_labels_ner=NUM_LABELS_NER
    )

# ============================================================================
# Custom Data Collator
# ============================================================================
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DataCollator:
    """Custom collator for variable-length sequences"""
    tokenizer: object

    def __call__(self, batch: List[Dict]) -> Dict:
        max_length = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        intent_labels = []
        ner_labels = []

        for example in batch:
            # Pad sequences
            input_id = example["input_ids"] + [0] * (max_length - len(example["input_ids"]))
            attn_mask = example["attention_mask"] + [0] * (max_length - len(example["attention_mask"]))
            ner_label = example["ner_labels"] + [-100] * (max_length - len(example["ner_labels"]))

            input_ids.append(input_id)
            attention_mask.append(attn_mask)
            intent_labels.append(example["intent_labels"])
            ner_labels.append(ner_label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "intent_labels": torch.tensor(intent_labels),
            "ner_labels": torch.tensor(ner_labels),
        }

data_collator = DataCollator(tokenizer)

# ============================================================================
# Fine-tuning Arguments & Trainer
# ============================================================================
print("\n⚙️  Setting up fine-tuning...")

# Create output directory BEFORE training (so Trainer can save checkpoints)
Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"✓ Created output directory: {MODEL_OUTPUT_DIR}")

# Clean up old checkpoints to free space
import shutil
import glob
checkpoint_pattern = f"{MODEL_OUTPUT_DIR}/checkpoint-*"
old_checkpoints = glob.glob(checkpoint_pattern)
for checkpoint in old_checkpoints:
    try:
        shutil.rmtree(checkpoint)
        print(f"✓ Removed old checkpoint: {checkpoint}")
    except Exception as e:
        print(f"⚠️  Could not remove {checkpoint}: {e}")

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",  # Evaluate only at the end of each epoch
    save_strategy="epoch",  # Save only at the end of each epoch
    save_total_limit=1,  # Keep only the best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=42,
    report_to="none",
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,
    remove_unused_columns=True,  # Remove unnecessary columns to save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ============================================================================
# Fine-tune Model
# ============================================================================
print("\n🚀 Starting fine-tuning...")

trainer.train()

# ============================================================================
# Evaluate on Test Set
# ============================================================================
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Intent Accuracy: {test_results.get('eval_intent_accuracy', 0):.4f}")
print(f"Test NER F1: {test_results.get('eval_ner_f1', 0):.4f}")

# ============================================================================
# Save Fine-tuned Model to Google Drive
# ============================================================================
print(f"\n💾 Saving fine-tuned model to Google Drive...")
print(f"   Location: {MODEL_OUTPUT_DIR}")

# Save model weights using safetensors
from safetensors.torch import save_file
save_file(model.state_dict(), f"{MODEL_OUTPUT_DIR}/model.safetensors")
print(f"   ✓ Model weights saved (model.safetensors)")

# Save tokenizer
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"   ✓ Tokenizer saved")

# Save config
config = {
    "model_name": "xlm-roberta-base",
    "num_labels_intent": NUM_LABELS_INTENT,
    "num_labels_ner": NUM_LABELS_NER,
    "max_length": MAX_LENGTH,
    "intent_map": INTENT_MAP,
    "tag_map": TAG_MAP,
}

with open(f"{MODEL_OUTPUT_DIR}/model_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"   ✓ Config saved (model_config.json)")

# Save test results
with open(f"{MODEL_OUTPUT_DIR}/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)
print(f"   ✓ Test results saved (test_results.json)")

print("\n" + "=" * 70)
print("✅ FINE-TUNING COMPLETE!")
print("=" * 70)
print("\n📂 Output Location (Google Drive):")
print(f"   /My Drive/xlm_roberta_model_finetuned/")
print("\n📊 Final Results:")
print(f"   Loss:     {test_results.get('eval_loss', 'N/A')}")
print(f"   Epoch:    {test_results.get('epoch', 'N/A')}")
print("\n" + "=" * 70)
