"""
Google Colab Fine-tuning Script
Fine-tune the existing XLM-RoBERTa model on enriched dataset with paraphrasing + relative dates.

Run in Colab:
    !git clone <your_repo_url>
    %cd NLP_training
    !python fine_tune_colab.py
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys

# ============================================================================
# Google Colab Setup
# ============================================================================
def setup_colab():
    """Mount Google Drive if running in Colab."""
    if 'google.colab' in sys.modules:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            print("✓ Google Drive mounted at /content/drive")
            return True
        except Exception as e:
            print(f"⚠️  Colab mode detected but Drive mount failed: {e}")
            return False
    return False

# Try to mount Drive in Colab
IS_COLAB = setup_colab()

# ============================================================================
# Configuration & Setup
# ============================================================================
# Set data directory based on environment
if IS_COLAB:
    DATA_DIR = "/content/drive/MyDrive/data"
    MODEL_OUTPUT_DIR = "/content/drive/MyDrive/xlm_roberta_model_finetuned"
    OLD_MODEL_DIR = "/content/drive/MyDrive/xlm_roberta_model"
else:
    DATA_DIR = "./data"
    MODEL_OUTPUT_DIR = "./model_finetuned"
    OLD_MODEL_DIR = "./model"

MAX_LENGTH = 128

print(f"Running in {'Colab' if IS_COLAB else 'Local'} mode")
print(f"Data directory: {DATA_DIR}")
print(f"Model output directory: {MODEL_OUTPUT_DIR}")

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
            # Only load compatible weights
            model_state = model.state_dict()
            compatible_weights = {k: v for k, v in state_dict.items() if k in model_state}
            model.load_state_dict(compatible_weights, strict=False)
            print(f"✓ Loaded {len(compatible_weights)}/{len(state_dict)} weights from existing model")
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

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=2,  # Fine-tune for 2 epochs (not 3+)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    learning_rate=1e-5,  # Lower learning rate for fine-tuning
    weight_decay=0.01,
    logging_dir=f"{MODEL_OUTPUT_DIR}/logs",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="ner_f1",
    greater_is_better=True,
    save_total_limit=2,
    seed=42,
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
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
print(f"Device: {model.device}")

trainer.train()

# ============================================================================
# Evaluate on Test Set
# ============================================================================
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Intent Accuracy: {test_results.get('eval_intent_accuracy', 0):.4f}")
print(f"Test NER F1: {test_results.get('eval_ner_f1', 0):.4f}")

# ============================================================================
# Save Fine-tuned Model
# ============================================================================
print(f"\n💾 Saving fine-tuned model to {MODEL_OUTPUT_DIR}...")

# Create output directory
Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Save model weights using safetensors
from safetensors.torch import save_file
save_file(model.state_dict(), f"{MODEL_OUTPUT_DIR}/model.safetensors")

# Save tokenizer
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

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

# Save test results
with open(f"{MODEL_OUTPUT_DIR}/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print(f"✓ Model saved to {MODEL_OUTPUT_DIR}")
print(f"✓ Tokenizer saved")
print(f"✓ Config saved")
print(f"✓ Test results saved")

print("\n" + "=" * 70)
print("✅ FINE-TUNING COMPLETE!")
print("=" * 70)
print(f"Model location: {MODEL_OUTPUT_DIR}")
print(f"Test Results: {test_results}")
print("\nYou can now use this model for inference!")
