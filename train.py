"""
XLM-RoBERTa Fine-tuning Script for Joint Intent Classification + NER
Optimized for Google Colab with GPU support.

Run in Colab:
    !git clone <your_repo_url>
    %cd NLP_training
    !python train.py --output_dir ./xlm_roberta_model --num_epochs 3
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

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "xlm-roberta-base"
DATA_DIR = "/content/drive/MyDrive/data"
MAX_LENGTH = 128

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
# Multi-Task Model: Intent Classification + NER
# ============================================================================
class XLMRobertaForIntentAndNER(nn.Module):
    """
    Joint Intent Classification + Named Entity Recognition model.

    Architecture:
    - Shared encoder: XLM-RoBERTa base
    - Task 1: Intent classification (sentence-level)
    - Task 2: NER (token-level)
    """
    def __init__(self, model_name, num_labels_intent, num_labels_ner, dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Intent classification head (sentence-level)
        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels_intent),
        )

        # NER head (token-level)
        self.ner_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels_ner),
        )

    def forward(self, input_ids, attention_mask, ner_labels=None, intent_labels=None):
        # Shared encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Intent classification: use [CLS] token (first token)
        cls_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)
        intent_logits = self.intent_classifier(cls_output)  # (batch_size, num_intent_labels)

        # NER: token-level predictions
        ner_logits = self.ner_classifier(sequence_output)  # (batch_size, seq_len, num_ner_labels)

        loss = None
        if ner_labels is not None and intent_labels is not None:
            # Loss weights (increase NER weight if it's undershooting)
            intent_weight = 1.0
            ner_weight = 1.5  # Slightly higher weight to focus on NER accuracy

            # Intent loss
            loss_fn_intent = nn.CrossEntropyLoss()
            intent_loss = loss_fn_intent(intent_logits, intent_labels)

            # NER loss (ignore -100 labels automatically)
            loss_fn_ner = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = loss_fn_ner(
                ner_logits.view(-1, NUM_LABELS_NER),
                ner_labels.view(-1)
            )

            # Weighted combined loss
            loss = intent_weight * intent_loss + ner_weight * ner_loss

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "ner_logits": ner_logits,
        }


# ============================================================================
# Data Processing & Metrics
# ============================================================================
def prepare_dataset(datasets_dict, tokenizer):
    """Process raw datasets for training."""
    def preprocess_function(examples):
        # Input: raw tokens (already tokenized by data generator)
        # We need to align them properly

        # Use the text field and tokenize directly
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=False,
        )

        # Align NER tags to new tokenization
        # Since we already have aligned tags, we just need to pad them
        aligned_ner_tags = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            # Pad NER tags to match tokenized length
            padded_tags = ner_tags + [-100] * (MAX_LENGTH - len(ner_tags))
            padded_tags = padded_tags[:MAX_LENGTH]
            aligned_ner_tags.append(padded_tags)

        encodings["ner_labels"] = aligned_ner_tags
        encodings["intent_labels"] = examples["intent"]

        return encodings

    # Process all splits
    processed_datasets = DatasetDict({
        split: dataset.map(
            preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
        )
        for split, dataset in datasets_dict.items()
    })

    return processed_datasets


def compute_metrics(eval_pred):
    """Compute F1 and accuracy metrics for both tasks."""
    predictions, labels = eval_pred

    # Extract predictions and labels for NER and Intent
    ner_predictions = predictions[0]  # (batch_size, seq_len, num_ner_labels)
    intent_predictions = predictions[1]  # (batch_size, num_intent_labels)

    ner_labels_batch = labels[0]  # (batch_size, seq_len)
    intent_labels_batch = labels[1]  # (batch_size,)

    # Convert torch tensors to numpy if needed
    if hasattr(ner_predictions, 'cpu'):
        ner_predictions = ner_predictions.cpu().numpy()
    if hasattr(intent_predictions, 'cpu'):
        intent_predictions = intent_predictions.cpu().numpy()
    if hasattr(ner_labels_batch, 'cpu'):
        ner_labels_batch = ner_labels_batch.cpu().numpy()
    if hasattr(intent_labels_batch, 'cpu'):
        intent_labels_batch = intent_labels_batch.cpu().numpy()

    results = {}

    # ---- Intent Classification Metrics ----
    intent_preds = np.argmax(intent_predictions, axis=1)
    intent_accuracy = accuracy_score(intent_labels_batch, intent_preds)
    results["intent_accuracy"] = intent_accuracy

    # ---- NER Metrics ----
    # Flatten and filter out -100 (ignored) labels
    ner_preds_flat = np.argmax(ner_predictions, axis=2).flatten()
    ner_labels_flat = ner_labels_batch.flatten()

    # Remove -100 labels
    mask = ner_labels_flat != -100
    ner_preds_filtered = ner_preds_flat[mask]
    ner_labels_filtered = ner_labels_flat[mask]

    if len(ner_labels_filtered) > 0:
        ner_f1 = f1_score(
            ner_labels_filtered,
            ner_preds_filtered,
            average="weighted",
            zero_division=0,
        )
        results["ner_f1"] = ner_f1

    return results


# ============================================================================
# Custom Trainer for Multi-Task Learning
# ============================================================================
class MultiTaskTrainer(Trainer):
    """Custom trainer that handles multi-task learning."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            ner_labels=inputs.get("ner_labels"),
            intent_labels=inputs.get("intent_labels"),
        )
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to extract logits in the correct format."""
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    ner_labels=inputs.get("ner_labels"),
                    intent_labels=inputs.get("intent_labels"),
                )
            loss = outputs["loss"].detach()

        if prediction_loss_only:
            return (loss, None, None)

        # Extract logits - keep as torch tensors for accelerator
        ner_logits = outputs["ner_logits"].detach()
        intent_logits = outputs["intent_logits"].detach()

        # Stack logits: [ner_logits, intent_logits]
        predictions = (ner_logits, intent_logits)

        # Extract labels - keep as torch tensors
        ner_labels = inputs.get("ner_labels")
        intent_labels = inputs.get("intent_labels")

        labels = (ner_labels, intent_labels)

        return loss, predictions, labels


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for Intent + NER")
    parser.add_argument("--output_dir", default="./xlm_roberta_model", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("XLM-RoBERTa Joint Intent Classification + NER Training")
    print("="*70 + "\n")

    # ---- Load Datasets ----
    print("📦 Loading datasets...")
    try:
        train_dataset = load_from_disk(f"{DATA_DIR}/train")
        val_dataset = load_from_disk(f"{DATA_DIR}/val")
        test_dataset = load_from_disk(f"{DATA_DIR}/test")

        datasets = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })

        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return

    # ---- Load Tokenizer ----
    print(f"\n🤖 Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Prepare Datasets ----
    print("\n⚙️  Preparing datasets...")
    processed_datasets = prepare_dataset(datasets, tokenizer)

    # ---- Initialize Model ----
    print(f"\n🏗️  Initializing model ({MODEL_NAME})...")
    model = XLMRobertaForIntentAndNER(
        model_name=MODEL_NAME,
        num_labels_intent=NUM_LABELS_INTENT,
        num_labels_ner=NUM_LABELS_NER,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Using device: {device}")

    # ---- Training Arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_ner_f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=100,
        log_level="info",
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    )

    # ---- Initialize Trainer ----
    print("\n🎓 Setting up trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ],
    )

    # ---- Train ----
    print("\n" + "="*70)
    print("🚀 Starting training...")
    print("="*70 + "\n")

    trainer.train()

    # ---- Evaluate on Test Set ----
    print("\n" + "="*70)
    print("📊 Evaluating on test set...")
    print("="*70 + "\n")

    test_results = trainer.evaluate(processed_datasets["test"])
    print(f"\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # ---- Save Model ----
    print(f"\n💾 Saving model to {args.output_dir}...")
    
    # Save encoder with pretrained format
    model.encoder.save_pretrained(args.output_dir)
    
    # Save complete model state_dict (including classifier heads) as safetensors
    from safetensors.torch import save_file
    model_state = model.state_dict()
    save_file(model_state, f"{args.output_dir}/model.safetensors")
    print(f"   ✓ Saved full model state_dict with {len(model_state)} weight tensors")
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Save config with model architecture info
    config = {
        "model_name": MODEL_NAME,
        "num_labels_intent": NUM_LABELS_INTENT,
        "num_labels_ner": NUM_LABELS_NER,
        "max_length": MAX_LENGTH,
        "intent_map": INTENT_MAP,
        "tag_map": TAG_MAP,
    }
    with open(f"{args.output_dir}/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"   ✓ Saved model config")

    # Save results
    with open(f"{args.output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n✅ Training complete!")
    print(f"   Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
