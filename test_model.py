"""
Model Testing & Inference Script
Test the trained XLM-RoBERTa model and diagnose data/model issues.

Usage:
    python test_model.py --model_dir ./xlm_roberta_model --test_samples 50
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# ============================================================================
# Configuration & Setup
# ============================================================================
DATA_DIR = "/content/drive/MyDrive/data"
MAX_LENGTH = 128

# Load mappings
with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)
with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}
ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}

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
            intent_weight = 1.0
            ner_weight = 1.5

            loss_fn_intent = nn.CrossEntropyLoss()
            intent_loss = loss_fn_intent(intent_logits, intent_labels)

            loss_fn_ner = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = loss_fn_ner(
                ner_logits.view(-1, len(TAG_MAP)),
                ner_labels.view(-1)
            )

            loss = intent_weight * intent_loss + ner_weight * ner_loss

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "ner_logits": ner_logits,
        }


# ============================================================================
# Inference & Testing
# ============================================================================
class ModelTester:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create model architecture
        self.model = XLMRobertaForIntentAndNER(
            model_name="xlm-roberta-base",
            num_labels_intent=len(INTENT_MAP),
            num_labels_ner=len(TAG_MAP),
        )
        
        # Load fine-tuned weights from model.safetensors
        from safetensors.torch import load_file
        import os
        weights_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weights_path):
            weights = load_file(weights_path)
            self.model.load_state_dict(weights)
        else:
            print(f"⚠️  Warning: model.safetensors not found at {weights_path}")
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """Predict intent and NER tags for input text."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )

        # Get predictions
        intent_pred = torch.argmax(outputs["intent_logits"], dim=1).item()
        ner_preds = torch.argmax(outputs["ner_logits"], dim=2)[0].cpu().numpy()

        # Decode tokens and align with NER tags
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        offset_mapping = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )["offset_mapping"]

        # Remove special tokens
        valid_predictions = []
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start != end:  # Skip special tokens
                tag_id = ner_preds[token_idx]
                tag = ID_TO_TAG.get(tag_id, "O")
                valid_predictions.append({
                    "token": tokens[token_idx],
                    "tag": tag,
                    "confidence": float(torch.softmax(outputs["ner_logits"], dim=2)[0, token_idx].max().cpu().numpy())
                })

        return {
            "text": text,
            "intent": ID_TO_INTENT[intent_pred],
            "intent_id": intent_pred,
            "ner_tags": valid_predictions,
        }

    def evaluate_dataset(self, dataset, max_samples=None):
        """Evaluate model on a dataset."""
        all_intent_preds = []
        all_intent_true = []
        all_ner_preds = []
        all_ner_true = []

        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for sample in samples:
            # Decode from token IDs if needed
            if "text" in sample:
                text = sample["text"]
            else:
                # Try to reconstruct from tokens (less ideal but possible)
                text = f"[encoded sample]"

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )

            # Intent predictions
            intent_pred = torch.argmax(outputs["intent_logits"], dim=1).item()
            intent_true = sample["intent"]
            all_intent_preds.append(intent_pred)
            all_intent_true.append(intent_true)

            # NER predictions
            ner_preds = torch.argmax(outputs["ner_logits"], dim=2)[0].cpu().numpy()
            
            # Handle both "ner_tags" (raw) and "ner_labels" (preprocessed) field names
            if "ner_labels" in sample:
                ner_true = sample["ner_labels"]
            elif "ner_tags" in sample:
                ner_true = sample["ner_tags"]
            else:
                print(f"⚠️  Warning: Neither 'ner_labels' nor 'ner_tags' found in sample. Keys: {list(sample.keys())}")
                continue

            # Ensure arrays are proper length
            ner_preds = ner_preds[:len(ner_true)]
            
            # Collect all valid (non-padding) NER predictions
            for token_idx in range(len(ner_true)):
                if token_idx < len(ner_preds) and ner_true[token_idx] != -100:
                    all_ner_preds.append(ner_preds[token_idx])
                    all_ner_true.append(ner_true[token_idx])

        return {
            "intent_preds": np.array(all_intent_preds),
            "intent_true": np.array(all_intent_true),
            "ner_preds": np.array(all_ner_preds) if len(all_ner_preds) > 0 else np.array([]),
            "ner_true": np.array(all_ner_true) if len(all_ner_true) > 0 else np.array([]),
        }


# ============================================================================
# Evaluation & Diagnostics
# ============================================================================
def print_intent_metrics(preds, true):
    """Print detailed intent classification metrics."""
    print("\n" + "="*70)
    print("INTENT CLASSIFICATION METRICS")
    print("="*70)

    accuracy = accuracy_score(true, preds)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    print("\nPer-Intent Performance:")
    for intent_id, intent_name in ID_TO_INTENT.items():
        mask = true == intent_id
        if mask.sum() == 0:
            continue
        intent_acc = (preds[mask] == intent_id).mean()
        print(f"  {intent_name:30s}: {intent_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        true, preds,
        target_names=[ID_TO_INTENT[i] for i in range(len(INTENT_MAP))],
        zero_division=0
    ))


def print_ner_metrics(preds, true):
    """Print detailed NER metrics per tag."""
    print("\n" + "="*70)
    print("NER PERFORMANCE METRICS")
    print("="*70)

    # Handle empty predictions
    if len(preds) == 0 or len(true) == 0:
        print("\n⚠️  WARNING: No NER predictions found!")
        print("   This could mean:")
        print("   1. Dataset has no labeled NER tags (all -100)")
        print("   2. NER tags are not being extracted from samples")
        print("   3. Data loader is not returning ner_tags field")
        print("\n   Run: python analyze_data.py")
        print("   to check dataset structure.\n")
        return

    overall_f1 = f1_score(true, preds, average="weighted", zero_division=0)
    print(f"\nOverall F1 (weighted): {overall_f1:.4f}")

    print("\nPer-Tag Performance:")
    tags_with_data = False
    for tag_id, tag_name in ID_TO_TAG.items():
        mask = true == tag_id
        if mask.sum() == 0:
            continue

        tags_with_data = True
        pred_mask = preds == tag_id
        if pred_mask.sum() == 0:
            print(f"  {tag_name:20s}: NO PREDICTIONS (recall=0)")
            continue

        # Calculate per-tag metrics
        tp = ((preds == tag_id) & (true == tag_id)).sum()
        fp = ((preds == tag_id) & (true != tag_id)).sum()
        fn = ((preds != tag_id) & (true == tag_id)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  {tag_name:20s}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    if not tags_with_data:
        print("  (No tags found with data)")

    print("\nDetailed Classification Report:")
    # Only include labels that actually exist in the true labels
    unique_labels = np.unique(true)
    target_names = [ID_TO_TAG.get(int(label), f"UNKNOWN_{label}") for label in unique_labels]
    print(classification_report(
        true, preds,
        labels=unique_labels.astype(int).tolist(),
        target_names=target_names,
        zero_division=0
    ))


def print_dataset_diagnostics():
    """Analyze dataset characteristics."""
    print("\n" + "="*70)
    print("DATASET DIAGNOSTICS")
    print("="*70)

    try:
        train_dataset = load_from_disk(f"{DATA_DIR}/train")
        val_dataset = load_from_disk(f"{DATA_DIR}/val")
        test_dataset = load_from_disk(f"{DATA_DIR}/test")

        print(f"\nDataset Sizes:")
        print(f"  Train:  {len(train_dataset)} samples")
        print(f"  Val:    {len(val_dataset)} samples")
        print(f"  Test:   {len(test_dataset)} samples")

        # Intent distribution
        print(f"\nIntent Distribution (Train):")
        intent_counts = Counter(train_dataset["intent"])
        total = len(train_dataset)
        for intent_id in sorted(intent_counts.keys()):
            count = intent_counts[intent_id]
            pct = 100 * count / total
            print(f"  {ID_TO_INTENT[intent_id]:30s}: {count:5d} ({pct:5.1f}%)")

        # NER tag distribution
        print(f"\nNER Tag Distribution (Train - flattened):")
        all_tags = []
        for sample in train_dataset:
            all_tags.extend(sample["ner_tags"])
        tag_counts = Counter(all_tags)

        for tag_id in sorted(tag_counts.keys()):
            if tag_id == -100:
                continue  # Skip padding
            count = tag_counts[tag_id]
            pct = 100 * count / (len(all_tags) - tag_counts[-100])
            print(f"  {ID_TO_TAG[tag_id]:20s}: {count:5d} ({pct:5.1f}%)")

    except Exception as e:
        print(f"❌ Error loading datasets: {e}")


def print_recommendations(intent_metrics, ner_metrics, intent_acc_overall, ner_f1_overall):
    """Provide actionable recommendations based on metrics."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    issues = []

    # Intent classification checks
    if intent_acc_overall < 0.70:
        issues.append({
            "severity": "HIGH",
            "task": "Intent Classification",
            "metric": f"Accuracy: {intent_acc_overall:.4f}",
            "action": "❌ Poor intent classification. Consider: (1) Increase dataset size 2x-3x, (2) Review template diversity, (3) Add more synonyms"
        })
    elif intent_acc_overall < 0.85:
        issues.append({
            "severity": "MEDIUM",
            "task": "Intent Classification",
            "metric": f"Accuracy: {intent_acc_overall:.4f}",
            "action": "⚠️  Acceptable but improvable. Try: (1) Add 50% more samples, (2) Include edge cases"
        })
    else:
        issues.append({
            "severity": "LOW",
            "task": "Intent Classification",
            "metric": f"Accuracy: {intent_acc_overall:.4f}",
            "action": "✅ Good! Intent classification is working well."
        })

    # NER checks
    if ner_f1_overall < 0.70:
        issues.append({
            "severity": "HIGH",
            "task": "NER",
            "metric": f"F1: {ner_f1_overall:.4f}",
            "action": "❌ Poor NER performance. Consider: (1) Increase dataset size 3x-5x, (2) Add more date/category variations, (3) Check tag alignment issues"
        })
    elif ner_f1_overall < 0.80:
        issues.append({
            "severity": "MEDIUM",
            "task": "NER",
            "metric": f"F1: {ner_f1_overall:.4f}",
            "action": "⚠️  NER needs improvement. Try: (1) Add more diverse dates, (2) Include ambiguous cases, (3) 50% more training data"
        })
    else:
        issues.append({
            "severity": "LOW",
            "task": "NER",
            "metric": f"F1: {ner_f1_overall:.4f}",
            "action": "✅ Strong NER performance!"
        })

    # Class imbalance check
    print("\nIssues Found:\n")
    for issue in issues:
        print(f"[{issue['severity']}] {issue['task']} ({issue['metric']})")
        print(f"     {issue['action']}\n")

    print("\nGeneral Data Quality Checklist:")
    print("  [ ] All intents have >= 100 samples each")
    print("  [ ] All NER tags have >= 50 examples")
    print("  [ ] Dates vary across the 2023-2025 range")
    print("  [ ] Categories are diverse (not just one repeated)")
    print("  [ ] Synonyms are used effectively")
    print("  [ ] Test with real-world examples similar to production")


# ============================================================================
# Main Testing Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test XLM-RoBERTa Intent + NER Model")
    parser.add_argument("--model_dir", default="./xlm_roberta_model", help="Model directory")
    parser.add_argument("--test_samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("XLM-RoBERTa Model Testing & Evaluation")
    print("="*70 + "\n")

    # Initialize tester
    print(f"🔧 Loading model from {args.model_dir}...")
    try:
        tester = ModelTester(args.model_dir)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Dataset diagnostics
    print_dataset_diagnostics()

    # Load test dataset
    print(f"\n📊 Loading test dataset...")
    try:
        test_dataset = load_from_disk(f"{DATA_DIR}/test")
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return

    # Evaluate on test set
    print(f"\n🔍 Evaluating on {min(args.test_samples or len(test_dataset), len(test_dataset))} samples...")
    results = tester.evaluate_dataset(test_dataset, max_samples=args.test_samples)

    # Print metrics
    print_intent_metrics(results["intent_preds"], results["intent_true"])
    print_ner_metrics(results["ner_preds"], results["ner_true"])

    # Recommendations
    intent_acc = accuracy_score(results["intent_true"], results["intent_preds"])
    
    # Only compute NER F1 if we have NER predictions
    if len(results["ner_true"]) > 0:
        ner_f1 = f1_score(results["ner_true"], results["ner_preds"], average="weighted", zero_division=0)
    else:
        ner_f1 = 0.0
        print("\n⚠️  NER evaluation skipped: No valid predictions found")
    
    print_recommendations({}, {}, intent_acc, ner_f1)

    # Interactive mode
    if args.interactive:
        print("\n" + "="*70)
        print("INTERACTIVE TESTING MODE")
        print("="*70)
        print("\nEnter test sentences (or 'quit' to exit):\n")

        while True:
            text = input(">>> ").strip()
            if text.lower() == "quit":
                break
            if not text:
                continue

            result = tester.predict(text)
            print(f"\n📝 Text: {result['text']}")
            print(f"🎯 Intent: {result['intent']} (ID: {result['intent_id']})")
            print(f"🏷️  NER Tags:")
            for token_info in result["ner_tags"]:
                if token_info["tag"] != "O":
                    print(f"   {token_info['token']:15s} {token_info['tag']:20s} (conf: {token_info['confidence']:.3f})")
            print()

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
