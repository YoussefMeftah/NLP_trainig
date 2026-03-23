"""
Model Prediction Diagnostic
Debug what the model is actually predicting.

Usage:
    python diagnose_predictions.py --model_dir ./model/checkpoint-1395
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from collections import Counter

DATA_DIR = "./data"
MAX_LENGTH = 128

with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)
with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}
ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}


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


def diagnose_predictions(model_path):
    """Diagnose what the model is predicting."""
    print("\n" + "="*70)
    print("🔍 MODEL PREDICTION DIAGNOSTIC")
    print("="*70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = XLMRobertaForIntentAndNER(
            model_name=model_path,
            num_labels_intent=len(INTENT_MAP),
            num_labels_ner=len(TAG_MAP),
        )
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load test data
    print("Loading test dataset...")
    try:
        test_dataset = load_from_disk(f"{DATA_DIR}/test")
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples\n")
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return

    # Test on first few samples
    print("="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70 + "\n")

    all_ner_preds = Counter()
    all_ner_true = Counter()

    for idx in range(min(5, len(test_dataset))):
        sample = test_dataset[idx]
        text = sample.get("text", "[no text]")

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )

        # Get predictions
        intent_logits = outputs["intent_logits"][0].cpu()
        ner_logits = outputs["ner_logits"][0].cpu()

        intent_probs = torch.softmax(intent_logits, dim=0)
        intent_pred = torch.argmax(intent_logits).item()
        intent_conf = intent_probs[intent_pred].item()

        ner_preds = torch.argmax(ner_logits, dim=1).numpy()

        print(f"Sample {idx}:")
        print(f"  Text: {text[:70]}...")
        print(f"  Intent: {ID_TO_INTENT[intent_pred]} (conf: {intent_conf:.3f})")
        print(f"  True Intent: {ID_TO_INTENT[sample['intent']]}")

        # NER predictions
        print(f"  NER Predictions:")
        ner_true = sample.get("ner_labels", sample.get("ner_tags", []))

        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].cpu().numpy())
        
        pred_counts = Counter()
        true_counts = Counter()

        for token_idx in range(len(tokens)):
            if token_idx < len(ner_preds):
                pred_tag = ner_preds[token_idx]
                pred_name = ID_TO_TAG.get(pred_tag, f"UNKNOWN_{pred_tag}")
                pred_counts[pred_name] += 1
                all_ner_preds[pred_name] += 1

            if token_idx < len(ner_true):
                true_tag = ner_true[token_idx]
                if true_tag != -100:
                    true_name = ID_TO_TAG.get(true_tag, f"UNKNOWN_{true_tag}")
                    true_counts[true_name] += 1
                    all_ner_true[true_name] += 1

        print(f"    Predicted tags: {dict(pred_counts)}")
        print(f"    True tags: {dict(true_counts)}")
        print()

    # Summary statistics
    print("="*70)
    print("OVERALL PREDICTION STATISTICS")
    print("="*70 + "\n")

    print("Model is predicting (all samples):")
    for tag, count in all_ner_preds.most_common():
        pct = 100 * count / sum(all_ner_preds.values())
        print(f"  {tag:20s}: {count:5d} ({pct:5.1f}%)")

    print("\nTrue labels in data:")
    for tag, count in all_ner_true.most_common():
        pct = 100 * count / sum(all_ner_true.values())
        print(f"  {tag:20s}: {count:5d} ({pct:5.1f}%)")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70 + "\n")

    o_pred_pct = 100 * all_ner_preds.get("O", 0) / sum(all_ner_preds.values())
    o_true_pct = 100 * all_ner_true.get("O", 0) / sum(all_ner_true.values())

    print(f"Model predicts 'O' (outside): {o_pred_pct:.1f}%")
    print(f"Actual 'O' in data: {o_true_pct:.1f}%\n")

    if o_pred_pct > 90:
        print("❌ CRITICAL ISSUE: Model is predicting almost everything as 'O'")
        print("   This means:")
        print("   1. ❌ NER model is NOT learning properly")
        print("   2. ❌ Model weights might not be loaded correctly")
        print("   3. ❌ NER head is stuck at initialization\n")
        print("   POSSIBLE CAUSES:")
        print("   • Model checkpoint doesn't contain trained NER weights")
        print("   • NER head wasn't trained (training crashed before NER converged)")
        print("   • Weights were saved/loaded incorrectly\n")
        print("   ACTION ITEMS:")
        print("   1. Check if training completed successfully")
        print("   2. Verify checkpoint has NER weights: check if model size is > 100MB")
        print("   3. Consider retraining with: python train.py --num_epochs 5")
    elif o_pred_pct > 70:
        print("⚠️  WARNING: Model is heavily biased toward 'O' predictions")
        print("   NER is underperforming. Likely causes:")
        print("   • Insufficient training data for NER")
        print("   • Too few training epochs")
        print("   • Class imbalance in NER tags")
    else:
        print("✅ Model shows reasonable NER predictions")

    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose model predictions")
    parser.add_argument("--model_dir", default="./xlm_roberta_model", help="Model directory")
    args = parser.parse_args()

    diagnose_predictions(args.model_dir)


if __name__ == "__main__":
    main()
