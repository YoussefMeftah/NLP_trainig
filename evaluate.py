"""
Comprehensive Testing & Evaluation Script for the XLM-RoBERTa Joint Intent + NER Model.

Usage:
    # Dataset analysis only (no model required):
    python evaluate.py --data_dir ./data --output_dir ./evaluation_output

    # Full model evaluation (requires a trained model):
    python evaluate.py --model_dir ./xlm_roberta_model --data_dir ./data --output_dir ./evaluation_output

    # Skip plots (useful in headless / Colab environments):
    python evaluate.py --model_dir ./xlm_roberta_model --no_plots
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily so dataset-only mode works
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (safe for servers / Colab)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Constants & Default Mappings
# ============================================================================

DEFAULT_INTENT_MAP = {
    "get_sales_by_article": 0,
    "get_sales_by_client": 1,
    "get_purchases_by_article": 2,
    "get_purchases_by_supplier": 3,
    "get_payments": 4,
}

DEFAULT_TAG_MAP = {
    "O": 0,
    "B-DATE_START": 1,
    "I-DATE_START": 2,
    "B-DATE_END": 3,
    "I-DATE_END": 4,
    "B-CATEGORY": 5,
    "I-CATEGORY": 6,
}

# Manual test cases covering normal, edge, and abbreviation scenarios
MANUAL_TEST_CASES = [
    # --- Normal cases ---
    {
        "text": "afficher les ventes par article du 2024-01-01 au 2024-12-31",
        "expected_intent": "get_sales_by_article",
        "expected_entities": [("2024-01-01", "DATE_START"), ("2024-12-31", "DATE_END")],
    },
    {
        "text": "ventes par client entre 2023-06-01 et 2023-12-31",
        "expected_intent": "get_sales_by_client",
        "expected_entities": [("2023-06-01", "DATE_START"), ("2023-12-31", "DATE_END")],
    },
    {
        "text": "afficher les achats par fournisseur du 2024-03-01 au 2024-09-30",
        "expected_intent": "get_purchases_by_supplier",
        "expected_entities": [("2024-03-01", "DATE_START"), ("2024-09-30", "DATE_END")],
    },
    {
        "text": "répartition des paiements par client entre 2024-01-01 et 2024-06-30",
        "expected_intent": "get_payments",
        "expected_entities": [
            ("2024-01-01", "DATE_START"),
            ("2024-06-30", "DATE_END"),
            ("client", "CATEGORY"),
        ],
    },
    # --- Synonym / variation cases ---
    {
        "text": "chiffre d'affaires par produit du 2023-07-01 au 2023-07-31",
        "expected_intent": "get_sales_by_article",
        "expected_entities": [("2023-07-01", "DATE_START"), ("2023-07-31", "DATE_END")],
    },
    {
        "text": "statistiques des approvisionnements articles du 2024-01-01 au 2024-03-31",
        "expected_intent": "get_purchases_by_article",
        "expected_entities": [("2024-01-01", "DATE_START"), ("2024-03-31", "DATE_END")],
    },
    # --- Abbreviation edge cases ---
    {
        "text": "ventes par clt du 2024-01-01 au 2024-12-31",
        "expected_intent": "get_sales_by_client",
        "expected_entities": [("2024-01-01", "DATE_START"), ("2024-12-31", "DATE_END")],
    },
    {
        "text": "achats par frs entre 2023-01-01 et 2023-12-31",
        "expected_intent": "get_purchases_by_supplier",
        "expected_entities": [("2023-01-01", "DATE_START"), ("2023-12-31", "DATE_END")],
    },
    # --- Date format edge cases ---
    {
        "text": "statistiques des ventes articles du 2025-11-01 au 2025-11-30",
        "expected_intent": "get_sales_by_article",
        "expected_entities": [("2025-11-01", "DATE_START"), ("2025-11-30", "DATE_END")],
    },
    {
        "text": "donne moi les paiements ventilés par mode du 2024-02-01 au 2024-02-29",
        "expected_intent": "get_payments",
        "expected_entities": [
            ("2024-02-01", "DATE_START"),
            ("2024-02-29", "DATE_END"),
            ("mode", "CATEGORY"),
        ],
    },
]


# ============================================================================
# Utility helpers
# ============================================================================

def _load_mappings(data_dir: str):
    """Load TAG_MAP and INTENT_MAP from data directory, falling back to defaults."""
    tag_map = DEFAULT_TAG_MAP.copy()
    intent_map = DEFAULT_INTENT_MAP.copy()

    tag_path = Path(data_dir) / "tag_mapping.json"
    intent_path = Path(data_dir) / "intent_mapping.json"

    if tag_path.exists():
        with open(tag_path) as f:
            tag_map = json.load(f)
    else:
        print(f"  ⚠️  tag_mapping.json not found at {tag_path}, using defaults.")

    if intent_path.exists():
        with open(intent_path) as f:
            intent_map = json.load(f)
    else:
        print(f"  ⚠️  intent_mapping.json not found at {intent_path}, using defaults.")

    return tag_map, intent_map


def _save_figure(fig, output_dir: str, filename: str):
    """Save a matplotlib figure and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  💾 Saved plot: {path}")


# ============================================================================
# Section 1 – Dataset Analysis
# ============================================================================

class DatasetAnalyzer:
    """Analyse the raw HuggingFace datasets stored on disk."""

    def __init__(self, data_dir: str, tag_map: dict, intent_map: dict):
        self.data_dir = data_dir
        self.tag_map = tag_map
        self.intent_map = intent_map
        self.id_to_tag = {v: k for k, v in tag_map.items()}
        self.id_to_intent = {v: k for k, v in intent_map.items()}
        self.splits = {}

    # ------------------------------------------------------------------
    def load(self):
        """Load train / val / test splits from disk."""
        if not DATASETS_AVAILABLE:
            print("  ❌ 'datasets' package not installed – skipping dataset analysis.")
            return False

        for split in ("train", "val", "test"):
            split_path = Path(self.data_dir) / split
            if split_path.exists():
                self.splits[split] = load_from_disk(str(split_path))
                print(f"  ✅ Loaded {split}: {len(self.splits[split])} samples")
            else:
                print(f"  ⚠️  {split} split not found at {split_path}")

        return bool(self.splits)

    # ------------------------------------------------------------------
    def analyze(self) -> dict:
        """Run all analyses and return a summary dict."""
        if not self.splits:
            return {}

        summary = {}

        print("\n--- Class Distribution ---")
        summary["class_distribution"] = self._class_distribution()

        print("\n--- Token Length Distribution ---")
        summary["token_lengths"] = self._token_length_distribution()

        print("\n--- Entity Coverage ---")
        summary["entity_coverage"] = self._entity_coverage()

        print("\n--- Data Quality Assessment ---")
        summary["data_quality"] = self._data_quality()

        return summary

    # ------------------------------------------------------------------
    def _class_distribution(self) -> dict:
        result = {}
        for split_name, ds in self.splits.items():
            counts = Counter(ds["intent"])
            labeled = {
                self.id_to_intent.get(k, str(k)): v for k, v in sorted(counts.items())
            }
            result[split_name] = labeled
            total = sum(labeled.values())
            for intent, cnt in labeled.items():
                pct = cnt / total * 100
                print(f"  [{split_name}] {intent}: {cnt} ({pct:.1f}%)")

            # Imbalance detection
            counts_list = list(labeled.values())
            if len(counts_list) > 1:
                ratio = max(counts_list) / max(min(counts_list), 1)
                if ratio > 3:
                    print(
                        f"  ⚠️  [{split_name}] Class imbalance detected "
                        f"(max/min ratio = {ratio:.1f})"
                    )
        return result

    # ------------------------------------------------------------------
    def _token_length_distribution(self) -> dict:
        result = {}
        for split_name, ds in self.splits.items():
            lengths = [len(tags) for tags in ds["ner_tags"]]
            arr = np.array(lengths)
            stats = {
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
            }
            result[split_name] = stats
            print(
                f"  [{split_name}] min={stats['min']} max={stats['max']} "
                f"mean={stats['mean']:.1f} p95={stats['p95']:.0f}"
            )
            if stats["p95"] > 128:
                print(
                    f"  ⚠️  [{split_name}] p95 token length ({stats['p95']:.0f}) "
                    f"exceeds MAX_LENGTH=128 – some sequences will be truncated."
                )
        return result

    # ------------------------------------------------------------------
    def _entity_coverage(self) -> dict:
        result = {}
        for split_name, ds in self.splits.items():
            entity_counts: Counter = Counter()
            samples_with_entity = 0

            for tags in ds["ner_tags"]:
                entity_found = False
                for tag_id in tags:
                    if tag_id == -100 or tag_id == self.tag_map.get("O", 0):
                        continue
                    tag_name = self.id_to_tag.get(tag_id, str(tag_id))
                    if tag_name.startswith("B-"):
                        entity_counts[tag_name[2:]] += 1
                        entity_found = True
                if entity_found:
                    samples_with_entity += 1

            total = len(ds)
            coverage_pct = samples_with_entity / total * 100 if total else 0
            result[split_name] = {
                "entity_counts": dict(entity_counts),
                "samples_with_entity": samples_with_entity,
                "coverage_pct": coverage_pct,
            }
            print(f"  [{split_name}] Entity coverage: {coverage_pct:.1f}% of samples")
            for ent, cnt in sorted(entity_counts.items()):
                print(f"    {ent}: {cnt}")
        return result

    # ------------------------------------------------------------------
    def _data_quality(self) -> dict:
        """Detect duplicate texts, empty samples, and very short sequences."""
        result = {}
        for split_name, ds in self.splits.items():
            texts = ds["text"]
            total = len(texts)
            unique = len(set(texts))
            duplicates = total - unique
            dup_pct = duplicates / total * 100 if total else 0

            short = sum(1 for t in texts if len(t.split()) < 3)

            result[split_name] = {
                "total": total,
                "unique": unique,
                "duplicates": duplicates,
                "duplicate_pct": dup_pct,
                "short_samples": short,
            }
            print(
                f"  [{split_name}] total={total} unique={unique} "
                f"duplicates={duplicates} ({dup_pct:.1f}%) short={short}"
            )
            if dup_pct > 20:
                print(
                    f"  ⚠️  [{split_name}] High duplicate rate ({dup_pct:.1f}%) "
                    "– consider increasing template diversity."
                )
        return result

    # ------------------------------------------------------------------
    def plot(self, output_dir: str):
        """Generate and save all dataset-analysis plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("  ⚠️  matplotlib not available – skipping plots.")
            return
        if not self.splits:
            return

        self._plot_class_distribution(output_dir)
        self._plot_token_lengths(output_dir)
        self._plot_entity_distribution(output_dir)

    # ------------------------------------------------------------------
    def _plot_class_distribution(self, output_dir: str):
        n_splits = len(self.splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))
        if n_splits == 1:
            axes = [axes]

        for ax, (split_name, ds) in zip(axes, self.splits.items()):
            counts = Counter(ds["intent"])
            labels = [self.id_to_intent.get(k, str(k)) for k in sorted(counts)]
            values = [counts[k] for k in sorted(counts)]

            bars = ax.bar(range(len(labels)), values, color="steelblue", edgecolor="white")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"Intent distribution – {split_name}")
            ax.set_ylabel("Count")
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        fig.tight_layout()
        _save_figure(fig, output_dir, "dataset_class_distribution.png")

    # ------------------------------------------------------------------
    def _plot_token_lengths(self, output_dir: str):
        n_splits = len(self.splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 4))
        if n_splits == 1:
            axes = [axes]

        for ax, (split_name, ds) in zip(axes, self.splits.items()):
            lengths = [len(tags) for tags in ds["ner_tags"]]
            ax.hist(lengths, bins=30, color="coral", edgecolor="white")
            ax.axvline(128, color="red", linestyle="--", label="MAX_LENGTH=128")
            ax.set_title(f"Token length distribution – {split_name}")
            ax.set_xlabel("Token count")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)

        fig.tight_layout()
        _save_figure(fig, output_dir, "dataset_token_lengths.png")

    # ------------------------------------------------------------------
    def _plot_entity_distribution(self, output_dir: str):
        n_splits = len(self.splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(7 * n_splits, 5))
        if n_splits == 1:
            axes = [axes]

        for ax, (split_name, ds) in zip(axes, self.splits.items()):
            entity_counts: Counter = Counter()
            for tags in ds["ner_tags"]:
                for tag_id in tags:
                    if tag_id not in (-100, self.tag_map.get("O", 0)):
                        tag_name = self.id_to_tag.get(tag_id, str(tag_id))
                        entity_counts[tag_name] += 1

            if not entity_counts:
                ax.text(0.5, 0.5, "No entities found", ha="center", va="center")
                continue

            labels = list(entity_counts.keys())
            values = list(entity_counts.values())
            ax.barh(labels, values, color="mediumseagreen", edgecolor="white")
            ax.set_title(f"NER tag distribution – {split_name}")
            ax.set_xlabel("Count")

        fig.tight_layout()
        _save_figure(fig, output_dir, "dataset_entity_distribution.png")


# ============================================================================
# Section 2 – Model Loader & Predictor
# ============================================================================

class ModelEvaluator:
    """Load the fine-tuned model and run predictions / evaluations."""

    def __init__(
        self,
        model_dir: str,
        data_dir: str,
        tag_map: dict,
        intent_map: dict,
        max_length: int = 128,
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tag_map = tag_map
        self.intent_map = intent_map
        self.id_to_tag = {v: k for k, v in tag_map.items()}
        self.id_to_intent = {v: k for k, v in intent_map.items()}
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = None

    # ------------------------------------------------------------------
    def load_model(self) -> bool:
        """Load tokenizer and model weights.  Returns True on success."""
        if not TORCH_AVAILABLE:
            print("  ❌ PyTorch not installed.")
            return False
        if not TRANSFORMERS_AVAILABLE:
            print("  ❌ transformers not installed.")
            return False

        # We need train.py to be importable to get XLMRobertaForIntentAndNER
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from train import XLMRobertaForIntentAndNER  # noqa: PLC0415
        except ImportError as exc:
            print(f"  ❌ Could not import XLMRobertaForIntentAndNER from train.py: {exc}")
            return False

        model_path = Path(self.model_dir)
        if not model_path.exists():
            print(f"  ❌ Model directory not found: {model_path}")
            return False

        print(f"  Loading tokenizer from {model_path}…")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        print(f"  Initialising model architecture…")
        self.model = XLMRobertaForIntentAndNER(
            model_name=str(model_path),
            num_labels_intent=len(self.intent_map),
            num_labels_ner=len(self.tag_map),
        )

        # Load saved weights if present
        weights_path = model_path / "pytorch_model.bin"
        safetensors_path = model_path / "model.safetensors"
        if weights_path.exists():
            state = torch.load(str(weights_path), map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print("  ✅ Loaded pytorch_model.bin")
        elif safetensors_path.exists():
            try:
                from safetensors.torch import load_file  # noqa: PLC0415
                state = load_file(str(safetensors_path))
                self.model.load_state_dict(state, strict=False)
                print("  ✅ Loaded model.safetensors")
            except ImportError:
                print("  ⚠️  safetensors not installed; encoder weights loaded from config only.")
        else:
            print("  ⚠️  No pytorch_model.bin or model.safetensors found – using random weights.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✅ Model ready on {self.device}")
        return True

    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """Run inference on a single text string."""
        assert self.model is not None and self.tokenizer is not None

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=False,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        intent_id = int(outputs["intent_logits"].argmax(dim=1).item())
        ner_ids = outputs["ner_logits"].argmax(dim=2)[0].cpu().numpy()
        intent_scores = torch.softmax(outputs["intent_logits"], dim=1)[0].cpu().numpy()

        # Decode tokens & NER tags (skip special tokens)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        entities = []
        for token, tag_id in zip(tokens, ner_ids):
            if token in ("<s>", "</s>", "<pad>"):
                continue
            tag = self.id_to_tag.get(int(tag_id), "O")
            entities.append((token, tag))

        # Reconstruct entity spans
        entity_spans = self._reconstruct_entity_spans(entities)

        return {
            "text": text,
            "intent": self.id_to_intent.get(intent_id, str(intent_id)),
            "intent_id": intent_id,
            "intent_confidence": float(intent_scores.max()),
            "intent_scores": {
                self.id_to_intent.get(i, str(i)): float(s)
                for i, s in enumerate(intent_scores)
            },
            "token_tags": entities,
            "entity_spans": entity_spans,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _reconstruct_entity_spans(token_tags: list) -> list:
        """Merge B-/I- tokens into entity spans."""
        spans = []
        current = None
        for token, tag in token_tags:
            if tag.startswith("B-"):
                if current:
                    spans.append(current)
                current = {"type": tag[2:], "tokens": [token]}
            elif tag.startswith("I-") and current and current["type"] == tag[2:]:
                current["tokens"].append(token)
            else:
                if current:
                    spans.append(current)
                    current = None
        if current:
            spans.append(current)
        return spans

    # ------------------------------------------------------------------
    def evaluate_test_dataset(self) -> dict:
        """Run predictions on the held-out test split and compute metrics."""
        if not DATASETS_AVAILABLE:
            print("  ❌ 'datasets' package not available.")
            return {}

        test_path = Path(self.data_dir) / "test"
        if not test_path.exists():
            print(f"  ❌ Test split not found at {test_path}")
            return {}

        print("  Loading test split…")
        test_ds = load_from_disk(str(test_path))
        texts = test_ds["text"]
        true_intents = list(test_ds["intent"])
        true_ner_tags = list(test_ds["ner_tags"])

        pred_intents = []
        pred_ner_tags = []

        print(f"  Running inference on {len(texts)} test samples…")
        for i, text in enumerate(texts):
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(texts)}")
            result = self.predict(text)
            pred_intents.append(result["intent_id"])

            # Align predicted NER tags to the stored tag sequence length
            token_tags = result["token_tags"]
            tag_ids = [self.tag_map.get(tag, 0) for _, tag in token_tags]
            # Pad / truncate to match the true tag length (after filtering -100)
            true_len = len([t for t in true_ner_tags[i] if t != -100])
            tag_ids = tag_ids[:true_len]
            if len(tag_ids) < true_len:
                tag_ids += [0] * (true_len - len(tag_ids))
            pred_ner_tags.append(tag_ids)

        return self._compute_detailed_metrics(
            true_intents, pred_intents, true_ner_tags, pred_ner_tags
        )

    # ------------------------------------------------------------------
    def _compute_detailed_metrics(
        self,
        true_intents,
        pred_intents,
        true_ner_tags_list,
        pred_ner_tags_list,
    ) -> dict:
        """Compute per-class intent + NER metrics and collect error samples."""
        results = {}

        # ---- Intent metrics ----
        if SKLEARN_AVAILABLE:
            intent_labels_present = sorted(set(true_intents) | set(pred_intents))
            intent_names = [self.id_to_intent.get(i, str(i)) for i in intent_labels_present]

            intent_accuracy = accuracy_score(true_intents, pred_intents)
            intent_report = classification_report(
                true_intents,
                pred_intents,
                labels=intent_labels_present,
                target_names=intent_names,
                output_dict=True,
                zero_division=0,
            )
            intent_cm = confusion_matrix(
                true_intents, pred_intents, labels=intent_labels_present
            ).tolist()

            results["intent_accuracy"] = intent_accuracy
            results["intent_report"] = intent_report
            results["intent_confusion_matrix"] = intent_cm
            results["intent_labels"] = intent_names

            print(f"\n  Intent accuracy: {intent_accuracy:.4f}")
            print(
                "\n  Per-intent metrics:\n"
                + classification_report(
                    true_intents,
                    pred_intents,
                    labels=intent_labels_present,
                    target_names=intent_names,
                    zero_division=0,
                )
            )

            # ---- NER metrics ----
            all_true_ner, all_pred_ner = [], []
            for true_tags, pred_tags in zip(true_ner_tags_list, pred_ner_tags_list):
                for t, p in zip(true_tags, pred_tags):
                    if t == -100:
                        continue
                    all_true_ner.append(t)
                    all_pred_ner.append(p)

            if all_true_ner:
                ner_labels_present = sorted(set(all_true_ner) | set(all_pred_ner))
                ner_names = [self.id_to_tag.get(i, str(i)) for i in ner_labels_present]
                ner_report = classification_report(
                    all_true_ner,
                    all_pred_ner,
                    labels=ner_labels_present,
                    target_names=ner_names,
                    output_dict=True,
                    zero_division=0,
                )
                ner_cm = confusion_matrix(
                    all_true_ner, all_pred_ner, labels=ner_labels_present
                ).tolist()

                results["ner_report"] = ner_report
                results["ner_confusion_matrix"] = ner_cm
                results["ner_labels"] = ner_names

                ner_weighted_f1 = f1_score(
                    all_true_ner, all_pred_ner, average="weighted", zero_division=0
                )
                results["ner_weighted_f1"] = ner_weighted_f1
                print(f"\n  NER weighted F1: {ner_weighted_f1:.4f}")
                print(
                    "\n  Per-tag NER metrics:\n"
                    + classification_report(
                        all_true_ner,
                        all_pred_ner,
                        labels=ner_labels_present,
                        target_names=ner_names,
                        zero_division=0,
                    )
                )

        return results

    # ------------------------------------------------------------------
    def run_manual_tests(self, test_cases: list) -> list:
        """Run inference on manual test cases and compare to expected output."""
        print(f"\n  Running {len(test_cases)} manual test cases…")
        results = []

        for tc in test_cases:
            pred = self.predict(tc["text"])
            intent_correct = pred["intent"] == tc["expected_intent"]

            # Check entity type coverage (not exact span, just types present)
            found_types = {span["type"] for span in pred["entity_spans"]}
            expected_types = {etype for _, etype in tc["expected_entities"]}
            entity_correct = expected_types.issubset(found_types)

            results.append(
                {
                    "text": tc["text"],
                    "expected_intent": tc["expected_intent"],
                    "predicted_intent": pred["intent"],
                    "intent_correct": intent_correct,
                    "intent_confidence": pred["intent_confidence"],
                    "expected_entity_types": sorted(expected_types),
                    "found_entity_types": sorted(found_types),
                    "entity_types_correct": entity_correct,
                    "entity_spans": pred["entity_spans"],
                }
            )

            status = "✅" if (intent_correct and entity_correct) else "❌"
            print(
                f"  {status} [{pred['intent']} | conf={pred['intent_confidence']:.2f}] "
                f"{tc['text'][:60]}"
            )
            if not intent_correct:
                print(f"     Intent: expected={tc['expected_intent']}")
            if not entity_correct:
                print(
                    f"     Entities: expected={sorted(expected_types)} "
                    f"found={sorted(found_types)}"
                )

        intent_acc = sum(r["intent_correct"] for r in results) / len(results)
        entity_acc = sum(r["entity_types_correct"] for r in results) / len(results)
        print(
            f"\n  Manual test summary: "
            f"intent accuracy={intent_acc:.0%} | entity accuracy={entity_acc:.0%}"
        )
        return results

    # ------------------------------------------------------------------
    def detect_overfitting(self, train_metrics: dict, val_metrics: dict) -> dict:
        """Compare train vs val metrics to flag overfitting / underfitting."""
        analysis = {}
        for metric in ("intent_accuracy", "ner_weighted_f1"):
            train_val = train_metrics.get(metric)
            val_val = val_metrics.get(metric)
            if train_val is None or val_val is None:
                continue
            gap = train_val - val_val
            if gap > 0.10:
                diagnosis = "overfitting"
            elif val_val < 0.70:
                diagnosis = "underfitting"
            else:
                diagnosis = "good_fit"
            analysis[metric] = {
                "train": train_val,
                "val": val_val,
                "gap": gap,
                "diagnosis": diagnosis,
            }
            print(
                f"  {metric}: train={train_val:.4f} val={val_val:.4f} "
                f"gap={gap:.4f} → {diagnosis}"
            )
        return analysis

    # ------------------------------------------------------------------
    def plot_metrics(self, metrics: dict, output_dir: str):
        """Generate evaluation plots: confusion matrices, per-class bars."""
        if not MATPLOTLIB_AVAILABLE:
            print("  ⚠️  matplotlib not available – skipping plots.")
            return

        if "intent_confusion_matrix" in metrics:
            self._plot_confusion_matrix(
                np.array(metrics["intent_confusion_matrix"]),
                metrics.get("intent_labels", []),
                "Intent Confusion Matrix",
                output_dir,
                "eval_intent_confusion_matrix.png",
            )

        if "ner_confusion_matrix" in metrics:
            self._plot_confusion_matrix(
                np.array(metrics["ner_confusion_matrix"]),
                metrics.get("ner_labels", []),
                "NER Tag Confusion Matrix",
                output_dir,
                "eval_ner_confusion_matrix.png",
            )

        if "intent_report" in metrics:
            self._plot_per_class_metrics(
                metrics["intent_report"],
                "Per-intent F1 / Precision / Recall",
                output_dir,
                "eval_intent_per_class.png",
            )

        if "ner_report" in metrics:
            self._plot_per_class_metrics(
                metrics["ner_report"],
                "Per-NER-tag F1 / Precision / Recall",
                output_dir,
                "eval_ner_per_class.png",
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _plot_confusion_matrix(
        cm: np.ndarray, labels: list, title: str, output_dir: str, filename: str
    ):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        fig.tight_layout()
        _save_figure(fig, output_dir, filename)

    # ------------------------------------------------------------------
    @staticmethod
    def _plot_per_class_metrics(report: dict, title: str, output_dir: str, filename: str):
        classes = [
            k for k in report
            if k not in ("accuracy", "macro avg", "weighted avg")
        ]
        if not classes:
            return

        precision = [report[c].get("precision", 0) for c in classes]
        recall = [report[c].get("recall", 0) for c in classes]
        f1 = [report[c].get("f1-score", 0) for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
        ax.bar(x - width, precision, width, label="Precision", color="steelblue")
        ax.bar(x, recall, width, label="Recall", color="coral")
        ax.bar(x + width, f1, width, label="F1", color="mediumseagreen")

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        fig.tight_layout()
        _save_figure(fig, output_dir, filename)


# ============================================================================
# Section 3 – Diagnostics & Recommendations Engine
# ============================================================================

class RecommendationsEngine:
    """Analyse evaluation results and emit actionable recommendations."""

    def __init__(self, tag_map: dict, intent_map: dict):
        self.tag_map = tag_map
        self.intent_map = intent_map
        self.id_to_intent = {v: k for k, v in intent_map.items()}
        self.id_to_tag = {v: k for k, v in tag_map.items()}

    # ------------------------------------------------------------------
    def analyse(
        self,
        dataset_summary: dict,
        eval_metrics: dict,
        manual_results: list,
    ) -> dict:
        """Generate a structured recommendations report."""
        recs: dict = defaultdict(list)

        self._check_dataset_balance(dataset_summary, recs)
        self._check_token_length(dataset_summary, recs)
        self._check_data_quality(dataset_summary, recs)
        self._check_model_performance(eval_metrics, recs)
        self._check_manual_tests(manual_results, recs)
        self._check_entity_coverage(dataset_summary, recs)

        return dict(recs)

    # ------------------------------------------------------------------
    def _check_dataset_balance(self, summary: dict, recs: dict):
        dist = summary.get("class_distribution", {}).get("train", {})
        if not dist:
            return
        counts = list(dist.values())
        if len(counts) < 2:
            return
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 3:
            minority = [k for k, v in dist.items() if v == min(counts)]
            recs["data_quantity"].append(
                f"Class imbalance detected (max/min ratio={ratio:.1f}). "
                f"Consider oversampling or generating more data for: {minority}."
            )
        total = sum(counts)
        if total < 5000:
            recs["data_quantity"].append(
                f"Training set is small ({total} samples). "
                "Consider increasing REPEAT_FACTOR in data_genrator.py."
            )

    # ------------------------------------------------------------------
    def _check_token_length(self, summary: dict, recs: dict):
        for split, stats in summary.get("token_lengths", {}).items():
            if stats.get("p95", 0) > 128:
                recs["data_quality"].append(
                    f"[{split}] p95 token length exceeds MAX_LENGTH=128. "
                    "Increase MAX_LENGTH in train.py or shorten templates."
                )

    # ------------------------------------------------------------------
    def _check_data_quality(self, summary: dict, recs: dict):
        for split, info in summary.get("data_quality", {}).items():
            if info.get("duplicate_pct", 0) > 30:
                recs["data_quality"].append(
                    f"[{split}] {info['duplicate_pct']:.1f}% duplicate texts detected. "
                    "Increase template variety or add more synonym groups."
                )

    # ------------------------------------------------------------------
    def _check_model_performance(self, metrics: dict, recs: dict):
        if not metrics:
            return

        intent_acc = metrics.get("intent_accuracy", 1.0)
        ner_f1 = metrics.get("ner_weighted_f1", 1.0)

        if intent_acc < 0.85:
            recs["model_performance"].append(
                f"Intent accuracy is low ({intent_acc:.2%}). "
                "Consider more training data, longer training, or lower learning rate."
            )

        if ner_f1 < 0.80:
            recs["model_performance"].append(
                f"NER weighted F1 is low ({ner_f1:.2%}). "
                "Try increasing ner_weight in train.py or adding entity-rich templates."
            )

        # Per-intent weak spots
        intent_report = metrics.get("intent_report", {})
        for intent, scores in intent_report.items():
            if intent in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(scores, dict) and scores.get("f1-score", 1.0) < 0.75:
                recs["weak_spots"].append(
                    f"Intent '{intent}' has low F1={scores['f1-score']:.2f}. "
                    "Add more diverse training examples for this intent."
                )

        # Per-tag weak spots
        ner_report = metrics.get("ner_report", {})
        for tag, scores in ner_report.items():
            if tag in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(scores, dict) and scores.get("f1-score", 1.0) < 0.75:
                recs["weak_spots"].append(
                    f"NER tag '{tag}' has low F1={scores['f1-score']:.2f}. "
                    "Ensure this tag appears frequently in training data."
                )

    # ------------------------------------------------------------------
    def _check_manual_tests(self, manual_results: list, recs: dict):
        if not manual_results:
            return
        failed_intents: Counter = Counter()
        failed_entities: Counter = Counter()
        for r in manual_results:
            if not r["intent_correct"]:
                failed_intents[r["expected_intent"]] += 1
            if not r["entity_types_correct"]:
                missing = set(r["expected_entity_types"]) - set(r["found_entity_types"])
                for m in missing:
                    failed_entities[m] += 1

        for intent, cnt in failed_intents.most_common():
            recs["manual_test_failures"].append(
                f"Intent '{intent}' failed {cnt} manual test(s). "
                "Review test cases and add similar patterns to training data."
            )
        for entity, cnt in failed_entities.most_common():
            recs["manual_test_failures"].append(
                f"Entity type '{entity}' was missed in {cnt} manual test(s). "
                "Ensure entity appears in templates and synonym lists."
            )

    # ------------------------------------------------------------------
    def _check_entity_coverage(self, summary: dict, recs: dict):
        for split, info in summary.get("entity_coverage", {}).items():
            coverage = info.get("coverage_pct", 100)
            if coverage < 80:
                recs["data_quality"].append(
                    f"[{split}] Only {coverage:.1f}% of samples contain entities. "
                    "Check entity extraction in data_genrator.py."
                )

    # ------------------------------------------------------------------
    def print_report(self, recs: dict):
        """Pretty-print the recommendations to stdout."""
        print("\n" + "=" * 70)
        print("📋  RECOMMENDATIONS REPORT")
        print("=" * 70)

        category_icons = {
            "data_quantity": "📊",
            "data_quality": "🔍",
            "model_performance": "🤖",
            "weak_spots": "⚠️ ",
            "manual_test_failures": "🧪",
        }

        if not recs:
            print("  ✅  No issues detected – model and dataset look healthy!")
            return

        for category, items in recs.items():
            icon = category_icons.get(category, "•")
            print(f"\n{icon}  {category.replace('_', ' ').upper()}")
            for item in items:
                print(f"   • {item}")

    # ------------------------------------------------------------------
    def save_report(self, recs: dict, output_dir: str):
        """Save recommendations as JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "recommendations.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved recommendations: {path}")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation script for the XLM-RoBERTa Intent+NER model."
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to the trained model directory (optional). "
             "Omit to run dataset analysis only.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Path to the directory containing train/val/test splits and mapping JSONs.",
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_output",
        help="Directory where plots and reports will be saved.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length (should match training config).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation (useful in headless environments).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🔬  XLM-RoBERTa Model Evaluation & Dataset Analysis")
    print("=" * 70 + "\n")

    # ---- Load mappings ----
    print("📂  Loading tag & intent mappings…")
    tag_map, intent_map = _load_mappings(args.data_dir)
    print(f"  NER tags  : {list(tag_map.keys())}")
    print(f"  Intents   : {list(intent_map.keys())}")

    all_results: dict = {}

    # ================================================================
    # Step 1 – Dataset Analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("📊  STEP 1 – Dataset Analysis")
    print("=" * 70)

    analyzer = DatasetAnalyzer(args.data_dir, tag_map, intent_map)
    if analyzer.load():
        dataset_summary = analyzer.analyze()
        all_results["dataset_summary"] = dataset_summary
        if not args.no_plots:
            print("\n  Generating dataset plots…")
            analyzer.plot(args.output_dir)
    else:
        dataset_summary = {}

    # ================================================================
    # Step 2 – Model Evaluation
    # ================================================================
    eval_metrics: dict = {}
    manual_results: list = []

    if args.model_dir:
        print("\n" + "=" * 70)
        print("🤖  STEP 2 – Model Evaluation")
        print("=" * 70)

        evaluator = ModelEvaluator(
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            tag_map=tag_map,
            intent_map=intent_map,
            max_length=args.max_length,
        )

        if evaluator.load_model():
            # 2a – Test dataset evaluation
            print("\n--- Test Dataset Evaluation ---")
            eval_metrics = evaluator.evaluate_test_dataset()
            all_results["eval_metrics"] = eval_metrics

            # 2b – Manual test cases (including edge cases)
            print("\n--- Manual & Edge Case Tests ---")
            manual_results = evaluator.run_manual_tests(MANUAL_TEST_CASES)
            all_results["manual_test_results"] = manual_results

            # 2c – Overfitting diagnostics (requires trainer_state.json)
            trainer_state_path = Path(args.model_dir) / "trainer_state.json"
            if trainer_state_path.exists():
                print("\n--- Overfitting / Underfitting Diagnostics ---")
                with open(trainer_state_path) as f:
                    trainer_state = json.load(f)
                logs = trainer_state.get("log_history", [])
                # Extract last train / eval epoch metrics
                train_log = next(
                    (l for l in reversed(logs) if "loss" in l and "eval_loss" not in l),
                    {},
                )
                eval_log = next(
                    (l for l in reversed(logs) if "eval_loss" in l),
                    {},
                )
                train_metrics_snap = {
                    "intent_accuracy": train_log.get("intent_accuracy"),
                    "ner_weighted_f1": train_log.get("ner_f1"),
                }
                val_metrics_snap = {
                    "intent_accuracy": eval_log.get("eval_intent_accuracy"),
                    "ner_weighted_f1": eval_log.get("eval_ner_f1"),
                }
                overfit_analysis = evaluator.detect_overfitting(
                    train_metrics_snap, val_metrics_snap
                )
                all_results["overfitting_analysis"] = overfit_analysis

            # 2d – Plots
            if not args.no_plots and eval_metrics:
                print("\n  Generating evaluation plots…")
                evaluator.plot_metrics(eval_metrics, args.output_dir)
        else:
            print("  ⚠️  Model could not be loaded – skipping model evaluation.")
    else:
        print(
            "\n  ℹ️  No --model_dir provided. "
            "Running dataset analysis only (pass --model_dir to enable model evaluation)."
        )

    # ================================================================
    # Step 3 – Recommendations
    # ================================================================
    print("\n" + "=" * 70)
    print("💡  STEP 3 – Recommendations Engine")
    print("=" * 70)

    rec_engine = RecommendationsEngine(tag_map, intent_map)
    recommendations = rec_engine.analyse(dataset_summary, eval_metrics, manual_results)
    rec_engine.print_report(recommendations)
    rec_engine.save_report(recommendations, args.output_dir)
    all_results["recommendations"] = recommendations

    # ================================================================
    # Save full results JSON
    # ================================================================
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    # Convert numpy types for JSON serialisation
    def _to_serialisable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_to_serialisable)
    print(f"\n💾  Full results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("✅  Evaluation complete!")
    print(f"   Output directory: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
