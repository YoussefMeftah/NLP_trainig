"""
Quick Model Diagnostic Tool
Run this for a fast summary of model performance and data quality.

Usage:
    python quick_check.py
"""

import torch
import json
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import numpy as np

DATA_DIR = "./data"
MODEL_DIR = "./model/checkpoint-1395"

# Load config
with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)
with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}
ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}


def health_check():
    """Run quick health check on model and data."""
    print("\n" + "="*70)
    print("⚕️  XLM-RoBERTa Model Health Check")
    print("="*70 + "\n")

    checks = {
        "gpu": False,
        "model_exists": False,
        "dataset_exists": False,
        "config_valid": False,
        "balance_ok": False,
    }

    # Check GPU
    print("🔍 Checking resources...")
    if torch.cuda.is_available():
        print(f"   ✅ GPU Available: {torch.cuda.get_device_name(0)}")
        checks["gpu"] = True
    else:
        print(f"   ⚠️  GPU Not Available (will use CPU - slower training)")

    # Check model
    model_path = Path(MODEL_DIR)
    if model_path.exists():
        print(f"   ✅ Model found: {MODEL_DIR}")
        checks["model_exists"] = True
    else:
        print(f"   ❌ Model not found: {MODEL_DIR}")

    # Check dataset
    try:
        train = load_from_disk(f"{DATA_DIR}/train")
        val = load_from_disk(f"{DATA_DIR}/val")
        test = load_from_disk(f"{DATA_DIR}/test")
        print(f"   ✅ Dataset found: {len(train)} train, {len(val)} val, {len(test)} test")
        checks["dataset_exists"] = True
    except Exception as e:
        print(f"   ❌ Dataset error: {e}")

    # Check config
    if len(INTENT_MAP) > 0 and len(TAG_MAP) > 0:
        print(f"   ✅ Config valid: {len(INTENT_MAP)} intents, {len(TAG_MAP)} NER tags")
        checks["config_valid"] = True
    else:
        print(f"   ❌ Config invalid")

    # Check balance
    if checks["dataset_exists"]:
        print(f"\n📊 Checking data balance...")
        intent_counts = Counter(train["intent"])
        min_count = min(intent_counts.values())
        max_count = max(intent_counts.values())
        imbalance = max_count / (min_count + 1)

        if imbalance < 2:
            print(f"   ✅ Intents well-balanced ({imbalance:.1f}x ratio)")
            checks["balance_ok"] = True
        elif imbalance < 5:
            print(f"   ⚠️  Intents somewhat imbalanced ({imbalance:.1f}x ratio)")
        else:
            print(f"   ❌ Intents very imbalanced ({imbalance:.1f}x ratio)")

        # NER balance
        all_tags = []
        for sample in train:
            all_tags.extend([t for t in sample["ner_tags"] if t != -100 and t != TAG_MAP.get("O", 0)])

        tag_counts = Counter(all_tags)
        if tag_counts:
            min_tag = min(tag_counts.values())
            max_tag = max(tag_counts.values())
            tag_imbalance = max_tag / (min_tag + 1)
            if tag_imbalance < 3:
                print(f"   ✅ NER tags well-balanced ({tag_imbalance:.1f}x ratio)")
            elif tag_imbalance < 10:
                print(f"   ⚠️  NER tags somewhat imbalanced ({tag_imbalance:.1f}x ratio)")
            else:
                print(f"   ❌ NER tags very imbalanced ({tag_imbalance:.1f}x ratio)")

    # Overall health
    print(f"\n📈 Overall Status:")
    passed = sum(checks.values())
    total = len(checks)
    health_pct = (passed / total) * 100

    if health_pct == 100:
        print(f"   ✅ All systems GO! ({passed}/{total}) - Ready to test")
    elif health_pct >= 80:
        print(f"   ⚠️  Good condition ({passed}/{total}) - Minor issues")
    else:
        print(f"   ❌ Needs attention ({passed}/{total}) - Fix issues before testing")

    return checks


def quick_stats():
    """Print quick dataset statistics."""
    print(f"\n" + "="*70)
    print("📊 Quick Dataset Statistics")
    print("="*70 + "\n")

    try:
        train = load_from_disk(f"{DATA_DIR}/train")

        print("Intent Distribution:")
        intent_counts = Counter(train["intent"])
        for intent_id in sorted(intent_counts.keys()):
            count = intent_counts[intent_id]
            pct = 100 * count / len(train)
            bar_len = int(pct / 5)  # 20% = 1 char
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {ID_TO_INTENT[intent_id]:25s} {bar} {count:4d} ({pct:5.1f}%)")

        print(f"\nNER Tag Frequency:")
        all_tags = []
        for sample in train:
            all_tags.extend(sample["ner_tags"])

        tag_counts = Counter(all_tags)
        total = len(all_tags)

        for tag_id in sorted(tag_counts.keys()):
            if tag_id == -100:
                continue  # Skip padding
            count = tag_counts[tag_id]
            pct = 100 * count / total
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {ID_TO_TAG[tag_id]:20s} {bar} {count:5d} ({pct:5.1f}%)")

    except Exception as e:
        print(f"   ❌ Error: {e}")


def recommendations():
    """Quick recommendations."""
    print(f"\n" + "="*70)
    print("💡 Next Steps")
    print("="*70 + "\n")

    print("1️⃣  Test your model:")
    print("   python test_model.py --model_dir ./model/checkpoint-1395\n")

    print("2️⃣  Analyze data quality:")
    print("   python analyze_data.py\n")

    print("3️⃣  Interactive testing:")
    print("   python test_model.py --model_dir ./model/checkpoint-1395 --interactive\n")

    print("4️⃣  See full guide:")
    print("   Read TESTING_GUIDE.md for detailed metrics interpretation\n")

    print("📖 Common Issues & Fixes:")
    print("   - Low intent accuracy (< 70%)? → Increase REPEAT_FACTOR in data_generator.py")
    print("   - Low NER F1 (< 65%)? → Add more diverse date/category formats")
    print("   - Class imbalance? → Run analyze_data.py to identify underrepresented classes")
    print("   - Slow training? → Reduce REPEAT_FACTOR or use GPU (check above)\n")


def main():
    # Run checks
    health_check()
    quick_stats()
    recommendations()

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
