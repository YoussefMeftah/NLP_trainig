"""
Quick Dataset Structure Checker
Quickly see what's in your dataset without loading everything into memory.

Usage:
    python check_dataset.py
    python check_dataset.py --dataset test
"""

import argparse
import json
from datasets import load_from_disk

DATA_DIR = "./data"

with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}


def check_split(split_name):
    """Check structure of a dataset split."""
    print(f"\n{'='*70}")
    print(f"📊 {split_name.upper()} DATASET")
    print(f"{'='*70}\n")

    try:
        dataset = load_from_disk(f"{DATA_DIR}/{split_name}")
    except Exception as e:
        print(f"❌ Error loading {split_name} dataset: {e}")
        return False

    print(f"Size: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return False

    # Get some samples
    print(f"\n📋 Sample Structure:")

    for idx in range(min(3, len(dataset))):
        sample = dataset[idx]
        print(f"\n   Sample {idx}:")

        for key, value in sample.items():
            if isinstance(value, (list, tuple)):
                value_str = f"[{len(value)} items]"
                if len(value) > 0:
                    # Show first few items
                    first_items = str(value[:10])[:50] + "..."
                    value_str += f" {first_items}"
            elif isinstance(value, str):
                value_str = f'"{value[:50]}' + ('..."' if len(value) > 50 else '"')
            else:
                value_str = str(value)

            print(f"      {key:20s}: {value_str}")

    # Check specific fields
    print(f"\n🔍 Field Analysis:")

    sample = dataset[0]

    # Check for NER fields
    if "ner_labels" in sample:
        ner_field = "ner_labels"
        ner_value = sample["ner_labels"]
        print(f"\n   ✅ Has 'ner_labels' (preprocessed format)")
        print(f"      Length: {len(ner_value)}")
        print(f"      Unique values: {set(ner_value)}")
        non_padding = [v for v in ner_value if v != -100]
        print(f"      Non-padding tags: {len(non_padding)}/{len(ner_value)}")
        if len(non_padding) > 0:
            print(f"      ✅ Data contains actual NER labels")
        else:
            print(f"      ❌ All NER labels are padding (-100)")

    elif "ner_tags" in sample:
        ner_field = "ner_tags"
        ner_value = sample["ner_tags"]
        print(f"\n   ✅ Has 'ner_tags' (raw format)")
        print(f"      Length: {len(ner_value)}")
        print(f"      Unique values: {set(ner_value)}")
        non_padding = [v for v in ner_value if v != -100]
        print(f"      Non-padding tags: {len(non_padding)}/{len(ner_value)}")
        if len(non_padding) > 0:
            print(f"      ✅ Data contains actual NER tags")
        else:
            print(f"      ❌ All NER tags are padding (-100)")
    else:
        print(f"\n   ❌ No NER field found!")
        print(f"      Available: {list(sample.keys())}")

    # Check intent
    if "intent_labels" in sample:
        print(f"\n   ✅ Has 'intent_labels' (preprocessed format)")
        print(f"      Value: {sample['intent_labels']}")
    elif "intent" in sample:
        print(f"\n   ✅ Has 'intent' (raw format)")
        print(f"      Value: {sample['intent']}")
    else:
        print(f"\n   ❌ No intent field found!")

    # Check text
    if "text" in sample:
        text = sample["text"]
        print(f"\n   ✅ Has 'text' field")
        print(f"      Length: {len(text)} chars")
        print(f"      Preview: {text[:80]}...")
    else:
        print(f"\n   ❌ No 'text' field found (likely preprocessed)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Check dataset structure")
    parser.add_argument("--dataset", default=None, choices=["train", "val", "test"],
                       help="Check specific split (default: all)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("DATASET STRUCTURE CHECKER")
    print("="*70)

    if args.dataset:
        check_split(args.dataset)
    else:
        for split in ["train", "val", "test"]:
            check_split(split)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
