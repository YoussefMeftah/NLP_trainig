"""
NER Tag Diagnostic Tool
Diagnose why NER tags aren't being extracted or evaluated properly.

Usage:
    python diagnose_ner.py
"""

import json
from datasets import load_from_disk
from collections import Counter

DATA_DIR = "./data"

with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)
with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}


def diagnose_dataset():
    """Comprehensive NER tag diagnosis."""
    print("\n" + "="*70)
    print("🔍 NER TAG DIAGNOSTIC")
    print("="*70 + "\n")

    try:
        test_dataset = load_from_disk(f"{DATA_DIR}/test")
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return

    print(f"Test Dataset Size: {len(test_dataset)} samples\n")

    if len(test_dataset) == 0:
        print("❌ Test dataset is empty!")
        return

    # Check structure of first sample
    print("📋 Sample Structure Check:")
    first_sample = test_dataset[0]
    print(f"   Fields in sample: {list(first_sample.keys())}")

    if "ner_tags" not in first_sample:
        print("   ❌ ERROR: 'ner_tags' field not found in samples!")
        print(f"   Available fields: {list(first_sample.keys())}")
        return

    print(f"   ✅ 'ner_tags' field found\n")

    # Check sample details
    print("🔎 Sample Details:")
    print(f"   Text: {first_sample.get('text', 'NOT FOUND')[:100]}...")
    print(f"   Intent: {first_sample.get('intent', 'NOT FOUND')}")
    print(f"   NER Tags length: {len(first_sample['ner_tags'])}")
    print(f"   NER Tags (first 20): {first_sample['ner_tags'][:20]}")

    # Count special vs real tags
    all_tags = []
    samples_with_tags = 0
    samples_without_tags = 0
    sample_lengths = []

    for sample in test_dataset:
        ner_tags = sample["ner_tags"]
        all_tags.extend(ner_tags)
        sample_lengths.append(len(ner_tags))

        # Check if sample has any non-padding tags
        non_padding = [t for t in ner_tags if t != -100]
        if len(non_padding) > 0:
            samples_with_tags += 1
        else:
            samples_without_tags += 1

    print(f"\n📊 NER Tag Statistics:")
    print(f"   Samples with NER tags: {samples_with_tags}/{len(test_dataset)}")
    print(f"   Samples without NER tags (all -100): {samples_without_tags}/{len(test_dataset)}")
    print(f"   Avg sample length: {sum(sample_lengths)/len(sample_lengths):.1f}")

    # Tag distribution
    tag_counts = Counter(all_tags)
    print(f"\n🏷️  Tag Distribution:")
    total_tags = len(all_tags)

    for tag_id in sorted(tag_counts.keys()):
        count = tag_counts[tag_id]
        pct = 100 * count / total_tags
        tag_name = ID_TO_TAG.get(tag_id, f"UNKNOWN_{tag_id}")
        print(f"   {tag_name:20s}: {count:6d} ({pct:5.1f}%)")

    # Check for issues
    print(f"\n⚠️  Diagnostics:")

    if samples_without_tags == len(test_dataset):
        print("   ❌ ALL samples are empty (all tags are -100)")
        print("   This is why NER evaluation is failing!")
        print("\n   CAUSES:")
        print("   1. Data generation didn't create NER tags properly")
        print("   2. NER tags were lost during data preprocessing")
        print("   3. align_tags_to_tokens() is filtering everything out")
        print("\n   FIX:")
        print("   1. Check prepare_dataset() in train.py")
        print("   2. Verify data_generator.py is creating ner_tags correctly")
        print("   3. Run: python analyze_data.py to get detailed report")

    elif samples_without_tags > len(test_dataset) * 0.5:
        print(f"   ⚠️  Most samples are empty ({samples_without_tags} out of {len(test_dataset)})")
        print("   NER quality will be poor - check data generation")

    else:
        print(f"   ✅ {samples_with_tags} samples have NER tags - data looks good")

    # Check if we have all entity types
    non_padding_tags = [t for t in tag_counts.keys() if t != -100 and t != TAG_MAP.get("O", 0)]
    print(f"\n🔖 Entity Types Found: {len(non_padding_tags)}")
    for tag_id in non_padding_tags:
        tag_name = ID_TO_TAG.get(tag_id, f"UNKNOWN_{tag_id}")
        count = tag_counts[tag_id]
        print(f"   {tag_name}: {count} occurrences")

    if len(non_padding_tags) == 0:
        print("   ❌ No entity tags found! Only 'O' (outside) or padding.")
        print("   NER learned nothing useful - check data generation")


def check_data_processing():
    """Check if data preprocessing is working."""
    print(f"\n" + "="*70)
    print("🔧 Data Processing Check")
    print("="*70 + "\n")

    # Load train dataset to see raw form
    try:
        train_dataset = load_from_disk(f"{DATA_DIR}/train")

        if len(train_dataset) == 0:
            print("   ❌ Train dataset is empty!")
            return

        sample = train_dataset[0]
        print(f"Train Sample (raw):")
        print(f"   Text: {sample.get('text', 'NOT FOUND')[:100]}...")
        print(f"   Tokens: {sample.get('tokens', 'NOT FOUND')}")
        print(f"   NER Tags: {sample.get('ner_tags', 'NOT FOUND')[:20]}...")

        # Check if tokens are being generated
        if "tokens" in sample:
            print(f"\n   ✅ Tokens field exists")
            if len(sample["tokens"]) > 0:
                print(f"   ✅ Tokens are not empty ({len(sample['tokens'])} tokens)")
            else:
                print(f"   ❌ Tokens field is empty")
        else:
            print(f"   ❌ Tokens field missing")

    except Exception as e:
        print(f"   ❌ Error loading train dataset: {e}")


def suggestions():
    """Provide actionable suggestions."""
    print(f"\n" + "="*70)
    print("💡 Suggested Actions")
    print("="*70 + "\n")

    print("1. Verify data was generated correctly:")
    print("   python data_generator.py\n")

    print("2. Check if NER tags are corrupted during preprocessing:")
    print("   - Check prepare_dataset() in train.py")
    print("   - Ensure ner_tags are NOT being removed\n")

    print("3. Run full analysis to understand issue better:")
    print("   python analyze_data.py --output_report debug_report.json\n")

    print("4. If issue persists:")
    print("   - Delete ./data/train, ./data/val, ./data/test directories")
    print("   - Regenerate dataset: python data_generator.py")
    print("   - Retrain model: python train.py\n")


def main():
    diagnose_dataset()
    check_data_processing()
    suggestions()

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
