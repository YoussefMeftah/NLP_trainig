"""
Data Quality & Augmentation Analysis Script
Analyze your dataset to identify issues and suggest improvements.

Usage:
    python analyze_data.py --output_report analysis_report.json
"""

import json
import argparse
from collections import Counter, defaultdict
from datasets import load_from_disk
from datetime import datetime
import re

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = "./data"

with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)
with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}
ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}


# ============================================================================
# Data Analysis Functions
# ============================================================================
class DataAnalyzer:
    def __init__(self):
        self.report = {}

    def analyze_dataset(self, dataset, split_name):
        """Comprehensive dataset analysis."""
        analysis = {
            "split": split_name,
            "total_samples": len(dataset),
            "intent_distribution": {},
            "ner_tag_distribution": {},
            "text_length_stats": {},
            "entity_coverage": {},
            "potential_issues": [],
        }

        # Intent distribution
        intent_counts = Counter(dataset["intent"])
        total = len(dataset)
        for intent_id in sorted(intent_counts.keys()):
            count = intent_counts[intent_id]
            pct = 100 * count / total
            analysis["intent_distribution"][ID_TO_INTENT[intent_id]] = {
                "count": count,
                "percentage": round(pct, 2),
            }

        # NER tag distribution
        all_tags = []
        all_texts = []
        text_lengths = []

        for sample in dataset:
            all_tags.extend(sample["ner_tags"])
            if "text" in sample:
                all_texts.append(sample["text"])
                text_lengths.append(len(sample["text"]))

        tag_counts = Counter(all_tags)
        total_tokens = len(all_tags) - tag_counts[-100]  # Exclude padding

        for tag_id in sorted(tag_counts.keys()):
            if tag_id == -100:
                continue  # Skip padding
            count = tag_counts[tag_id]
            pct = 100 * count / total_tokens if total_tokens > 0 else 0
            analysis["ner_tag_distribution"][ID_TO_TAG[tag_id]] = {
                "count": count,
                "percentage": round(pct, 2),
            }

        # Text length statistics
        if text_lengths:
            analysis["text_length_stats"] = {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "avg": round(sum(text_lengths) / len(text_lengths), 2),
            }

        # Entity coverage (what % of samples have each entity type)
        entity_presence = defaultdict(int)
        for sample in dataset:
            tags_in_sample = set(sample["ner_tags"])
            for tag_id in tags_in_sample:
                if tag_id != -100 and tag_id != TAG_MAP.get("O", 0):
                    entity_presence[ID_TO_TAG[tag_id]] += 1

        for entity_type, count in entity_presence.items():
            pct = 100 * count / len(dataset)
            analysis["entity_coverage"][entity_type] = round(pct, 2)

        return analysis

    def check_for_issues(self, analysis):
        """Identify data quality issues."""
        issues = analysis.get("potential_issues", [])

        # Check intent imbalance
        intent_dist = analysis["intent_distribution"]
        intent_counts = [v["count"] for v in intent_dist.values()]
        if intent_counts:
            min_count = min(intent_counts)
            max_count = max(intent_counts)
            imbalance_ratio = max_count / (min_count + 1)  # Avoid div by zero
            if imbalance_ratio > 3:
                issues.append({
                    "severity": "MEDIUM",
                    "type": "Intent Imbalance",
                    "description": f"Intents have {imbalance_ratio:.1f}x difference in sample counts",
                    "affected": [k for k, v in intent_dist.items() if v["count"] == min_count],
                    "fix": "Add more samples for underrepresented intents"
                })

        # Check NER tag imbalance
        ner_dist = analysis["ner_tag_distribution"]
        ner_counts = [v["count"] for v in ner_dist.values()]
        if ner_counts:
            min_count = min(ner_counts)
            max_count = max(ner_counts)
            imbalance_ratio = max_count / (min_count + 1)
            if imbalance_ratio > 5:
                issues.append({
                    "severity": "HIGH",
                    "type": "NER Tag Imbalance",
                    "description": f"NER tags have {imbalance_ratio:.1f}x difference in occurrence",
                    "affected": [k for k, v in ner_dist.items() if v["count"] == min_count],
                    "fix": "Balance tag distribution by adding more diverse examples"
                })

        # Check if any intent has < 100 samples
        for intent_name, stats in intent_dist.items():
            if stats["count"] < 100:
                issues.append({
                    "severity": "MEDIUM",
                    "type": "Insufficient Samples",
                    "description": f"Intent '{intent_name}' has only {stats['count']} samples (recommended: >= 100)",
                    "affected": [intent_name],
                    "fix": f"Add at least {100 - stats['count']} more samples for {intent_name}"
                })

        # Check entity coverage
        coverage = analysis["entity_coverage"]
        for entity, pct in coverage.items():
            if pct < 50:
                issues.append({
                    "severity": "LOW",
                    "type": "Low Entity Coverage",
                    "description": f"Entity '{entity}' appears in only {pct:.1f}% of samples",
                    "affected": [entity],
                    "fix": "Add more samples with this entity type"
                })

        analysis["potential_issues"] = issues
        return analysis

    def generate_report(self):
        """Generate comprehensive analysis report."""
        try:
            train_dataset = load_from_disk(f"{DATA_DIR}/train")
            val_dataset = load_from_disk(f"{DATA_DIR}/val")
            test_dataset = load_from_disk(f"{DATA_DIR}/test")
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return None

        print("\n" + "="*70)
        print("DATASET QUALITY ANALYSIS")
        print("="*70 + "\n")

        # Analyze each split
        for dataset, split_name in [(train_dataset, "train"), (val_dataset, "val"), (test_dataset, "test")]:
            print(f"\n📊 Analyzing {split_name} split...")
            analysis = self.analyze_dataset(dataset, split_name)
            analysis = self.check_for_issues(analysis)
            self.report[split_name] = analysis

            # Print summary
            print(f"   Samples: {analysis['total_samples']}")
            print(f"   Intents: {len(analysis['intent_distribution'])}")
            print(f"   NER Tags: {len(analysis['ner_tag_distribution'])}")

            if analysis["potential_issues"]:
                print(f"   Issues Found: {len(analysis['potential_issues'])}")

        # Cross-split validation
        print(f"\n🔍 Cross-Split Validation...")
        self._validate_splits()

        return self.report

    def _validate_splits(self):
        """Check that train/val/test have consistent intent/tag distributions."""
        validation_issues = []

        # Check intents are consistent across splits
        train_intents = set(self.report["train"]["intent_distribution"].keys())
        val_intents = set(self.report["val"]["intent_distribution"].keys())
        test_intents = set(self.report["test"]["intent_distribution"].keys())

        all_intents = train_intents | val_intents | test_intents

        if not (train_intents == val_intents == test_intents):
            missing_intents = {
                "train": all_intents - train_intents,
                "val": all_intents - val_intents,
                "test": all_intents - test_intents,
            }
            validation_issues.append({
                "severity": "HIGH",
                "type": "Intent Distribution Mismatch",
                "description": "Not all intents are present in all splits",
                "details": {k: list(v) for k, v in missing_intents.items() if v},
                "fix": "Ensure all intents are represented in train/val/test splits"
            })

        self.report["validation_issues"] = validation_issues

    def print_report(self):
        """Pretty print the analysis report."""
        for split in ["train", "val", "test"]:
            analysis = self.report[split]
            print(f"\n{'='*70}")
            print(f"{split.upper()} SPLIT ANALYSIS")
            print(f"{'='*70}")

            print(f"\nTotal Samples: {analysis['total_samples']}")

            print(f"\nIntent Distribution:")
            for intent_name, stats in sorted(analysis["intent_distribution"].items()):
                print(f"  {intent_name:30s}: {stats['count']:5d} ({stats['percentage']:5.1f}%)")

            print(f"\nNER Tag Distribution:")
            for tag_name, stats in sorted(analysis["ner_tag_distribution"].items()):
                print(f"  {tag_name:20s}: {stats['count']:6d} ({stats['percentage']:5.1f}%)")

            print(f"\nEntity Coverage (% of samples):")
            for entity, pct in sorted(analysis["entity_coverage"].items()):
                print(f"  {entity:20s}: {pct:5.1f}%")

            if analysis["potential_issues"]:
                print(f"\n⚠️  Issues Found ({len(analysis['potential_issues'])}):")
                for i, issue in enumerate(analysis["potential_issues"], 1):
                    print(f"\n  [{issue['severity']}] {i}. {issue['type']}")
                    print(f"      Description: {issue['description']}")
                    print(f"      Fix: {issue['fix']}")

        # Validation issues
        if self.report.get("validation_issues"):
            print(f"\n{'='*70}")
            print("CROSS-SPLIT VALIDATION")
            print(f"{'='*70}")
            for issue in self.report["validation_issues"]:
                print(f"\n⚠️  [{issue['severity']}] {issue['type']}")
                print(f"   {issue['description']}")
                print(f"   Fix: {issue['fix']}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze dataset quality")
    parser.add_argument("--output_report", default=None, help="Save JSON report")
    args = parser.parse_args()

    analyzer = DataAnalyzer()
    report = analyzer.generate_report()

    # Print detailed report
    analyzer.print_report()

    # Save JSON report
    if args.output_report:
        print(f"\n💾 Saving report to {args.output_report}...")
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved!")

    # Summary recommendations
    print(f"\n{'='*70}")
    print("ACTIONABLE RECOMMENDATIONS")
    print(f"{'='*70}")

    total_samples = report["train"]["total_samples"]
    print(f"\nCurrent Dataset Size: {total_samples} samples")

    if total_samples < 500:
        print(f"⚠️  Dataset is small (< 500 samples). Consider increasing to 1000-2000.")
    elif total_samples < 1000:
        print(f"⚠️  Dataset is moderate. For production, aim for 2000-5000 samples.")
    elif total_samples < 5000:
        print(f"✅ Dataset size is good. Could still benefit from more diverse examples.")
    else:
        print(f"✅ Dataset size is solid. Focus on quality over quantity.")

    print(f"\nDataset Quality Checklist:")
    print(f"  [ ] All intents have >= 100 samples")
    print(f"  [ ] NER tags are balanced (within 3x ratio)")
    print(f"  [ ] Entity coverage >= 90% for critical entities")
    print(f"  [ ] Dates span across multiple years/months")
    print(f"  [ ] Synonyms are diverse and realistic")
    print(f"  [ ] No data leakage between train/val/test")


if __name__ == "__main__":
    main()
