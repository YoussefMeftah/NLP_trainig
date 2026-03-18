import json
import random
from datetime import datetime, timedelta
from transformers import AutoTokenizer
from datasets import Dataset
import re

# ----------------------------
# Configuration
# ----------------------------
INTENT_MAP = {
    "get_sales_by_article": 0,
    "get_sales_by_client": 1,
    "get_purchases_by_article": 2,
    "get_purchases_by_supplier": 3,
    "get_payments": 4,
}

NER_TAG_MAP = {
    "O": 0,
    "B-DATE_START": 1,
    "I-DATE_START": 2,
    "B-DATE_END": 3,
    "I-DATE_END": 4,
    "B-CATEGORY": 5,
    "I-CATEGORY": 6,
}

intents = {
    "get_sales_by_article": {
        "templates": [
            "afficher les ventes par article du {start} au {end}",
            "ventes par article entre {start} et {end}",
            "donne moi les ventes des articles du {start} au {end}",
            "statistiques des ventes articles du {start} au {end}"
        ],
        "synonyms": {
            "ventes": ["chiffre d'affaires", "transactions", "revenus"],
            "article": ["produit", "item", "marchandise"]
        }
    },
    "get_sales_by_client": {
        "templates": [
            "afficher les ventes par client du {start} au {end}",
            "ventes par client entre {start} et {end}",
            "donne moi les ventes des clients du {start} au {end}",
            "chiffre d'affaires par client du {start} au {end}"
        ],
        "synonyms": {
            "client": ["acheteur", "partenaire", "clt"]
        }
    },
    "get_purchases_by_article": {
        "templates": [
            "afficher les achats par article du {start} au {end}",
            "achats par article entre {start} et {end}",
            "statistiques des achats articles du {start} au {end}"
        ],
        "synonyms": {
            "achats": ["approvisionnements", "commandes"]
        }
    },
    "get_purchases_by_supplier": {
        "templates": [
            "afficher les achats par fournisseur du {start} au {end}",
            "achats par fournisseur entre {start} et {end}",
            "statistiques fournisseurs du {start} au {end}"
        ],
        "synonyms": {
            "fournisseur": ["frs", "partenaire", "vendeur"]
        }
    },
    "get_payments": {
        "templates": [
            "afficher les paiements groupés par {category} du {start} au {end}",
            "répartition des paiements par {category} entre {start} et {end}",
            "statistiques des paiements répartis par {category} du {start} au {end}",
            "donne moi les paiements ventilés par {category} du {start} au {end}"
        ],
        "categories": {"client": "client", "mode": "mode", "année": "année", "tous": "tous"},
        "synonyms": {}
    }
}

# ----------------------------
# Pre-compiled regex patterns (avoid recompiling in hot path)
# ----------------------------
ENTITY_PATTERNS = {}

# ----------------------------
# Utilities
# ----------------------------
def random_date():
    start = datetime(2023, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def generate_date_range():
    d1 = random_date()
    d2 = random_date()
    return min(d1, d2), max(d1, d2)

def apply_variations(text, synonyms):
    """Generate variations using synonyms with word-boundary matching"""
    seen = {text}
    variations = [text]
    for word, replacements in synonyms.items():
        for replacement in replacements:
            # Replace all occurrences of the word
            new_text = text.replace(word, replacement)
            if new_text not in seen:
                seen.add(new_text)
                variations.append(new_text)
    return variations

def add_noise(text):
    """Add minor variations (lowercase, abbreviations)"""
    seen = {text}
    variations = [text]

    lower_text = text.lower()
    if lower_text not in seen:
        seen.add(lower_text)
        variations.append(lower_text)

    noise_map = {
        "clients": "clts",
        "articles": "arts",
    }

    for original, abbrev in noise_map.items():
        if original in text:
            new_text = text.replace(original, abbrev)
            if new_text not in seen:
                seen.add(new_text)
                variations.append(new_text)

    return variations

def _get_entity_pattern(entity_value):
    """Get or create pre-compiled regex pattern for entity matching"""
    if entity_value not in ENTITY_PATTERNS:
        ENTITY_PATTERNS[entity_value] = re.compile(
            re.escape(entity_value),
            re.IGNORECASE
        )
    return ENTITY_PATTERNS[entity_value]

def find_entity_spans(text, entity_value, entity_type):
    """Find all character-level spans of an entity in text"""
    if not entity_value or f"B-{entity_type}" not in NER_TAG_MAP:
        return []

    spans = []
    pattern = _get_entity_pattern(entity_value)
    for match in pattern.finditer(text):
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "type": entity_type,
            "value": match.group()  # Use matched text, not input case
        })
    return spans

def align_tags_to_tokens(text, entities, tokenizer):
    """
    Convert character-level entities to token-level NER tags.
    Returns list of NER tag IDs aligned with tokenized text.
    Handles partial token-entity overlaps correctly.

    Special tokens (no character mapping) are labeled as -100, which PyTorch's
    CrossEntropyLoss ignores by default. This prevents the model from learning
    to classify separator/padding tokens.
    """
    encoding = tokenizer(text, truncation=False, padding=False,
                        return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"]

    # Initialize all tags as "O" (outside)
    tags = ["O"] * len(offset_mapping)

    # Map entities to token-level tags
    for entity in entities:
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_type = entity["type"]

        # Validate entity type
        tag_prefix = f"B-{entity_type}"
        if tag_prefix not in NER_TAG_MAP:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {list(set(s.split('-')[1] for s in NER_TAG_MAP.keys() if '-' in s))}")

        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == char_end:  # Special tokens
                continue

            # Check for any overlap between token and entity
            if not (char_end <= entity_start or char_start >= entity_end):
                # Token overlaps with entity
                if tags[token_idx] == "O":  # Only tag if not already assigned
                    if char_start >= entity_start:
                        # Token starts within entity
                        tags[token_idx] = f"B-{entity_type}"
                    else:
                        # Token starts before entity starts (partial overlap)
                        tags[token_idx] = f"I-{entity_type}"

    # Convert tags to IDs with validation
    # Special tokens (offset (0,0)) get -100 label, others use NER_TAG_MAP
    tag_ids = []
    for idx, tag in enumerate(tags):
        if offset_mapping[idx] == (0, 0):  # Special token
            tag_ids.append(-100)
        else:
            if tag not in NER_TAG_MAP:
                raise ValueError(f"Invalid tag '{tag}' not in NER_TAG_MAP: {list(NER_TAG_MAP.keys())}")
            tag_ids.append(NER_TAG_MAP[tag])

    return tag_ids, encoding["input_ids"]

# ----------------------------
# Dataset Generation
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def _generate_samples_for_text(text, intent_name, synonyms, entities_to_find):
    """
    Generate training samples from a single text.

    Args:
        text: Base text with placeholders filled
        intent_name: Intent identifier
        synonyms: Dict of word -> replacement list
        entities_to_find: List of tuples (value, entity_type)

    Returns:
        List of sample dictionaries
    """
    samples = []
    for syn_var in apply_variations(text, synonyms):
        for noisy_var in add_noise(syn_var):
            entities = []
            for entity_value, entity_type in entities_to_find:
                entities.extend(find_entity_spans(noisy_var, entity_value, entity_type))

            if entities:  # Only add if we found entities
                try:
                    ner_tags, input_ids = align_tags_to_tokens(noisy_var, entities, tokenizer)
                    samples.append({
                        "text": noisy_var,
                        "tokens": input_ids,
                        "ner_tags": ner_tags,
                        "intent": INTENT_MAP[intent_name],
                    })
                except ValueError as e:
                    print(f"⚠️  Skipping sample: {e}")
                    continue

    return samples

data_samples = []
REPEAT_FACTOR = 200  # Reduced from 200 to avoid duplicates

for intent_name, intent_data in intents.items():
    templates = intent_data["templates"]
    synonyms = intent_data.get("synonyms", {})
    categories = intent_data.get("categories", {})

    for template in templates:
        for _ in range(REPEAT_FACTOR):
            start, end = generate_date_range()

            if intent_name == "get_payments":
                # Validate that categories exist
                if not categories:
                    raise ValueError(f"Intent '{intent_name}' requires categories but none found")

                for cat_label, cat_value in categories.items():
                    text = template.format(start=start, end=end, category=cat_label)
                    entities_to_find = [
                        (start, "DATE_START"),
                        (end, "DATE_END"),
                        (cat_label, "CATEGORY")
                    ]
                    data_samples.extend(
                        _generate_samples_for_text(text, intent_name, synonyms, entities_to_find)
                    )
            else:
                text = template.format(start=start, end=end)
                entities_to_find = [
                    (start, "DATE_START"),
                    (end, "DATE_END")
                ]
                data_samples.extend(
                    _generate_samples_for_text(text, intent_name, synonyms, entities_to_find)
                )

# ----------------------------
# Split & Create HuggingFace Datasets
# ----------------------------
random.shuffle(data_samples)

total = len(data_samples)
train_split = int(total * 0.8)
val_split = int(total * 0.9)

train_data = data_samples[:train_split]
val_data = data_samples[train_split:val_split]
test_data = data_samples[val_split:]

def _create_dataset(data_samples):
    """Convert list of sample dicts to HuggingFace Dataset"""
    return Dataset.from_dict({
        "text": [d["text"] for d in data_samples],
        "tokens": [d["tokens"] for d in data_samples],
        "ner_tags": [d["ner_tags"] for d in data_samples],
        "intent": [d["intent"] for d in data_samples],
    })

# Create HuggingFace Datasets
train_dataset = _create_dataset(train_data)
val_dataset = _create_dataset(val_data)
test_dataset = _create_dataset(test_data)

# Save as Arrow format (efficient for HuggingFace)
train_dataset.save_to_disk("./data/train")
val_dataset.save_to_disk("./data/val")
test_dataset.save_to_disk("./data/test")

# Also save tag mappings for reference
with open("./data/tag_mapping.json", "w") as f:
    json.dump(NER_TAG_MAP, f, indent=2)

with open("./data/intent_mapping.json", "w") as f:
    json.dump(INTENT_MAP, f, indent=2)

print(f"✅ Dataset generated: {len(data_samples)} samples")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
print(f"💾 Saved to ./data/ (train, val, test directories)")
print(f"🏷️ NER Tags: {NER_TAG_MAP}")
print(f"🎯 Intents: {INTENT_MAP}")
