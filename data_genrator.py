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
            "statistiques des ventes articles du {start} au {end}",
            "montrez les ventes articles du {start} au {end}",
            "je veux connaître les ventes par article du {start} au {end}",
            "tableau ventes articles {start} à {end}",
            "quelles sont les ventes des articles du {start} au {end}",
            "ventes produits du {start} au {end}",
            "détail des ventes par article de {start} à {end}",
            "génère un rapport ventes articles du {start} au {end}",
            "sales by article from {start} to {end}",
            "affiche ventes par produit entre {start} et {end}",
            "ventes marchandises {start} {end}",
            "donne le chiffre d'affaires par article du {start} au {end}",
        ],
        "paraphrases": [
            # Different sentence structures
            "je souhaite voir les ventes par article pour la période du {start} au {end}",
            "peux-tu afficher un rapport détaillé des ventes articles entre {start} et {end}",
            "merci de me fournir les ventes par article depuis le {start} jusqu'au {end}",
            "montre-moi combien d'articles ont été vendus du {start} au {end}",
            "je dois obtenir les statistiques de ventes pour chaque article de {start} à {end}",
            "affiche le volume de ventes par référence article pour {start} - {end}",
            "peux-tu générer un bilan des ventes par produit entre le {start} et le {end}",
            "quand je veux voir les totales de ventes par article, période {start} à {end}",
            "donne moi un détail complet des articles vendus du {start} au {end}",
            "j'aimerais connaître les ventes par article pour la plage {start} au {end}",
        ],
        "synonyms": {
            "ventes": ["chiffre d'affaires", "transactions", "revenus", "CA", "ventes brutes"],
            "article": ["produit", "item", "marchandise", "bien", "référence"],
            "afficher": ["montrez", "montrer", "affiche", "génère"],
        }
    },
    "get_sales_by_client": {
        "templates": [
            "afficher les ventes par client du {start} au {end}",
            "ventes par client entre {start} et {end}",
            "donne moi les ventes des clients du {start} au {end}",
            "chiffre d'affaires par client du {start} au {end}",
            "montrez les ventes clients du {start} au {end}",
            "sales by customer from {start} to {end}",
            "statistiques ventes clients {start} à {end}",
            "je veux voir les ventes par client du {start} au {end}",
            "tableau des ventes par client du {start} au {end}",
            "ventes acheteur du {start} au {end}",
            "détail ventes par client de {start} à {end}",
            "rapport ventes clients du {start} au {end}",
            "affiche les ventes clts du {start} au {end}",
            "quelles ventes par client du {start} au {end}",
            "ventes partenaires du {start} au {end}",
        ],
        "paraphrases": [
            # Different sentence structures
            "je souhaite connaître les ventes réalisées par chaque client entre {start} et {end}",
            "peux-tu afficher un rapport détaillé des ventes par client pour {start} - {end}",
            "merci de me fournir le chiffre d'affaires par client du {start} au {end}",
            "montre-moi les totales de ventes pour chaque client entre le {start} et le {end}",
            "j'aimerais voir un détail des ventes cliente pour la période {start} à {end}",
            "affiche le montant des ventes associées à chaque client du {start} au {end}",
            "peux-tu générer un bilan des ventes par acheteur entre {start} et {end}",
            "quand je veux un résumé des ventes par client, période {start} à {end}",
            "donne moi un détail complet des ventes par cliente pour {start} - {end}",
            "j'aimerais connaître les ventes réalisées avec chaque client, plage {start} au {end}",
        ],
        "synonyms": {
            "client": ["acheteur", "partenaire", "clt", "customer", "acheteurs"],
            "ventes": ["chiffre d'affaires", "revenus", "transactions", "CA"],
            "afficher": ["montrez", "montrer", "affiche", "génère"],
        }
    },
    "get_purchases_by_article": {
        "templates": [
            "afficher les achats par article du {start} au {end}",
            "achats par article entre {start} et {end}",
            "statistiques des achats articles du {start} au {end}",
            "donne moi les achats articles du {start} au {end}",
            "montrez les achats par produit du {start} au {end}",
            "purchases by article from {start} to {end}",
            "tableau achats articles {start} à {end}",
            "détail achats par article du {start} au {end}",
            "je veux voir les achats articles du {start} au {end}",
            "affiche les achats par marchandise du {start} au {end}",
            "rapport achats articles du {start} au {end}",
            "affiche achats articles du {start} au {end}",
            "achats produits du {start} au {end}",
            "commandes par article du {start} au {end}",
            "approvisionnements articles {start} {end}",
        ],
        "paraphrases": [
            # Different sentence structures
            "je souhaite voir les achats par article pour la période du {start} au {end}",
            "peux-tu afficher un rapport détaillé des achats articles entre {start} et {end}",
            "merci de me fournir les achats par article depuis le {start} jusqu'au {end}",
            "montre-moi les quantités achetées pour chaque article du {start} au {end}",
            "j'aimerais connaître les achats par référence article pour {start} - {end}",
            "affiche le montant des approvisionnements par article du {start} au {end}",
            "peux-tu générer un bilan des achats par produit entre {start} et {end}",
            "quand je veux voir les totales d'achats par article, période {start} à {end}",
            "donne moi un détail complet des approvisionnements par article pour {start} - {end}",
            "j'aimerais obtenir les statistiques d'achats par article, plage {start} au {end}",
        ],
        "synonyms": {
            "achats": ["approvisionnements", "commandes", "acquisitions", "achats totaux"],
            "article": ["produit", "item", "marchandise", "bien", "référence"],
            "afficher": ["montrez", "montrer", "affiche", "génère"],
        }
    },
    "get_purchases_by_supplier": {
        "templates": [
            "afficher les achats par fournisseur du {start} au {end}",
            "achats par fournisseur entre {start} et {end}",
            "statistiques fournisseurs du {start} au {end}",
            "donne moi les achats des fournisseurs du {start} au {end}",
            "montrez les achats par fournisseur du {start} au {end}",
            "purchases by supplier from {start} to {end}",
            "tableau achats fournisseurs {start} à {end}",
            "détail achats par fournisseur du {start} au {end}",
            "je veux voir les achats par fournisseur du {start} au {end}",
            "rapport achats fournisseurs du {start} au {end}",
            "affiche achats par supplier du {start} au {end}",
            "achats partenaires du {start} au {end}",
            "commandes fournisseurs du {start} au {end}",
            "affiche les achats par vendeur du {start} au {end}",
            "approvisionnements fournisseurs {start} {end}",
        ],
        "paraphrases": [
            # Different sentence structures
            "je souhaite connaître les achats effectués auprès de chaque fournisseur entre {start} et {end}",
            "peux-tu afficher un rapport détaillé des achats par fournisseur pour {start} - {end}",
            "merci de me fournir les statistiques d'achats par fournisseur du {start} au {end}",
            "montre-moi le montant des achats pour chaque fournisseur entre le {start} et le {end}",
            "j'aimerais voir un détail des achats par partenaire fournisseur pour {start} à {end}",
            "affiche le volume des commandes passées auprès de chaque fournisseur du {start} au {end}",
            "peux-tu générer un bilan des achats par vendeur entre {start} et {end}",
            "quand je veux un résumé des achats par fournisseur, période {start} à {end}",
            "donne moi un détail complet des approvisionnements par fournisseur pour {start} - {end}",
            "j'aimerais obtenir les statistiques d'achats auprès de chaque fournisseur, plage {start} au {end}",
        ],
        "synonyms": {
            "fournisseur": ["frs", "partenaire", "vendeur", "supplier", "fournisseurs"],
            "achats": ["commandes", "approvisionnements", "acquisitions"],
            "afficher": ["montrez", "montrer", "affiche", "génère"],
        }
    },
    "get_payments": {
        "templates": [
            "afficher les paiements groupés par {category} du {start} au {end}",
            "répartition des paiements par {category} entre {start} et {end}",
            "statistiques des paiements répartis par {category} du {start} au {end}",
            "donne moi les paiements ventilés par {category} du {start} au {end}",
            "paiements par {category} du {start} au {end}",
            "montrez les paiements par {category} du {start} au {end}",
            "payments by {category} from {start} to {end}",
            "tableau paiements par {category} {start} à {end}",
            "détail paiements {category} du {start} au {end}",
            "je veux voir les paiements par {category} du {start} au {end}",
            "rapport paiements {category} du {start} au {end}",
            "affiche les paiements par {category} du {start} au {end}",
            "répartition paiements {category} entre {start} et {end}",
            "donne paiements groupés {category} du {start} au {end}",
            "paiement groupé par {category} de {start} à {end}",
        ],
        "paraphrases": [
            # Different sentence structures
            "je souhaite voir les détails des paiements ventilés par {category} pour {start} à {end}",
            "peux-tu afficher un rapport détaillé des paiements par {category} entre {start} et {end}",
            "merci de me fournir la répartition des paiements par {category} du {start} au {end}",
            "montre-moi combien a été payé par {category} entre le {start} et le {end}",
            "j'aimerais connaître le détail des paiements groupés par {category} pour {start} - {end}",
            "affiche le montant total des paiements répartis par {category} du {start} au {end}",
            "peux-tu générer un bilan des paiements par {category} entre {start} et {end}",
            "quand je veux un résumé des paiements par {category}, période {start} à {end}",
            "donne moi un détail complet des paiements ventilés par {category} pour {start} - {end}",
            "j'aimerais obtenir les statistiques de paiements par {category}, plage {start} au {end}",
        ],
        "categories": {
            "client": "client", 
            "mode": "mode", 
            "mode de paiement": "mode de paiement",
            "année": "année", 
            "année fiscale": "année fiscale",
            "tous": "tous",
            "type": "type",
            "devise": "devise",
        },
        "synonyms": {
            "paiements": ["versements", "règlements", "transactions"],
            "afficher": ["montrez", "montrer", "affiche", "génère"],
        }
    }
}

# ----------------------------
# Pre-compiled regex patterns (avoid recompiling in hot path)
# ----------------------------
ENTITY_PATTERNS = {}

# ----------------------------
# Utilities
# ----------------------------
def get_relative_date_expression():
    """Generate random relative date expressions in French"""
    expressions = [
        # Current periods
        "ce mois",
        "ce mois-ci",
        "cette semaine",
        "cette semaine-ci",
        "aujourd'hui",
        "ce jour",
        
        # Past periods
        "le mois dernier",
        "la semaine dernière",
        "hier",
        "l'année dernière",
        "année dernière",
        "le trimestre dernier",
        "les 3 derniers mois",
        "les 3 dernières semaines",
        "les 7 derniers jours",
        "les 30 derniers jours",
        "les 12 derniers mois",
        
        # Month references
        "mois 1",
        "mois 2",
        "mois 3",
        "mois 4",
        "mois 5",
        "mois 6",
        "mois 7",
        "mois 8",
        "mois 9",
        "mois 10",
        "mois 11",
        "mois 12",
        
        # Quarter references
        "la premier trimestre de cette année",
        "le deuxième trimestre de cette année",
        "le troisième trimestre de cette année",
        "le quatrième trimestre de cette année",
        "le premier trimestre",
        "deuxième trimestre",
        "trimestre 1",
        "trimestre 2",
        "trimestre 3",
        "trimestre 4",
        
        # Future periods
        "le mois prochain",
        "la semaine prochaine",
        "demain",
        "l'année prochaine",
        "année prochaine",
        "le trimestre prochain",
    ]
    return random.choice(expressions)


def random_date(format_type=None):
    """Generate random date with optional format variation"""
    start = datetime(2023, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    date = start + timedelta(days=random.randint(0, delta.days))
    
    # Random format variations
    if format_type is None:
        format_type = random.choice([0, 1, 2, 3])
    
    if format_type == 0:  # YYYY-MM-DD
        return date.strftime("%Y-%m-%d")
    elif format_type == 1:  # DD/MM/YYYY
        return date.strftime("%d/%m/%Y")
    elif format_type == 2:  # YYYY/MM/DD
        return date.strftime("%Y/%m/%d")
    elif format_type == 3:  # DD-MM-YYYY
        return date.strftime("%d-%m-%Y")
    else:  # Text format (Jan, Feb, etc.)
        return date.strftime("%B %d, %Y")

def generate_date_range():
    """Generate date range with consistent or varied formats"""
    # Decide on format (keep consistent within a range for realism)
    fmt = random.choice([0, 1, 2, 3])
    d1 = random_date(format_type=fmt)
    d2 = random_date(format_type=fmt)
    return min(d1, d2), max(d1, d2)

def apply_variations(text, synonyms, paraphrases=None):
    """Generate variations using synonyms and paraphrases with better coverage"""
    seen = {text}
    variations = [text]
    
    # Add paraphrases (if provided)
    if paraphrases:
        for paraphrase in paraphrases[:10]:  # Limit paraphrases to avoid explosion
            if paraphrase not in seen:
                seen.add(paraphrase)
                variations.append(paraphrase)
    
    # Single word replacements
    for word, replacements in synonyms.items():
        for replacement in replacements:
            new_text = text.replace(word, replacement)
            if new_text not in seen:
                seen.add(new_text)
                variations.append(new_text)
    
    # Multi-word replacements (combine multiple synonyms)
    if len(synonyms) > 1:
        words = list(synonyms.keys())
        # Try replacing 2 words at a time
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                rep1 = random.choice(synonyms[word1])
                rep2 = random.choice(synonyms[word2])
                new_text = text.replace(word1, rep1).replace(word2, rep2)
                if new_text not in seen and len(variations) < 100:  # Limit variations
                    seen.add(new_text)
                    variations.append(new_text)
    
    return variations[:30]  # Return top 30 variations to avoid explosion

def add_noise(text):
    """Add multiple types of text variations"""
    seen = {text}
    variations = [text]

    # Lowercase
    lower_text = text.lower()
    if lower_text not in seen:
        seen.add(lower_text)
        variations.append(lower_text)

    # Uppercase
    upper_text = text.upper()
    if upper_text not in seen:
        seen.add(upper_text)
        variations.append(upper_text)

    # Mixed case
    title_text = text.title()
    if title_text not in seen:
        seen.add(title_text)
        variations.append(title_text)

    # Abbreviations
    abbrev_map = {
        "clients": "clts",
        "articles": "arts",
        "fournisseur": "frs",
        "fournisseurs": "frss",
        "client": "clt",
        "article": "art",
    }

    for original, abbrev in abbrev_map.items():
        if original in text:
            new_text = text.replace(original, abbrev)
            if new_text not in seen:
                seen.add(new_text)
                variations.append(new_text)
    
    # Extra spaces
    extra_space_text = re.sub(r'\s+', '  ', text)  # Double spaces
    if extra_space_text not in seen:
        seen.add(extra_space_text)
        variations.append(extra_space_text)

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

def _generate_samples_for_text(text, intent_name, synonyms, entities_to_find, paraphrases=None):
    """
    Generate training samples from a single text.

    Args:
        text: Base text with placeholders filled
        intent_name: Intent identifier
        synonyms: Dict of word -> replacement list
        entities_to_find: List of tuples (value, entity_type)
        paraphrases: List of paraphrased template variations

    Returns:
        List of sample dictionaries
    """
    samples = []
    for syn_var in apply_variations(text, synonyms, paraphrases):
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
REPEAT_FACTOR = 20  # Increased from 200 - generate 50x more unique samples to prevent overfitting

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
