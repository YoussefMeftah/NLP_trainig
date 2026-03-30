"""
Manual Model Testing Script
Test the trained model interactively with custom phrases.

Simply run: python manual_test.py
Then input a phrase and get intent + NER predictions.
"""

import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
MODEL_DIR = "./model"
DATA_DIR = "./data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Load mappings
with open(f"{DATA_DIR}/intent_mapping.json") as f:
    INTENT_MAP = json.load(f)
with open(f"{DATA_DIR}/tag_mapping.json") as f:
    TAG_MAP = json.load(f)

ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}
ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}

MAX_LENGTH = 128

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

        return intent_logits, ner_logits


# ============================================================================
# Initialize Model and Tokenizer
# ============================================================================
print("Loading model...")
try:
    # Load config
    with open(f"{MODEL_DIR}/model_config.json") as f:
        config = json.load(f)

    model_name = config["model_name"]
    num_labels_intent = config["num_labels_intent"]
    num_labels_ner = config["num_labels_ner"]

    # Initialize model
    model = XLMRobertaForIntentAndNER(
        model_name=model_name,
        num_labels_intent=num_labels_intent,
        num_labels_ner=num_labels_ner
    )

    # Load weights from safetensors
    from safetensors.torch import load_file
    state_dict = load_file(f"{MODEL_DIR}/model.safetensors")
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("✓ Model loaded successfully!")
    print(f"✓ Intent classes: {list(ID_TO_INTENT.values())}")
    print(f"✓ NER tags: {list(ID_TO_TAG.values())}\n")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)


# ============================================================================
# Post-Processing: Relative Date Detection
# ============================================================================
import re

# French relative date patterns
RELATIVE_DATE_PATTERNS = {
    "DATE_START": [
        # This/current period
        r'\b(ce\s+mois|ce\s+mois-ci|ce\s+mois\s+ci)\b',
        r'\b(cette\s+semaine|cette\s+semaine-ci)\b',
        r'\b(aujourd\'hui|today)\b',
        r'\b(ce\s+jour|ce\s+jour-ci)\b',
        
        # Past periods
        r'\b(le\s+mois\s+dernier|mois\s+dernier)\b',
        r'\b(la\s+semaine\s+dernière|semaine\s+dernière)\b',
        r'\b(hier|yesterday)\b',
        r'\b(l\'année\s+dernière|année\s+dernière)\b',
        r'\b(le\s+trimestre\s+dernier|trimestre\s+dernier)\b',
        
        # Relative past
        r'\b(il\s+y\s+a\s+\d+\s+jours?)\b',
        r'\b(il\s+y\s+a\s+\d+\s+semaines?)\b',
        r'\b(il\s+y\s+a\s+\d+\s+mois?)\b',
        r'\b(il\s+y\s+a\s+\d+\s+ans?)\b',
    ],
    "DATE_END": [
        # Future periods
        r'\b(le\s+mois\s+prochain|mois\s+prochain)\b',
        r'\b(la\s+semaine\s+prochaine|semaine\s+prochaine)\b',
        r'\b(demain|tomorrow)\b',
        r'\b(l\'année\s+prochaine|année\s+prochaine)\b',
        r'\b(le\s+trimestre\s+prochain|trimestre\s+prochain)\b',
        
        # Relative future
        r'\b(dans\s+\d+\s+jours?)\b',
        r'\b(dans\s+\d+\s+semaines?)\b',
        r'\b(dans\s+\d+\s+mois?)\b',
        r'\b(dans\s+\d+\s+ans?)\b',
    ],
    "CATEGORY": [
        # Common business categories
        r'\b(marchandises?|products?|articles?)\b',
        r'\b(ventes?|sales?)\b',
        r'\b(achats?|purchases?|buys?)\b',
        r'\b(payments?|paiements?)\b',
        r'\b(clients?|customers?)\b',
        r'\b(fournisseurs?|suppliers?)\b',
    ]
}


def extract_relative_dates(text):
    """
    Extract relative dates and entities using regex patterns.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of detected entities with their types
    """
    entities = []
    
    for entity_type, patterns in RELATIVE_DATE_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "word": match.group(0),
                    "tag": entity_type,
                    "source": "regex"  # Mark as regex-based
                })
    
    return entities


# ============================================================================
# Inference Function
# ============================================================================
def predict(text):
    """
    Predict intent and NER tags for the given text.
    
    Args:
        text (str): Input phrase
    
    Returns:
        dict: Contains intent and NER predictions with explanations
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # Forward pass
    with torch.no_grad():
        intent_logits, ner_logits = model(input_ids, attention_mask)

    # Get predictions
    intent_pred = intent_logits.argmax(dim=1).item()
    intent_label = ID_TO_INTENT[intent_pred]
    intent_conf = torch.softmax(intent_logits, dim=1)[0, intent_pred].item()

    # NER predictions (for non-special tokens)
    ner_pred = ner_logits.argmax(dim=2)[0].cpu().numpy()

    # Decode tokens and align with NER tags
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    # Create word-level NER predictions by merging subword tokens
    ner_results = []
    word_tokens = []
    current_word = ""
    current_tag = "O"

    for i, (token, tag_id) in enumerate(zip(tokens, ner_pred)):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_word and current_tag != "O":
                ner_results.append({"word": current_word, "tag": current_tag})
            elif current_word:
                ner_results.append({"word": current_word, "tag": "O"})
            current_word = ""
            current_tag = "O"
            continue

        tag = ID_TO_TAG[tag_id]

        # Handle subword tokens (##)
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                if current_tag != "O":
                    ner_results.append({"word": current_word, "tag": current_tag})
                else:
                    ner_results.append({"word": current_word, "tag": "O"})
            current_word = token

            # Update tag (handle B- and I- tags)
            if tag.startswith("B-"):
                current_tag = tag[2:]  # Remove B- prefix
            elif tag.startswith("I-"):
                current_tag = tag[2:]  # Remove I- prefix
            else:
                current_tag = "O"

    # Add last word
    if current_word:
        if current_tag != "O":
            ner_results.append({"word": current_word, "tag": current_tag})
        else:
            ner_results.append({"word": current_word, "tag": "O"})

    # Extract relative dates using regex patterns
    regex_entities = extract_relative_dates(text)

    # Combine model predictions with regex-based entities
    model_predictions = [r for r in ner_results if r["tag"] != "O"]
    
    # Merge both sources (avoid duplicates)
    all_predictions = model_predictions.copy()
    for regex_entity in regex_entities:
        # Check if this entity is not already detected by the model
        is_duplicate = any(
            e["word"].lower() == regex_entity["word"].lower() and e["tag"] == regex_entity["tag"]
            for e in model_predictions
        )
        if not is_duplicate:
            all_predictions.append(regex_entity)

    return {
        "input_text": text,
        "intent": intent_label,
        "intent_confidence": f"{intent_conf:.2%}",
        "ner_predictions": all_predictions,
        "all_tokens": ner_results
    }


# ============================================================================
# Main Interactive Loop
# ============================================================================
def main():
    print("=" * 70)
    print("MANUAL MODEL TESTER")
    print("=" * 70)
    print("\nEnter phrases to test the model's intent classification and NER.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Enter phrase: ").strip()

            if not user_input:
                print("Please enter a valid phrase.\n")
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            # Get predictions
            result = predict(user_input)

            # Display results
            print("\n" + "=" * 70)
            print(f"Input: {result['input_text']}")
            print("-" * 70)
            print(f"INTENT: {result['intent']} (confidence: {result['intent_confidence']})")
            print("-" * 70)

            if result["ner_predictions"]:
                print("NAMED ENTITIES DETECTED:")
                for entity in result["ner_predictions"]:
                    source = f" ({entity.get('source', 'model')})" if entity.get('source') == 'regex' else ""
                    print(f"  • {entity['word']:20} → {entity['tag']}{source}")
            else:
                print("NAMED ENTITIES: None detected")

            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error during prediction: {e}\n")


if __name__ == "__main__":
    main()
