from datasets import load_from_disk

# Load and inspect the training data
train = load_from_disk("./data/train")
print(f"Train samples: {len(train)}")
print(f"\nFirst sample:")
print(f"  Text: {train['text'][0]}")
print(f"  Intent: {train['intent'][0]}")
print(f"  NER tags: {train['ner_tags'][0][:10]}...")  # First 10 tags
print(f"  Special tokens (-100): {-100 in train['ner_tags'][0]}")