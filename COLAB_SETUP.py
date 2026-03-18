"""
Google Colab Setup & Training Script
Copy and paste this into Google Colab cells
"""

# ============================================================================
# Cell 1: Install Dependencies
# ============================================================================
# !pip install transformers datasets torch scikit-learn -q

# ============================================================================
# Cell 2: Mount Google Drive (to save model)
# ============================================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================================
# Cell 3: Clone/Upload Repository
# ============================================================================
# Option A: Clone from GitHub
# !git clone https://github.com/YOUR_USERNAME/NLP_training.git
# %cd NLP_training

# Option B: Upload files manually
# - Upload data_genrator.py
# - Upload train.py
# - Upload data/ folder (generated datasets)

# ============================================================================
# Cell 4: Generate Dataset (if needed)
# ============================================================================
# %cd /content/NLP_training
# !python data_genrator.py

# ============================================================================
# Cell 5: Run Training
# ============================================================================
# !python train.py \
#     --output_dir /content/drive/MyDrive/xlm_roberta_model \
#     --num_epochs 5 \
#     --batch_size 16 \
#     --learning_rate 2e-5

# ============================================================================
# Cell 6: Monitor Training (Optional - View Loss Curves)
# ============================================================================
# import json
# import matplotlib.pyplot as plt
#
# # Load training logs
# with open('/content/NLP_training/xlm_roberta_model/trainer_state.json') as f:
#     logs = json.load(f)
#
# losses = [log['loss'] for log in logs['log_history'] if 'loss' in log]
# epochs = range(1, len(losses) + 1)
#
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, losses, 'b-', label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Progression')
# plt.legend()
# plt.grid(True)
# plt.show()

# ============================================================================
# Cell 7: Test the Trained Model
# ============================================================================
# from transformers import AutoTokenizer
# import torch
# import json
#
# MODEL_PATH = '/content/drive/MyDrive/xlm_roberta_model'
#
# # Load model
# from train import XLMRobertaForIntentAndNER
# model = XLMRobertaForIntentAndNER.from_pretrained(MODEL_PATH)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#
# # Load mappings
# with open(f'{MODEL_PATH}/tag_mapping.json') as f:
#     TAG_MAP = json.load(f)
# with open(f'{MODEL_PATH}/intent_mapping.json') as f:
#     INTENT_MAP = json.load(f)
#
# ID_TO_TAG = {v: k for k, v in TAG_MAP.items()}
# ID_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}
#
# def predict(text):
#     encoding = tokenizer(
#         text,
#         truncation=True,
#         padding='max_length',
#         max_length=128,
#         return_tensors='pt'
#     )
#
#     model.eval()
#     with torch.no_grad():
#         outputs = model(
#             input_ids=encoding['input_ids'],
#             attention_mask=encoding['attention_mask'],
#         )
#
#     intent_pred = outputs['intent_logits'].argmax(dim=1).item()
#     ner_preds = outputs['ner_logits'].argmax(dim=2)[0].cpu().numpy()
#
#     # Get tokens
#     tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
#
#     # Filter out special tokens and padding
#     valid_tokens = []
#     valid_ner_tags = []
#     for token, tag_id in zip(tokens, ner_preds):
#         if token not in ['<s>', '</s>', '<pad>']:
#             valid_tokens.append(token)
#             valid_ner_tags.append(ID_TO_TAG.get(int(tag_id), 'O'))
#
#     return {
#         'text': text,
#         'intent': ID_TO_INTENT[intent_pred],
#         'entities': list(zip(valid_tokens, valid_ner_tags))
#     }
#
# # Test
# test_text = "afficher les ventes groupées par article du 2024-01-01 au 2024-12-31"
# result = predict(test_text)
# print(json.dumps(result, indent=2, ensure_ascii=False))

# ============================================================================
# Cell 8: Download Model from Drive (if needed locally)
# ============================================================================
# from google.colab import files
# files.download('/content/drive/MyDrive/xlm_roberta_model/pytorch_model.bin')
# files.download('/content/drive/MyDrive/xlm_roberta_model/config.json')
# files.download('/content/drive/MyDrive/xlm_roberta_model/tokenizer_config.json')

print("Colab notebook cells created! Copy-paste each cell into Google Colab.")
