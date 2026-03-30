# Fine-tuning with Enriched Dataset (Google Colab)

Complete guide for fine-tuning your NLP model with paraphrasing + relative date understanding.

---

## 📋 What's New

✅ **Paraphrasing**: Multiple sentence structures for each intent  
✅ **Relative Date Expressions**: Supports "ce mois", "la semaine dernière", "les 3 derniers mois", etc.  
✅ **50% Relative Dates in Dataset**: Mix of precise dates and relative expressions  
✅ **Efficient Fine-tuning**: 2 epochs with lower learning rate (keeps old knowledge)

---

## 🚀 Step-by-Step Guide for Google Colab

### Step 1: Prepare Your Data Locally

```bash
python data_genrator.py
```

This generates a much richer dataset with:
- Paraphrasing templates
- Relative date expressions  
- Synonym variations
- Noise/case variations
- ~2x-3x more training samples

**Output**: `./data/` folder with train/val/test datasets

---

### Step 2: Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder named `NLP_Data` (or any name)
3. Upload the entire `data` folder to Google Drive
4. The structure should be:
   ```
   /My Drive/
   ├── data/
   │   ├── train/
   │   ├── val/
   │   ├── test/
   │   ├── intent_mapping.json
   │   └── tag_mapping.json
   ```

---

### Step 3: Set Up Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook or open an existing one
3. Run these setup cells:

```python
# Cell 1: Install dependencies
!pip install transformers datasets safetensors torch scikit-learn

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Verify data exists
import os
data_path = '/content/drive/MyDrive/data'  # Adjust path if needed
os.listdir(data_path)
```

---

### Step 4: Run Fine-tuning in Google Colab

**Option A: Upload script to Colab**

```python
# Upload the fine_tune_colab.py file and run:
!python fine_tune_colab.py
```

**Option B: Copy script content directly to Colab**

1. Copy the entire content of `fine_tune_colab.py`
2. Paste into a Colab cell
3. Run the cell

---

### Step 5: Monitor Training

The training will show:
- Loss curves
- Intent accuracy
- NER F1 score
- Early stopping (if validation doesn't improve after 3 checks)

Expected time: **30-45 minutes** on Colab GPU

---

### Step 6: Results

When complete, you'll have:

```
/My Drive/
├── xlm_roberta_model_finetuned/  ← NEW fine-tuned model
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── model_config.json
│   ├── test_results.json
│   └── config.json
└── data/  ← original training data
```

---

## 🎯 Key Differences from Initial Training

| Aspect | Initial Training | Fine-tuning |
|--------|-----------------|------------|
| **Data** | Basic templates | Paraphrasing + relative dates |
| **Learning Rate** | 5e-5 (higher) | 1e-5 (lower) ✅ |
| **Epochs** | 3+ | 2 ✅ |
| **Starting Point** | Random weights | Existing model ✅ |
| **Training Time** | 4-6 hours | 30-45 min ✅ |
| **Old Knowledge** | N/A | Preserved ✅ |

---

## 📊 Expected Improvements

After fine-tuning on enriched data:

✅ Better handling of French relative dates  
✅ Improved paraphrase understanding  
✅ Higher overall accuracy (typically +3-5% on test set)  
✅ More robust to real-world variations  

---

## 🔧 Configuration Options

If you want to adjust fine-tuning, edit `fine_tune_colab.py`:

```python
# Change number of epochs
num_train_epochs=2,  # ← Adjust here

# Change learning rate
learning_rate=1e-5,  # ← Lower = safer, slower learning

# Change batch size
per_device_train_batch_size=16,  # ← Higher = faster but more memory

# Change evaluation frequency
eval_steps=100,  # ← How often to evaluate
```

---

## 🐛 Troubleshooting

### Error: "data directory not found"
- Check that your `data` folder path matches in the script
- Default expects `/content/drive/MyDrive/data`
- Adjust `DATA_DIR` variable if needed

### Error: "CUDA out of memory"
- Reduce batch size: `per_device_train_batch_size=8`
- Or reduce sequence length: `MAX_LENGTH=64`

### Model training is very slow
- Ensure you're using GPU in Colab (Runtime → Change runtime type → GPU)
- Check `fp16=True` is enabled for faster training

### Fine-tuned model performs worse
- This shouldn't happen, but if it does:
  - Check data quality in generated dataset
  - Increase learning rate slightly: `1e-5` → `2e-5`
  - Run for more epochs: `num_train_epochs=3`

---

## 📁 Using the Fine-tuned Model

After fine-tuning, update your `manual_test.py`:

```python
# Change model path from:
MODEL_DIR = "./model"

# To:
MODEL_DIR = "/content/drive/MyDrive/xlm_roberta_model_finetuned"  # Colab path
# Or locally:
MODEL_DIR = "./model_finetuned"
```

Then test it:

```bash
python manual_test.py
```

---

## 🎓 What's Happening Under the Hood

1. **Data Generation** (`data_genrator.py`):
   - Creates templates with placeholders
   - Applies synonym variations
   - Adds paraphrasing (different sentence structures)
   - Includes relative date expressions (50% of dataset)
   - Applies noise (case variations, abbrevations)
   - Generates token-level NER tags

2. **Fine-tuning** (`fine_tune_colab.py`):
   - Loads your existing trained model
   - Keeps encoder weights intact
   - Updates the classifier heads slightly
   - Uses lower learning rate (1e-5)
   - Trains for just 2 epochs
   - Saves best checkpoint based on NER F1 score

3. **Result**: Model understand more variations while retaining old knowledge

---

## 💡 Advanced: Adding New Intents

If you want to add a new intent later:

1. Edit `intents` dictionary in `data_genrator.py`
2. Add new intent with templates and paraphrases
3. Run `python data_genrator.py`
4. Run `fine_tune_colab.py` again

The model will automatically adapt!

---

## 📞 Questions?

- Check the generated `test_results.json` for detailed metrics
- Read training logs in `xlm_roberta_model_finetuned/logs/`
- Review the relative date patterns in `data_genrator.py` around line 30+

Happy fine-tuning! 🚀
