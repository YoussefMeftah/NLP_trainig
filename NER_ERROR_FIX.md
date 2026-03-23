# NER Tag Evaluation Error - Debugging Guide

## The Error You Encountered

```
ValueError: Number of classes, 0, does not match size of target_names, 7
```

### What This Means
The model's NER evaluation code was trying to print metrics for 7 NER tag classes, but no NER predictions were actually collected during evaluation (0 classes found).

---

## Root Causes (in order of likelihood)

### Cause 1: Dataset Field Name Mismatch ✅ FIXED
**What happened:**
- During training (`train.py`), the NER field is renamed from `ner_tags` → `ner_labels`
- Test script (`test_model.py`) was looking for `ner_tags` field that no longer existed
- Result: No NER data was loaded, causing empty predictions

**Status:** ✅ **FIXED** in latest test_model.py

**Solution applied:**
```python
# Now test_model.py looks for both field names
if "ner_labels" in sample:
    ner_true = sample["ner_labels"]
elif "ner_tags" in sample:
    ner_true = sample["ner_tags"]
```

---

### Cause 2: Empty/All-Padding NER Tags
**What this means:**
- All NER tags in your dataset are `-100` (padding/ignored index)
- No actual entity tags exist in the data
- Model never learned to recognize entities

**How to check:**
```bash
python diagnose_ner.py
```

**Expected output if this is the issue:**
```
⚠️  ALL samples are empty (all tags are -100)
This is why NER evaluation is failing!
```

**If confirmed, fix by:**
1. Check if data generation is creating entity tags properly
2. Run: `python analyze_data.py` to inspect raw dataset
3. Ensure `data_generator.py` is actually extracting entities
4. Regenerate dataset: Delete `./data/` and run `python data_generator.py`

---

### Cause 3: Dataset Preprocessing Issues
**What could go wrong:**
- NER tags get lost during the tokenization/preprocessing step in `train.py`
- Alignment between tokens and tags fails
- Tags get truncated or removed

**How to verify:**
```bash
python diagnose_ner.py
```

**Check the output section "Sample Details":**
- If `NER Tags (first 20)` shows all `-100`, the tags were lost

---

## How to Test the Fix

### Step 1: Verify the field names in your saved dataset
```python
from datasets import load_from_disk

test_data = load_from_disk("./data/test")
sample = test_data[0]
print("Sample fields:", list(sample.keys()))
```

**Expected output:**
```
Sample fields: ['input_ids', 'attention_mask', 'ner_labels', 'intent_labels']
```

If you see `ner_tags` instead of `ner_labels`, you're using raw (unprocessed) data.

### Step 2: Run the diagnostic
```bash
python diagnose_ner.py
```

This will tell you:
- How many samples have actual NER tags
- Distribution of entity types
- Whether data looks healthy

### Step 3: Test the model evaluation again
```bash
python test_model.py --model_dir ./model/checkpoint-1395
```

**Expected behavior:**
- Either shows proper NER metrics (if data is good)
- Or shows helpful diagnostic message explaining the issue

---

## Quick Fix Checklist

If you're still getting the error:

- [ ] **Step 1**: Update test_model.py to latest version (handles both field names)
- [ ] **Step 2**: Run `python diagnose_ner.py` to identify root cause
- [ ] **Step 3**: If all tags are -100:
  - [ ] Check data_generator.py is creating entities
  - [ ] Delete ./data/ directory
  - [ ] Run `python data_generator.py` to regenerate
  - [ ] Run `python train.py` to retrain
- [ ] **Step 4**: Test again with `python test_model.py`

---

## Understanding NER Tag Fields

### Raw Dataset (after data_generator.py)
```
sample = {
    "text": "afficher les ventes par article du 2024-01-01 au 2024-12-31",
    "tokens": [2, 4, 5, ...],  # Token IDs
    "ner_tags": [0, 0, 0, 1, 0, 0, 2, 3, 4, 0, 0, 0],  # Raw tags
    "intent": 0
}
```

### Processed Dataset (after train.py preprocessing)
```
sample = {
    "input_ids": [101, 2, 4, 5, ...],  # Tokenized
    "attention_mask": [1, 1, 1, 1, ...],
    "ner_labels": [-100, 0, 0, 0, 1, 0, 0, 2, 3, 4, -100, ...],  # Padded & renamed
    "intent_labels": 0
}
```

**Key difference:** `ner_labels` has `-100` padding tokens, while `ner_tags` doesn't.

---

## Expected Results After Fix

When you run the test with properly configured data:

```
======================================================================
NER PERFORMANCE METRICS
======================================================================

Overall F1 (weighted): 0.8234

Per-Tag Performance:
  B-DATE_START       : P=0.92 R=0.88 F1=0.90
  I-DATE_START       : P=0.87 R=0.85 F1=0.86
  B-DATE_END         : P=0.90 R=0.89 F1=0.89
  I-DATE_END         : P=0.86 R=0.84 F1=0.85
  B-CATEGORY         : P=0.78 R=0.82 F1=0.80
  I-CATEGORY         : P=0.75 R=0.79 F1=0.77

Detailed Classification Report:
                 precision    recall  f1-score   support
  B-DATE_START       0.92      0.88      0.90       524
  ...
```

Or if data is empty:

```
⚠️  WARNING: No NER predictions found!
   This could mean:
   1. Dataset has no labeled NER tags (all -100)
   2. NER tags are not being extracted from samples
   3. Data loader is not returning ner_tags field

   Run: python analyze_data.py
   to check dataset structure.
```

---

## Files Modified

- **test_model.py**: Now handles both `ner_tags` and `ner_labels` field names
- **test_model.py**: Better error handling for empty predictions
- **test_model.py**: Enhanced diagnostic messages when issues occur

---

## Next Steps

1. **Run diagnostic:**
   ```bash
   python diagnose_ner.py
   ```

2. **Review output** - it will clearly tell you if data is healthy or what's wrong

3. **If data is corrupt:**
   ```bash
   # Full dataset regeneration
   rm -r ./data
   python data_generator.py
   python train.py --num_epochs 3
   ```

4. **Test again:**
   ```bash
   python test_model.py --model_dir ./xlm_roberta_model
   ```

---

## Questions?

If you're still having issues:
1. Run `python diagnose_ner.py` and share the output
2. Check `python analyze_data.py --output_report debug.json` and review the JSON
3. Verify the data directory has train/, val/, test/ folders with actual data
