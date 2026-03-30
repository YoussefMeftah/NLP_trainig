# XLM-RoBERTa Testing & Data Diagnosis Guide

## Quick Start

### 1. Test Your Model
```bash
# Basic evaluation on test set
python test_model.py --model_dir ./xlm_roberta_model

# Interactive testing mode (test individual sentences)
python test_model.py --model_dir ./xlm_roberta_model --interactive

# Evaluate on subset of data
python test_model.py --model_dir ./xlm_roberta_model --test_samples 100
```

### 2. Analyze Your Dataset
```bash
# Generate comprehensive analysis report
python analyze_data.py

# Save detailed JSON report
python analyze_data.py --output_report analysis_report.json
```

---

## Understanding the Evaluation Metrics

### Intent Classification Metrics

**Accuracy**: Overall % of correct intent predictions
- **< 60%**: Poor → increase dataset 3-5x, review templates
- **60-80%**: Acceptable but needs work → add 50% more data
- **80-90%**: Good → minor improvements possible
- **> 90%**: Excellent → ready for production

**Per-Intent Performance**: Shows which intents are confusing
- If `get_sales_by_article` has 65% acc but others have 95%:
  - Add more diverse templates for that intent
  - Ensure it has >= 100 unique samples
  - Check for overlap with similar intents

---

### NER Performance Metrics

**F1 Score**: Harmonic mean of precision & recall
- **< 60%**: Poor → likely data quality issue or insufficient samples
- **60-75%**: Acceptable → needs improvement
- **75-85%**: Good → ready for field testing
- **> 85%**: Excellent → minimal errors

**Precision** vs **Recall**:
- **Low Precision, High Recall**: Model finds entities but has false positives
  - → Add more negative examples (text WITHOUT entities)
  - → Increase training epochs
  
- **High Precision, Low Recall**: Model misses some entities
  - → Add more positive examples with that entity
  - → Increase dataset size 2-3x

**Per-Tag Performance**:
```
B-DATE_START: P=0.92 R=0.88 F1=0.90  ✅ Good
I-DATE_START: P=0.65 R=0.70 F1=0.67  ⚠️  Needs work
B-CATEGORY:   P=0.78 R=0.82 F1=0.80  ✅ Acceptable
```

If `I-DATE_START` (inside date tokens) is low:
- Issue: Model struggles with multi-token dates
- Fix: Add more multi-word dates in different formats

---

## Data Quality Issues & Fixes

### Issue 1: Intent Imbalance
**Symptom**: 
- 500 samples of "get_sales_by_article"
- 100 samples of "get_payments"

**Fix**:
```python
# In data_generator.py, increase REPEAT_FACTOR or add custom templates
REPEAT_FACTOR = 400  # Was 200

# Also add more templates for underrepresented intents
intents["get_payments"]["templates"].extend([
    "montrez moi la répartition des paiements par {category} du {start} au {end}",
    "quels sont les paiements ventilés par {category} entre {start} et {end}",
    # Add 5-10 more similar variations
])
```

### Issue 2: NER Tag Imbalance
**Symptom**:
- "B-DATE_START": 5,000 occurrences (95%)
- "I-CATEGORY": 50 occurrences (0.1%)

**Fix**:
```python
# In data_generator.py, adjust entity generation
# Add more multi-word categories and dates
categories = {
    "client": "client",
    "mode de paiement": "mode de paiement",      # Multi-word
    "année fiscale": "année fiscale",            # Multi-word
    "tous": "tous"
}

# Extend template placeholders
"répartition des paiements par {category} avec détails du {category}"
```

### Issue 3: Low Entity Coverage
**Symptom**: DATE_END appears in only 30% of samples

**Fix**:
- Ensure every template uses all available placeholders
- Add mandatory entity injection:
  ```python
  # Make sure BOTH start and end dates always appear
  for template in templates:
      if "{start}" not in template or "{end}" not in template:
          print(f"⚠️ Template missing date: {template}")
  ```

### Issue 4: Date Range Too Narrow
**Symptom**: All dates are in early 2023

**Fix**:
```python
# In data_generator.py, confirm date range is diverse
def random_date():
    start = datetime(2023, 1, 1)   # Start of range
    end = datetime(2025, 12, 31)   # End of range
    # Check that dates cover full range
    
# Verify distribution:
# Run test_model.py and check if model confuses years
```

---

## Step-by-Step Debugging Workflow

### Step 1: Run Initial Tests
```bash
python test_model.py --model_dir ./xlm_roberta_model
```

**Look for RED FLAGS in output:**
- Intent Accuracy < 70%? → Dataset issue or need more training
- NER F1 < 60%? → Data quality or tag alignment problem
- High variance between classes? → Class imbalance

### Step 2: Analyze Dataset Issues
```bash
python analyze_data.py --output_report analysis.json
```

**Check for:**
- Intent samples < 100 each?
- NER tag with < 50 examples?
- Entity coverage < 50%?
- Missing intents in test set?

### Step 3: Test-Sample Analysis
```bash
python test_model.py --model_dir ./xlm_roberta_model --interactive
```

**Enter real-world test cases:**
```
>>> afficher les ventes par article du 2024-01-01 au 2024-12-31
Intent: get_sales_by_article ✅
Tags: DATE_START=2024-01-01, DATE_END=2024-12-31 ✅

>>> stat client janvier
Intent: get_sales_by_client ⚠️ Maybe should be uncertain?
Tags: No entities found ❌ DATE not extracted!
```

### Step 4: Identify Specific Issues

**If dates aren't extracted:**
- Check date format in generator matches your test input
- Verify regex patterns are correct
- Add more date format variations

**If categories are missed:**
- Ensure category words appear clearly in text
- Check if model confused category with other entities
- Add more category examples

**If intent is wrong:**
- Check if templates are too similar (confusion)
- Increase training epochs
- Add negative examples where it's obvious what it's NOT

### Step 5: Fix & Re-run

#### Option A: Increase Dataset Size (Fastest)
```bash
# 1. Edit data_generator.py
REPEAT_FACTOR = 400  # Was 200, doubled

# 2. Regenerate dataset
python data_generator.py

# 3. Retrain
python train.py --num_epochs 5 --output_dir ./xlm_roberta_model_v2

# 4. Test
python test_model.py --model_dir ./xlm_roberta_model_v2
```

#### Option B: Improve Data Quality (Best Long-term)
```python
# 1. Add more diverse templates
intents["get_sales_by_article"]["templates"] = [
    "afficher les ventes par article du {start} au {end}",
    "pour l'article, quelles sont les ventes du {start} au {end}",
    "liste les ventes de produits entre {start} et {end}",
    "tableau de ventes articles {start} à {end}",
    # Add 5-10 more realistic variations
]

# 2. Improve synonyms
"synonyms": {
    "ventes": ["chiffre d'affaires", "revenus", "CA", "ventes brutes", "chiffre"],
    "article": ["produit", "item", "marchandise", "produit", "bien"],
}

# 3. Add edge cases
# "ventes par article du 01/01/2024 au 31/12/2024"  (different date format)
# "donne moi les ventes par article from 2024-01-01 to 2024-12-31" (English)
# "ventes  article  2024-01-01  2024-12-31" (minimal, informal)
```

---

## Performance Targets

### Minimum Viable Quality
- Intent Accuracy: 75%+
- NER F1: 65%+
- All intents: >= 80% coverage

✅ **OK for**: Internal testing, demos, initial deployment

### Production-Ready Quality
- Intent Accuracy: 85%+
- NER F1: 80%+
- All intents: >= 90% coverage per entity type
- Zero critical entities missed

✅ **Ready for**: Customer-facing applications

### State-of-the-Art Quality
- Intent Accuracy: 92%+
- NER F1: 88%+
- Handles edge cases, typos, informal language

✅ **Ready for**: Enterprise systems, mission-critical

---

## Dataset Size Recommendations

Based on intent count (5 intents in your case):

| Dataset Size | Expected Performance | Use Case |
|---|---|---|
| 100-200 samples | 50-65% accuracy | Proof of concept |
| 500-1000 samples | 75-80% accuracy | Early testing |
| 1000-3000 samples | 80-90% accuracy | Production ready |
| 3000-10000 samples | 90-95% accuracy | Robust production |
| 10000+ samples | 95%+ accuracy | Industry-leading |

**Your current setup**: Generating ~200 samples × templates appears to be ~1000-5000 samples, which should give **80-90% accuracy**.

---

## Common Issues & Causes

### Issue: Same performance, no improvement after more data
**Causes:**
1. Data is synthetic and unrealistic
2. Template patterns not diverse enough  
3. Model reaches convergence (try lower learning rate)
4. Architectural mismatch (increase hidden size in classifiers)

**Fix:**
```python
# Make dropout stronger to reduce overfitting
class XLMRobertaForIntentAndNER(nn.Module):
    def __init__(self, model_name, num_labels_intent, num_labels_ner, dropout_rate=0.3):  # Was 0.1
        # ... rest of code
```

### Issue: Perfect training accuracy but low test accuracy
**Causes:**
1. Overfitting (too little data or no regularization)
2. Train/test distribution mismatch
3. Test data has different format than training

**Fix:**
```python
# In train.py, enable early stopping and reduce learning rate
training_args = TrainingArguments(
    learning_rate=1e-5,  # Reduce from 2e-5
    weight_decay=0.1,    # Increase from 0.01
    # ... rest
)
```

### Issue: Specific intent always confused with another
**Cause**: Templates too similar

**Example:**
- "get_sales_by_article" templates start with "ventes par article"
- "get_purchases_by_article" templates start with "achats par article"
- Model confuses because similarity is too high

**Fix:**
```python
# Make intents more distinct in initial template words
"get_sales_by_article": {
    "templates": [
        "AFFICHER chiffre d'affaires par article du {start} au {end}",  # Different opener
        "STATISTIQUES ventes par article entre {start} et {end}",
    ]
},
"get_purchases_by_article": {
    "templates": [
        "MONTREZ les achats par article du {start} au {end}",  # Different opener
        "COMMANDES articles entre {start} et {end}",
    ]
}
```

---

## Real-World Testing Checklist

Before deploying to production, manually test these cases:

```
□ English dates: "from 2024-01-01 to 2024-12-31"
□ French dates: "du 2024-01-01 au 2024-12-31"
□ Short dates: "01/01/24"
□ Text dates: "January to December"
□ Abbreviations: "client" vs "clt", "articles" vs "arts"
□ Typos: "ventes" vs "venntes"
□ Case variations: "VENTES" vs "ventes"
□ Extra spaces: "ventes  par  article"
□ Missing year: "du 01-01 au 31-12" (should still work)
□ Real intent mix: "donne-moi les ventes par article du 2024-01-01 au 2024-03-31"
```

---

## Next Steps

1. **Run test script** → `python test_model.py --model_dir ./xlm_roberta_model`
2. **Analyze results** → Review metrics in output
3. **Check data quality** → `python analyze_data.py`
4. **Interactive testing** → Test real phrases with `--interactive` mode
5. **Make improvements** → Modify data_generator.py based on findings
6. **Retrain** → `python train.py --num_epochs 5`
7. **Compare** → Test new model and compare metrics
8. **Iterate** → Repeat until metrics meet your targets

**Expected timeline**: 
- Small fixes: 30-60 minutes
- Data augmentation: 1-2 hours  
- Major dataset overhaul: 2-4 hours
