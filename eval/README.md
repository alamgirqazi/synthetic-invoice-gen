# Receipt OCR Evaluation Framework

A benchmark-style evaluation framework for comparing OCR model outputs on receipt/invoice documents designed specifically for structured document extraction.

```bash
uvicorn api:app --port 8000
```

## Quick Start

```bash
# Basic run
python eval.py --gt ground_truth.json --pred model_a.json

# With JSON export
python eval.py --gt ground_truth.json --pred model_a.json -o results/

# Fast mode (no sentence-transformers needed)
python eval.py --gt ground_truth.json --pred model_a.json --no-embeddings

# Verbose (see per-document, per-field details)
python eval.py --gt ground_truth.json --pred model_a.json -v

# Compare multiple models
python eval.py --gt ground_truth.json --pred model_a.json model_b.json model_c.json --compare

```



```bash
# Install dependencies
pip install numpy scipy
pip install sentence-transformers  # Optional: enables semantic similarity

# Evaluate a single model
python -m receipt_eval --gt ground_truth.json --pred model_output.json

# Compare multiple models
python -m receipt_eval --gt ground_truth.json \
    --pred olmocr2.json nanonets.json dots.json --compare

# Fast mode (no embeddings, pure lexical matching)
python -m receipt_eval --gt ground_truth.json --pred output.json --no-embeddings

# Only evaluate critical fields (amounts, dates, supplier)
python -m receipt_eval --gt ground_truth.json --pred output.json --tiers 1

# Export detailed JSON report
python -m receipt_eval --gt ground_truth.json --pred output.json -o results/
```

## Input Format

Both ground truth and prediction files are JSON arrays of documents. Documents are matched by `file_name`.

```json
[
  {
    "file_name": "receipt_001.pdf",
    "supplier_name": "Cork Chamber",
    "document_date": "2024-01-05",
    "document_currency": "EUR",
    "document_total_amount": 390.00,
    "document_total_net": 390.00,
    "document_total_tax": 0.00,
    "document_category": "services",
    "subcategory": "membership",
    "invoice_number": "78058",
    "line_items": [
      {
        "description": "Annual Membership",
        "quantity": 1,
        "unit_price": 390.00,
        "amount": 390.00
      }
    ],
    "notes": "Terms: 30 Days Nett.",
    ...
  }
]
```

## How It Works

### Field Types & Evaluation Strategy

| Field Type | Fields | Method | Why |
|---|---|---|---|
| **numeric** | amounts, tax, quantities | Exact match ±0.01 tolerance | Accounting precision |
| **date** | document_date, due_date | Parse → compare date objects | Format-agnostic (DD/MM/YYYY = YYYY-MM-DD) |
| **short_string** | supplier_name, category, currency | Normalized exact → Levenshtein → embedding similarity | Handles "FreeNow" vs "FREENOW Ireland Ltd." |
| **medium_string** | addresses, descriptions | Token F1 + embedding similarity | Order-independent, catches key tokens |
| **long_text** | notes | Token F1 + key info extraction | Structured nuggets (VAT numbers, IBANs, emails) |
| **line_items** | line_items array | Hungarian matching + per-field scoring | Optimal alignment of nested arrays |

### Null Handling (4-cell confusion matrix per field)

|  | GT has value | GT is null |
|---|---|---|
| **Pred has value** | True Positive (compare) | **Hallucination** (model invented data) |
| **Pred is null** | **Miss** (model failed to extract) | True Negative (both agree absent) |

### Grading Scale

Each field comparison produces a **grade**:

| Grade | Meaning | Similarity Threshold |
|---|---|---|
| `exact` | Essentially identical | ≥ 0.90 |
| `acceptable` | Same concept, different wording | ≥ 0.75 |
| `partial` | Related but not quite right | ≥ 0.55 |
| `wrong` | Completely off | < 0.55 |
| `null_match` | Both GT and pred are null | N/A |
| `hallucination` | GT is null, pred invented a value | N/A |
| `miss` | GT has value, pred is null | N/A |

### Field Tiers & Weights

Fields are organized into tiers with different weights:

- **Tier 1 (Critical, 60%):** supplier_name, document_date, currency, total_amount, total_net, total_tax, line_items
- **Tier 2 (Important, 25%):** category, subcategory, invoice_number, document_type, taxes, VAT numbers
- **Tier 3 (Supplementary, 15%):** addresses, phone, email, website, customer info, notes

### Line Item Evaluation (Hungarian Matching)

1. **Build similarity matrix** — For each (GT item, predicted item) pair, compute composite similarity (60% description match + 40% amount match)
2. **Optimal 1:1 alignment** — Hungarian algorithm finds the best pairing
3. **Score matched pairs** — Per-field comparison within each aligned pair
4. **Handle unmatched** — GT items with no match = false negatives; predicted items with no match = false positives
5. **Compute P/R/F1** — Item-level precision, recall, F1-score
