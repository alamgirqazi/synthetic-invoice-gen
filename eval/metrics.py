"""
Metrics for Receipt OCR Evaluation

All metric functions used by the comparators.
Organized by metric type: string, numeric, semantic.
"""

import re
from typing import Optional, List, Set
from normalizers import normalize_text, tokenize


# ============================================================
# STRING METRICS
# ============================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def levenshtein_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """
    Normalized Levenshtein similarity (0.0 to 1.0).
    1.0 = identical strings.
    """
    if s1 is None and s2 is None:
        return 1.0
    if s1 is None or s2 is None:
        return 0.0
    
    s1_norm = normalize_text(s1) or ""
    s2_norm = normalize_text(s2) or ""
    
    if s1_norm == s2_norm:
        return 1.0
    
    max_len = max(len(s1_norm), len(s2_norm))
    if max_len == 0:
        return 1.0
    
    dist = levenshtein_distance(s1_norm, s2_norm)
    return 1.0 - (dist / max_len)


def exact_match(s1: Optional[str], s2: Optional[str]) -> float:
    """
    Binary exact match after normalization.
    Returns 1.0 if match, 0.0 otherwise.
    """
    n1 = normalize_text(s1)
    n2 = normalize_text(s2)
    
    if n1 is None and n2 is None:
        return 1.0
    if n1 is None or n2 is None:
        return 0.0
    
    return 1.0 if n1 == n2 else 0.0


def token_f1(s1: Optional[str], s2: Optional[str]) -> dict:
    """
    Token-level precision, recall, F1.
    Treats strings as bags of words after normalization.
    
    Returns:
        dict with keys: precision, recall, f1
    """
    tokens1 = set(tokenize(s1))
    tokens2 = set(tokenize(s2))
    
    if not tokens1 and not tokens2:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not tokens1 or not tokens2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    common = tokens1 & tokens2
    
    precision = len(common) / len(tokens2) if tokens2 else 0.0  # How much of pred is correct
    recall = len(common) / len(tokens1) if tokens1 else 0.0     # How much of GT was found
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def token_f1_score(s1: Optional[str], s2: Optional[str]) -> float:
    """Convenience: returns just the F1 score."""
    return token_f1(s1, s2)["f1"]


# ============================================================
# NUMERIC METRICS
# ============================================================

def numeric_exact_match(val1: Optional[float], val2: Optional[float], 
                        tolerance: float = 0.01) -> float:
    """
    Numeric match with tolerance.
    Returns 1.0 if values match within tolerance, 0.0 otherwise.
    """
    if val1 is None and val2 is None:
        return 1.0
    if val1 is None or val2 is None:
        return 0.0
    
    return 1.0 if abs(val1 - val2) <= tolerance else 0.0


def numeric_relative_error(gt: Optional[float], pred: Optional[float]) -> Optional[float]:
    """
    Relative error: |gt - pred| / |gt|
    Returns None if GT is zero or null.
    """
    if gt is None or pred is None:
        return None
    if gt == 0:
        return 0.0 if pred == 0 else None
    return abs(gt - pred) / abs(gt)


def numeric_absolute_error(gt: Optional[float], pred: Optional[float]) -> Optional[float]:
    """Absolute error: |gt - pred|"""
    if gt is None or pred is None:
        return None
    return abs(gt - pred)


# ============================================================
# KEY INFORMATION EXTRACTION (for notes field)
# ============================================================

def extract_key_info(text: Optional[str]) -> dict:
    """
    Extract structured nuggets from free text.
    Returns dict of pattern types found.
    """
    if not text:
        return {"emails": set(), "urls": set(), "vat_numbers": set(), 
                "amounts": set(), "dates": set(), "phones": set()}
    
    info = {
        "emails": set(re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)),
        "urls": set(re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', text)),
        "vat_numbers": set(re.findall(r'(?:VAT|IE|GB|DE|FR|NL)\s*[\w\d]{6,}', text, re.IGNORECASE)),
        "amounts": set(re.findall(r'[€$£]\s*[\d,]+\.?\d*|\d+[.,]\d{2}\s*(?:EUR|USD|GBP)', text)),
        "dates": set(re.findall(r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}', text)),
        "phones": set(re.findall(r'[\+]?[\d\s\-()]{7,}', text)),
    }
    
    return info


def key_info_recall(gt_text: Optional[str], pred_text: Optional[str]) -> float:
    """
    What fraction of key information in GT was captured in prediction.
    """
    gt_info = extract_key_info(gt_text)
    pred_info = extract_key_info(pred_text)
    
    total_gt_items = 0
    found_items = 0
    
    for key in gt_info:
        gt_set = gt_info[key]
        pred_set = pred_info[key]
        total_gt_items += len(gt_set)
        
        for item in gt_set:
            # Normalize for comparison
            item_norm = normalize_text(item) or ""
            for pred_item in pred_set:
                pred_norm = normalize_text(pred_item) or ""
                if item_norm == pred_norm or item_norm in pred_norm or pred_norm in item_norm:
                    found_items += 1
                    break
    
    if total_gt_items == 0:
        return 1.0  # No key info to find
    
    return found_items / total_gt_items


# ============================================================
# EMBEDDING SIMILARITY (lazy-loaded)
# ============================================================

_embedding_model = None

def _get_embedding_model():
    """Lazy-load sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            print("  WARNING: sentence-transformers not installed. Falling back to token F1.")
            print("  Install with: pip install sentence-transformers")
            _embedding_model = False  # Mark as unavailable
    return _embedding_model


def embedding_similarity(text1: Optional[str], text2: Optional[str]) -> Optional[float]:
    """
    Compute cosine similarity using sentence embeddings.
    Returns None if sentence-transformers is not installed.
    Falls back gracefully.
    """
    if text1 is None or text2 is None:
        return None
    
    model = _get_embedding_model()
    if model is False:
        return None  # Not available
    
    import numpy as np
    
    t1 = normalize_text(text1) or ""
    t2 = normalize_text(text2) or ""
    
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0
    
    embeddings = model.encode([t1, t2], convert_to_numpy=True)
    
    # Cosine similarity
    dot = np.dot(embeddings[0], embeddings[1])
    norm = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    
    if norm == 0:
        return 0.0
    
    return float(dot / norm)


def batch_embedding_similarity(pairs: List[tuple]) -> List[Optional[float]]:
    """
    Batch compute embedding similarities for efficiency.
    pairs: list of (text1, text2) tuples
    Returns list of similarity scores.
    """
    model = _get_embedding_model()
    if model is False:
        return [None] * len(pairs)
    
    import numpy as np
    
    # Collect all unique texts
    all_texts = []
    text_to_idx = {}
    
    for t1, t2 in pairs:
        for t in [normalize_text(t1) or "", normalize_text(t2) or ""]:
            if t not in text_to_idx:
                text_to_idx[t] = len(all_texts)
                all_texts.append(t)
    
    if not all_texts:
        return [None] * len(pairs)
    
    # Encode all at once
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Compute similarities
    results = []
    for t1, t2 in pairs:
        n1 = normalize_text(t1) or ""
        n2 = normalize_text(t2) or ""
        
        if not n1 or not n2:
            results.append(0.0)
            continue
        if n1 == n2:
            results.append(1.0)
            continue
        
        e1 = embeddings[text_to_idx[n1]]
        e2 = embeddings[text_to_idx[n2]]
        
        dot = np.dot(e1, e2)
        norm = np.linalg.norm(e1) * np.linalg.norm(e2)
        results.append(float(dot / norm) if norm > 0 else 0.0)
    
    return results
