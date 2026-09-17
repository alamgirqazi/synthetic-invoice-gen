"""
Comparators for Receipt OCR Evaluation

Each field type has its own comparator that returns a standardized result dict.
The schema dispatches to the right comparator based on field_type.

FIXES:
  - Short strings: identity-critical fields (supplier_name, invoice_number)
    now weight exact/Levenshtein higher than embeddings to avoid false
    positives like "Cork Chamber" ≈ "Cork City Council".
  - Category fields (field_type="category"): uses exact match on normalized
    canonical form instead of fuzzy similarity.
  - Date comparison: DD/MM swaps now scored as "partial" with 0.5 similarity
    instead of being treated as completely wrong.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from normalizers import (
    normalize_text, normalize_numeric, normalize_date, 
    normalize_time, normalize_currency, normalize_text_aggressive,
    is_null
)
from metrics import (
    exact_match, levenshtein_similarity, token_f1, token_f1_score,
    numeric_exact_match, numeric_relative_error, numeric_absolute_error,
    embedding_similarity, key_info_recall
)
from schema import FieldDefinition


@dataclass
class FieldResult:
    """Result of comparing a single field between GT and prediction."""
    field_name: str
    field_type: str
    tier: int
    weight: float
    
    # Values
    gt_value: Any = None
    pred_value: Any = None
    gt_is_null: bool = False
    pred_is_null: bool = False
    
    # Null status classification
    # true_positive:  both have value, can compare
    # true_negative:  both null, agreement
    # false_positive: GT null, pred has value (hallucination)
    # false_negative: GT has value, pred is null (miss)
    null_status: str = ""  # tp, tn, fp, fn
    
    # Scores (0.0 to 1.0, higher is better)
    exact_match_score: float = 0.0
    similarity_score: float = 0.0      # Primary similarity (method depends on type)
    embedding_score: Optional[float] = None  # Embedding similarity if available
    
    # Grade based on thresholds
    grade: str = ""  # exact, acceptable, partial, wrong, null_match, hallucination, miss
    
    # Type-specific details
    details: Dict[str, Any] = field(default_factory=dict)


def compare_field(field_def: FieldDefinition, gt_value: Any, pred_value: Any) -> FieldResult:
    """
    Compare a single field between GT and prediction.
    Dispatches to the appropriate comparator based on field_type.
    """
    result = FieldResult(
        field_name=field_def.name,
        field_type=field_def.field_type,
        tier=field_def.tier,
        weight=field_def.weight,
        gt_value=gt_value,
        pred_value=pred_value,
        gt_is_null=is_null(gt_value),
        pred_is_null=is_null(pred_value),
    )
    
    # Classify null status
    if result.gt_is_null and result.pred_is_null:
        result.null_status = "tn"
        result.exact_match_score = 1.0
        result.similarity_score = 1.0
        result.grade = "null_match"
        return result
    elif result.gt_is_null and not result.pred_is_null:
        result.null_status = "fp"
        result.exact_match_score = 0.0
        result.similarity_score = 0.0
        result.grade = "hallucination"
        return result
    elif not result.gt_is_null and result.pred_is_null:
        result.null_status = "fn"
        result.exact_match_score = 0.0
        result.similarity_score = 0.0
        result.grade = "miss"
        return result
    else:
        result.null_status = "tp"
    
    # Dispatch to type-specific comparator
    comparators = {
        "numeric": _compare_numeric,
        "date": _compare_date,
        "time": _compare_time,
        "short_string": _compare_short_string,
        "medium_string": _compare_medium_string,
        "long_text": _compare_long_text,
        "category": _compare_category,
    }
    
    comparator = comparators.get(field_def.field_type)
    if comparator:
        comparator(result, field_def)
    else:
        # Fallback: treat as short_string
        _compare_short_string(result, field_def)
    
    return result


# ============================================================
# TYPE-SPECIFIC COMPARATORS
# ============================================================

def _compare_numeric(result: FieldResult, field_def: FieldDefinition):
    """Compare numeric values (amounts, quantities)."""
    gt_num = normalize_numeric(result.gt_value)
    pred_num = normalize_numeric(result.pred_value)
    
    result.details["gt_parsed"] = gt_num
    result.details["pred_parsed"] = pred_num
    
    if gt_num is None or pred_num is None:
        result.exact_match_score = 0.0
        result.similarity_score = 0.0
        result.grade = "wrong"
        result.details["parse_error"] = True
        return
    
    tol = field_def.numeric_tolerance
    result.exact_match_score = numeric_exact_match(gt_num, pred_num, tol)
    
    abs_err = numeric_absolute_error(gt_num, pred_num)
    rel_err = numeric_relative_error(gt_num, pred_num)
    
    result.details["absolute_error"] = abs_err
    result.details["relative_error"] = rel_err
    
    if result.exact_match_score == 1.0:
        result.similarity_score = 1.0
        result.grade = "exact"
    else:
        # Provide a similarity based on relative error
        if rel_err is not None and rel_err < 1.0:
            result.similarity_score = 1.0 - rel_err
        else:
            result.similarity_score = 0.0
        
        # Grade based on closeness
        if abs_err is not None and abs_err <= 0.05:
            result.grade = "acceptable"  # Rounding error
        elif rel_err is not None and rel_err < 0.05:
            result.grade = "acceptable"
        elif rel_err is not None and rel_err < 0.20:
            result.grade = "partial"
        else:
            result.grade = "wrong"


def _compare_date(result: FieldResult, field_def: FieldDefinition):
    """
    Compare date values (format-agnostic).
    
    FIX: DD/MM swaps are now detected and scored as "partial" with 0.5
    similarity instead of getting a raw component score (which could be
    0.33 if only the year matched). This better reflects that the model
    extracted the right digits but misinterpreted the format.
    """
    gt_date = normalize_date(result.gt_value)
    pred_date = normalize_date(result.pred_value)
    
    result.details["gt_parsed"] = str(gt_date) if gt_date else None
    result.details["pred_parsed"] = str(pred_date) if pred_date else None
    
    if gt_date is None or pred_date is None:
        result.exact_match_score = 0.0
        result.similarity_score = 0.0
        result.grade = "wrong"
        result.details["parse_error"] = True
        return
    
    if gt_date == pred_date:
        result.exact_match_score = 1.0
        result.similarity_score = 1.0
        result.grade = "exact"
    else:
        result.exact_match_score = 0.0
        
        # Component matching: year, month, day
        year_match = gt_date.year == pred_date.year
        month_match = gt_date.month == pred_date.month
        day_match = gt_date.day == pred_date.day
        
        components_correct = sum([year_match, month_match, day_match])
        
        result.details["year_match"] = year_match
        result.details["month_match"] = month_match
        result.details["day_match"] = day_match
        result.details["day_diff"] = abs((gt_date - pred_date).days)
        
        # Detect DD/MM ↔ MM/DD swap
        is_dd_mm_swap = (
            gt_date.year == pred_date.year and
            gt_date.month == pred_date.day and
            gt_date.day == pred_date.month and
            gt_date.month != gt_date.day  # Not same value (e.g., 05/05)
        )
        
        if is_dd_mm_swap:
            result.details["dd_mm_swap"] = True
            # Give meaningful partial credit — the model got the digits right,
            # just interpreted the format differently
            result.similarity_score = 0.5
            result.grade = "partial"
        else:
            result.similarity_score = components_correct / 3.0
            
            if components_correct >= 2:
                result.grade = "partial"
            else:
                result.grade = "wrong"


def _compare_time(result: FieldResult, field_def: FieldDefinition):
    """Compare time values."""
    gt_time = normalize_time(result.gt_value)
    pred_time = normalize_time(result.pred_value)
    
    if gt_time is None or pred_time is None:
        result.exact_match_score = 0.0
        result.similarity_score = 0.0
        result.grade = "wrong"
        return
    
    if gt_time == pred_time:
        result.exact_match_score = 1.0
        result.similarity_score = 1.0
        result.grade = "exact"
    else:
        result.exact_match_score = 0.0
        result.similarity_score = levenshtein_similarity(gt_time, pred_time)
        result.grade = "partial" if result.similarity_score > 0.5 else "wrong"


def _compare_short_string(result: FieldResult, field_def: FieldDefinition):
    """
    Compare short strings (1-10 words).
    Strategy: normalized exact match → Levenshtein → embedding similarity.
    
    FIX: For identity-critical fields (marked with identity_critical=True
    in schema), embeddings are used as a secondary signal only. The primary
    score is max(levenshtein, token_f1). This prevents false high scores
    for semantically similar but factually different values like
    "Cork Chamber" vs "Cork City Council".
    """
    gt_str = str(result.gt_value) if result.gt_value is not None else ""
    pred_str = str(result.pred_value) if result.pred_value is not None else ""
    
    # 1. Normalized exact match
    em = exact_match(gt_str, pred_str)
    result.exact_match_score = em
    
    if em == 1.0:
        result.similarity_score = 1.0
        result.grade = "exact"
        return
    
    # 2. Aggressive exact match (for identifiers: remove all non-alphanumeric)
    agg_gt = normalize_text_aggressive(gt_str)
    agg_pred = normalize_text_aggressive(pred_str)
    if agg_gt and agg_pred and agg_gt == agg_pred:
        result.similarity_score = 1.0
        result.grade = "exact"
        result.details["matched_via"] = "aggressive_normalization"
        return
    
    # 3. Levenshtein similarity
    lev_sim = levenshtein_similarity(gt_str, pred_str)
    result.details["levenshtein_similarity"] = lev_sim
    
    # 4. Token F1
    tf1 = token_f1(gt_str, pred_str)
    result.details["token_f1"] = tf1
    
    # 5. Embedding similarity (if available)
    emb_sim = embedding_similarity(gt_str, pred_str)
    result.embedding_score = emb_sim
    
    # 6. Choose primary score based on whether field is identity-critical
    is_identity = getattr(field_def, "identity_critical", False)
    
    if is_identity:
        # Identity fields: lexical match is primary, embedding is secondary
        lexical_score = max(lev_sim, tf1["f1"])
        if emb_sim is not None:
            # Embedding can only boost slightly, not dominate
            # 80% lexical, 20% embedding
            result.similarity_score = 0.80 * lexical_score + 0.20 * emb_sim
        else:
            result.similarity_score = lexical_score
        result.details["primary_metric"] = "lexical_dominant (identity_critical)"
    else:
        # Non-identity fields: embedding is primary if available
        if emb_sim is not None:
            result.similarity_score = emb_sim
            result.details["primary_metric"] = "embedding"
        else:
            result.similarity_score = max(lev_sim, tf1["f1"])
            result.details["primary_metric"] = "levenshtein+token_f1"
    
    # Grade
    _assign_grade(result, field_def)


def _compare_category(result: FieldResult, field_def: FieldDefinition):
    """
    Compare category/classification fields.
    
    NEW: Categories come from a fixed taxonomy, so we use normalized
    exact match as the primary metric. Embedding similarity is used
    only to distinguish "partial" from "wrong" for near-miss categories
    (e.g., "food" vs "dining").
    """
    gt_str = str(result.gt_value) if result.gt_value is not None else ""
    pred_str = str(result.pred_value) if result.pred_value is not None else ""
    
    # Normalized exact match
    em = exact_match(gt_str, pred_str)
    result.exact_match_score = em
    
    if em == 1.0:
        result.similarity_score = 1.0
        result.grade = "exact"
        return
    
    # Aggressive match (e.g., "food & drink" vs "food_and_drink")
    agg_gt = normalize_text_aggressive(gt_str)
    agg_pred = normalize_text_aggressive(pred_str)
    if agg_gt and agg_pred and agg_gt == agg_pred:
        result.similarity_score = 1.0
        result.grade = "exact"
        result.details["matched_via"] = "aggressive_normalization"
        return
    
    # Not an exact match — check if it's a reasonable near-miss
    lev_sim = levenshtein_similarity(gt_str, pred_str)
    result.details["levenshtein_similarity"] = lev_sim
    
    emb_sim = embedding_similarity(gt_str, pred_str)
    result.embedding_score = emb_sim
    
    # For categories, exact match is king. Near-misses get partial credit
    # only if semantically very close.
    if emb_sim is not None and emb_sim >= 0.85:
        # Very semantically similar (e.g., "food" vs "food & beverage")
        result.similarity_score = 0.6  # Cap: still not the right category
        result.grade = "partial"
        result.details["primary_metric"] = "category_near_miss"
    elif lev_sim >= 0.8:
        # Likely a typo or formatting difference
        result.similarity_score = 0.7
        result.grade = "partial"
        result.details["primary_metric"] = "category_typo"
    else:
        # Wrong category entirely
        result.similarity_score = 0.0
        result.grade = "wrong"
        result.details["primary_metric"] = "category_mismatch"


def _compare_medium_string(result: FieldResult, field_def: FieldDefinition):
    """
    Compare medium strings (10-100 words, e.g. addresses, descriptions).
    Strategy: token F1 + embedding similarity.
    """
    gt_str = str(result.gt_value) if result.gt_value is not None else ""
    pred_str = str(result.pred_value) if result.pred_value is not None else ""
    
    # Exact match
    em = exact_match(gt_str, pred_str)
    result.exact_match_score = em
    
    if em == 1.0:
        result.similarity_score = 1.0
        result.grade = "exact"
        return
    
    # Token F1
    tf1 = token_f1(gt_str, pred_str)
    result.details["token_f1"] = tf1
    
    # Levenshtein
    lev_sim = levenshtein_similarity(gt_str, pred_str)
    result.details["levenshtein_similarity"] = lev_sim
    
    # Embedding similarity
    emb_sim = embedding_similarity(gt_str, pred_str)
    result.embedding_score = emb_sim
    
    # For medium strings, combine token F1 and embedding
    if emb_sim is not None:
        # Weighted: 60% token F1, 40% embedding
        result.similarity_score = 0.6 * tf1["f1"] + 0.4 * emb_sim
        result.details["primary_metric"] = "token_f1+embedding"
    else:
        result.similarity_score = tf1["f1"]
        result.details["primary_metric"] = "token_f1"
    
    _assign_grade(result, field_def)


def _compare_long_text(result: FieldResult, field_def: FieldDefinition):
    """
    Compare long text (100+ words, e.g. notes).
    Strategy: token F1 + key info extraction recall.
    """
    gt_str = str(result.gt_value) if result.gt_value is not None else ""
    pred_str = str(result.pred_value) if result.pred_value is not None else ""
    
    # Exact match
    em = exact_match(gt_str, pred_str)
    result.exact_match_score = em
    
    if em == 1.0:
        result.similarity_score = 1.0
        result.grade = "exact"
        return
    
    # Token F1
    tf1 = token_f1(gt_str, pred_str)
    result.details["token_f1"] = tf1
    
    # Key info recall (structured nuggets)
    ki_recall = key_info_recall(gt_str, pred_str)
    result.details["key_info_recall"] = ki_recall
    
    # Combine: 50% token F1, 50% key info recall
    result.similarity_score = 0.5 * tf1["f1"] + 0.5 * ki_recall
    result.details["primary_metric"] = "token_f1+key_info"
    
    _assign_grade(result, field_def)


# ============================================================
# GRADING
# ============================================================

def _assign_grade(result: FieldResult, field_def: FieldDefinition):
    """Assign a grade based on similarity score and field thresholds."""
    score = result.similarity_score
    
    if score >= field_def.threshold_exact:
        result.grade = "exact"
    elif score >= field_def.threshold_acceptable:
        result.grade = "acceptable"
    elif score >= field_def.threshold_partial:
        result.grade = "partial"
    else:
        result.grade = "wrong"