"""
Line Item Matcher for Receipt OCR Evaluation

Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
optimal 1:1 alignment between GT and predicted line items, then scores
individual fields within matched pairs.

Also handles tax items with the same approach.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from normalizers import normalize_text, normalize_numeric, is_null
from metrics import (
    token_f1_score, levenshtein_similarity, 
    embedding_similarity, numeric_exact_match
)
from schema import LINE_ITEM_FIELDS, TAX_ITEM_FIELDS, LineItemFieldDef


@dataclass
class ItemMatchResult:
    """Result of matching and scoring a single pair of items."""
    gt_index: int
    pred_index: int
    gt_item: Dict[str, Any]
    pred_item: Dict[str, Any]
    
    # Per-field scores
    field_scores: Dict[str, float] = field(default_factory=dict)
    
    # Overall item similarity (weighted average of field scores)
    overall_score: float = 0.0
    
    # Matching similarity used for alignment
    alignment_score: float = 0.0


@dataclass 
class LineItemEvalResult:
    """Aggregate result for line item evaluation."""
    # Count metrics
    gt_count: int = 0
    pred_count: int = 0
    count_accuracy: float = 0.0
    
    # Matching metrics
    matched_pairs: List[ItemMatchResult] = field(default_factory=list)
    unmatched_gt: List[int] = field(default_factory=list)     # False negatives (missed)
    unmatched_pred: List[int] = field(default_factory=list)   # False positives (hallucinated)
    
    # Item-level precision/recall/F1
    item_precision: float = 0.0
    item_recall: float = 0.0
    item_f1: float = 0.0
    
    # Per-field averages (across matched pairs)
    avg_field_scores: Dict[str, float] = field(default_factory=dict)
    
    # Overall score
    overall_score: float = 0.0


def _compute_item_similarity(gt_item: Dict, pred_item: Dict, 
                              field_defs: Dict[str, LineItemFieldDef]) -> float:
    """
    Compute similarity between two items for alignment.
    Used to build the cost matrix for Hungarian matching.
    
    Weights description heavily (0.6) and amount (0.4) for alignment.
    """
    desc_score = 0.0
    amount_score = 0.0
    
    # Description similarity
    gt_desc = gt_item.get("description", "")
    pred_desc = pred_item.get("description", "")
    
    if gt_desc and pred_desc:
        # Use token F1 for alignment (fast, doesn't need embeddings)
        tf1 = token_f1_score(str(gt_desc), str(pred_desc))
        lev = levenshtein_similarity(str(gt_desc), str(pred_desc))
        desc_score = max(tf1, lev)
    
    # Amount similarity
    gt_amount = normalize_numeric(gt_item.get("amount"))
    pred_amount = normalize_numeric(pred_item.get("amount"))
    
    if gt_amount is not None and pred_amount is not None:
        amount_score = numeric_exact_match(gt_amount, pred_amount, tolerance=0.02)
        # If not exact, give partial credit based on closeness
        if amount_score == 0.0 and gt_amount != 0:
            rel_err = abs(gt_amount - pred_amount) / abs(gt_amount)
            if rel_err < 0.5:
                amount_score = 1.0 - rel_err
    elif gt_amount is None and pred_amount is None:
        amount_score = 1.0
    
    # Weighted combination for alignment
    return 0.6 * desc_score + 0.4 * amount_score


def _score_matched_pair(gt_item: Dict, pred_item: Dict,
                         field_defs: Dict[str, LineItemFieldDef]) -> Dict[str, float]:
    """
    Score individual fields within a matched pair.
    Returns dict of field_name -> score (0.0 to 1.0).
    """
    scores = {}
    
    for fname, fdef in field_defs.items():
        gt_val = gt_item.get(fname)
        pred_val = pred_item.get(fname)
        
        # Both null → perfect agreement
        if is_null(gt_val) and is_null(pred_val):
            scores[fname] = 1.0
            continue
        
        # One null → score 0
        if is_null(gt_val) or is_null(pred_val):
            scores[fname] = 0.0
            continue
        
        if fdef.field_type == "numeric":
            gt_num = normalize_numeric(gt_val)
            pred_num = normalize_numeric(pred_val)
            if gt_num is not None and pred_num is not None:
                scores[fname] = numeric_exact_match(gt_num, pred_num, tolerance=0.02)
            else:
                scores[fname] = 0.0
        
        elif fdef.field_type in ("short_string", "medium_string"):
            gt_str = str(gt_val)
            pred_str = str(pred_val)
            
            # Try embedding first, fall back to token F1
            emb = embedding_similarity(gt_str, pred_str)
            tf1 = token_f1_score(gt_str, pred_str)
            lev = levenshtein_similarity(gt_str, pred_str)
            
            if emb is not None:
                scores[fname] = max(emb, tf1, lev)
            else:
                scores[fname] = max(tf1, lev)
        else:
            # Default: exact match
            gt_norm = normalize_text(str(gt_val))
            pred_norm = normalize_text(str(pred_val))
            scores[fname] = 1.0 if gt_norm == pred_norm else 0.0
    
    return scores


def evaluate_line_items(gt_items: List[Dict], pred_items: List[Dict],
                        field_defs: Optional[Dict[str, LineItemFieldDef]] = None,
                        similarity_threshold: float = 0.3) -> LineItemEvalResult:
    """
    Evaluate line items using Hungarian matching.
    
    Args:
        gt_items: Ground truth line items
        pred_items: Predicted line items
        field_defs: Field definitions (defaults to LINE_ITEM_FIELDS)
        similarity_threshold: Minimum similarity to consider a match valid
    
    Returns:
        LineItemEvalResult with all metrics
    """
    if field_defs is None:
        field_defs = LINE_ITEM_FIELDS
    
    result = LineItemEvalResult()
    
    # Handle null/empty inputs
    gt_items = gt_items or []
    pred_items = pred_items or []
    
    result.gt_count = len(gt_items)
    result.pred_count = len(pred_items)
    
    # Count accuracy
    if result.gt_count == 0 and result.pred_count == 0:
        result.count_accuracy = 1.0
        result.item_precision = 1.0
        result.item_recall = 1.0
        result.item_f1 = 1.0
        result.overall_score = 1.0
        return result
    
    if result.gt_count == 0:
        # All predictions are hallucinations
        result.count_accuracy = 0.0
        result.unmatched_pred = list(range(result.pred_count))
        result.item_precision = 0.0
        result.item_recall = 1.0  # Vacuously true
        result.item_f1 = 0.0
        result.overall_score = 0.0
        return result
    
    if result.pred_count == 0:
        # All GT items missed
        result.count_accuracy = 0.0
        result.unmatched_gt = list(range(result.gt_count))
        result.item_precision = 1.0  # Vacuously true
        result.item_recall = 0.0
        result.item_f1 = 0.0
        result.overall_score = 0.0
        return result
    
    result.count_accuracy = 1.0 - abs(result.gt_count - result.pred_count) / max(result.gt_count, result.pred_count)
    
    # Build similarity matrix
    n_gt = len(gt_items)
    n_pred = len(pred_items)
    sim_matrix = np.zeros((n_gt, n_pred))
    
    for i, gt_item in enumerate(gt_items):
        for j, pred_item in enumerate(pred_items):
            sim_matrix[i, j] = _compute_item_similarity(gt_item, pred_item, field_defs)
    
    # Hungarian algorithm (minimize cost, so we negate similarity)
    try:
        from scipy.optimize import linear_sum_assignment
        
        # Pad matrix to be square if needed
        max_dim = max(n_gt, n_pred)
        cost_matrix = np.zeros((max_dim, max_dim))
        cost_matrix[:n_gt, :n_pred] = -sim_matrix  # Negate for minimization
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    except ImportError:
        # Fallback: greedy matching
        row_indices, col_indices = _greedy_match(sim_matrix)
    
    # Process matches
    matched_gt = set()
    matched_pred = set()
    
    for row, col in zip(row_indices, col_indices):
        if row >= n_gt or col >= n_pred:
            continue  # Padding entries
        
        alignment_score = sim_matrix[row, col]
        
        if alignment_score < similarity_threshold:
            continue  # Too dissimilar to match
        
        # Score the matched pair
        field_scores = _score_matched_pair(gt_items[row], pred_items[col], field_defs)
        
        # Weighted overall score for this pair
        total_weight = sum(field_defs[f].weight for f in field_scores if f in field_defs)
        weighted_sum = sum(
            field_scores[f] * field_defs[f].weight 
            for f in field_scores if f in field_defs
        )
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        match_result = ItemMatchResult(
            gt_index=row,
            pred_index=col,
            gt_item=gt_items[row],
            pred_item=pred_items[col],
            field_scores=field_scores,
            overall_score=overall,
            alignment_score=alignment_score,
        )
        
        result.matched_pairs.append(match_result)
        matched_gt.add(row)
        matched_pred.add(col)
    
    # Unmatched items
    result.unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    result.unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]
    
    # Item-level P/R/F1
    n_matched = len(result.matched_pairs)
    result.item_precision = n_matched / n_pred if n_pred > 0 else 0.0
    result.item_recall = n_matched / n_gt if n_gt > 0 else 0.0
    
    if result.item_precision + result.item_recall > 0:
        result.item_f1 = (
            2 * result.item_precision * result.item_recall / 
            (result.item_precision + result.item_recall)
        )
    
    # Per-field averages across matched pairs
    if result.matched_pairs:
        all_fields = set()
        for mp in result.matched_pairs:
            all_fields.update(mp.field_scores.keys())
        
        for fname in all_fields:
            scores = [mp.field_scores.get(fname, 0.0) for mp in result.matched_pairs]
            result.avg_field_scores[fname] = sum(scores) / len(scores) if scores else 0.0
    
    # Overall score: combine item-level F1 with average field accuracy
    if result.matched_pairs:
        avg_pair_score = sum(mp.overall_score for mp in result.matched_pairs) / len(result.matched_pairs)
        # Penalize for missed/hallucinated items
        result.overall_score = result.item_f1 * avg_pair_score
    else:
        result.overall_score = 0.0
    
    return result


def evaluate_tax_items(gt_taxes: List[Dict], pred_taxes: List[Dict]) -> LineItemEvalResult:
    """Evaluate tax items using same approach as line items."""
    return evaluate_line_items(gt_taxes, pred_taxes, field_defs=TAX_ITEM_FIELDS)


def _greedy_match(sim_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Greedy matching fallback when scipy is not available.
    Iteratively picks the highest-similarity unmatched pair.
    """
    n_gt, n_pred = sim_matrix.shape
    rows = []
    cols = []
    used_gt = set()
    used_pred = set()
    
    # Flatten and sort by similarity descending
    pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)
    
    for sim, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        rows.append(i)
        cols.append(j)
        used_gt.add(i)
        used_pred.add(j)
    
    return rows, cols
