"""
Standard Metrics for Benchmark Comparability

Computes Entity-level F1 (SROIE/CORD standard) and ANLS (DocVQA standard)
from existing evaluation results, enabling direct comparison with published
results on established benchmarks.

FIX: True negatives (both GT and pred null) are excluded from all metric
calculations. They tell us nothing about extraction quality and previously
inflated ANLS scores.

Usage:
    from standard_metrics import compute_standard_metrics
    
    # After running your normal evaluation:
    std = compute_standard_metrics(report)
    print(f"Entity F1: {std['entity_f1']:.4f}")
    print(f"ANLS:      {std['anls']:.4f}")
"""

from typing import Dict, List, Any
from report import EvalReport, DocumentResult


def compute_entity_f1(report: EvalReport) -> Dict[str, float]:
    """
    Compute Entity-level F1 as used in SROIE and CORD benchmarks.
    
    A field extraction is a True Positive if the predicted value exactly
    matches the ground truth after normalization. This is the standard
    metric used in ICDAR KIE competitions.
    
    True negatives (both null) are already excluded — Entity F1 only
    counts TP/FP/FN by definition.
    
    Returns:
        Dict with entity_precision, entity_recall, entity_f1,
        and per-field breakdown.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_field = {}
    
    for doc in report.document_results:
        for fname, fresult in doc.field_results.items():
            if fname not in per_field:
                per_field[fname] = {"tp": 0, "fp": 0, "fn": 0}
            
            if fresult.null_status == "tp":
                # Both have values — check if exact match
                if fresult.exact_match_score >= 1.0:
                    total_tp += 1
                    per_field[fname]["tp"] += 1
                else:
                    # Predicted something, but wrong → FP + FN
                    total_fp += 1
                    total_fn += 1
                    per_field[fname]["fp"] += 1
                    per_field[fname]["fn"] += 1
            elif fresult.null_status == "fp":
                # Hallucination: predicted value when GT is null
                total_fp += 1
                per_field[fname]["fp"] += 1
            elif fresult.null_status == "fn":
                # Miss: GT has value but prediction is null
                total_fn += 1
                per_field[fname]["fn"] += 1
            # TN (both null) — not counted in entity F1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Per-field F1
    field_f1 = {}
    for fname, counts in per_field.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
        field_f1[fname] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_field_f1": field_f1,
    }


def compute_anls(report: EvalReport, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute Average Normalized Levenshtein Similarity as used in DocVQA.
    
    For each field, the NLS score is:
        - 1 - NL(gt, pred)  if NL(gt, pred) < threshold
        - 0                  otherwise
    where NL is the normalized Levenshtein distance.
    
    For null handling:
        - Both null (TN): EXCLUDED from scoring (no extraction signal)
        - Hallucination (FP): score = 0.0
        - Miss (FN): score = 0.0
        - Both present (TP): use Levenshtein similarity with threshold
    
    For numeric fields, we use the similarity_score directly since
    it's already computed as a normalized distance.
    
    Args:
        report: EvalReport from the evaluation
        threshold: ANLS threshold (default 0.5, standard in DocVQA)
    
    Returns:
        Dict with anls score and per-field breakdown.
    """
    all_scores = []
    per_field_scores = {}
    total_tn_skipped = 0
    
    for doc in report.document_results:
        for fname, fresult in doc.field_results.items():
            if fname not in per_field_scores:
                per_field_scores[fname] = []
            
            if fresult.null_status == "tn":
                # FIX: Skip true negatives — they inflate ANLS without
                # providing any signal about extraction quality
                total_tn_skipped += 1
                continue
            elif fresult.null_status in ("fp", "fn"):
                # Hallucination or miss
                score = 0.0
            else:
                # Both present — use the Levenshtein-based similarity
                lev_sim = fresult.details.get("levenshtein_similarity")
                
                if fresult.field_type == "numeric":
                    raw_score = fresult.similarity_score
                    score = raw_score if raw_score >= threshold else 0.0
                elif fresult.field_type == "date":
                    score = fresult.similarity_score if fresult.similarity_score >= threshold else 0.0
                elif lev_sim is not None:
                    score = lev_sim if lev_sim >= threshold else 0.0
                else:
                    score = fresult.similarity_score if fresult.similarity_score >= threshold else 0.0
            
            all_scores.append(score)
            per_field_scores[fname].append(score)
    
    # Compute averages
    anls = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    per_field_anls = {}
    for fname, scores in per_field_scores.items():
        per_field_anls[fname] = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "anls": anls,
        "anls_threshold": threshold,
        "num_comparisons": len(all_scores),
        "num_tn_skipped": total_tn_skipped,
        "per_field_anls": per_field_anls,
    }


def compute_standard_metrics(report: EvalReport, anls_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute all standard metrics for benchmark comparability.
    
    Returns a dict with:
        - entity_f1: Entity-level F1 (SROIE/CORD standard)
        - anls: Average Normalized Levenshtein Similarity (DocVQA standard)
        - wfes: Weighted Field Extraction Score (our proposed metric)
        - line_item_f1: Line item F1 if available
    """
    entity_results = compute_entity_f1(report)
    anls_results = compute_anls(report, threshold=anls_threshold)
    
    result = {
        # Standard metrics
        "entity_f1": entity_results["entity_f1"],
        "entity_precision": entity_results["entity_precision"],
        "entity_recall": entity_results["entity_recall"],
        "anls": anls_results["anls"],
        
        # Our proposed metric
        "wfes": report.overall_weighted_score,
        
        # Line items (standard P/R/F1)
        "line_item_f1": report.line_item_metrics.get("item_f1", None),
        "line_item_precision": report.line_item_metrics.get("item_precision", None),
        "line_item_recall": report.line_item_metrics.get("item_recall", None),
        
        # Detailed breakdowns
        "entity_details": entity_results,
        "anls_details": anls_results,
    }
    
    return result


def print_standard_metrics(metrics: Dict[str, Any], model_name: str = ""):
    """Print standard metrics in a clean format."""
    print(f"\n  STANDARD BENCHMARK METRICS{f' ({model_name})' if model_name else ''}")
    print(f"  {'='*50}")
    
    print(f"\n  Entity-level F1 (SROIE/CORD standard)")
    print(f"  {'-'*40}")
    print(f"  {'Precision:':<25} {metrics['entity_precision']:.4f}")
    print(f"  {'Recall:':<25} {metrics['entity_recall']:.4f}")
    print(f"  {'F1:':<25} {metrics['entity_f1']:.4f}")
    
    print(f"\n  ANLS (DocVQA standard, TNs excluded)")
    print(f"  {'-'*40}")
    print(f"  {'ANLS:':<25} {metrics['anls']:.4f}")
    
    print(f"\n  WFES (Proposed, TNs excluded)")
    print(f"  {'-'*40}")
    print(f"  {'WFES:':<25} {metrics['wfes']:.4f}")
    
    if metrics.get("line_item_f1") is not None:
        print(f"\n  Line Item Metrics")
        print(f"  {'-'*40}")
        print(f"  {'Precision:':<25} {metrics['line_item_precision']:.4f}")
        print(f"  {'Recall:':<25} {metrics['line_item_recall']:.4f}")
        print(f"  {'F1:':<25} {metrics['line_item_f1']:.4f}")
    
    print()


def print_standard_comparison(all_metrics: Dict[str, Dict[str, Any]]):
    """Print side-by-side comparison of standard metrics across models."""
    models = list(all_metrics.keys())
    
    print(f"\n  {'='*70}")
    print(f"  STANDARD METRICS COMPARISON")
    print(f"  {'='*70}")
    
    # Header
    header = f"  {'Model':<20}"
    header += f" {'Entity F1':>10}"
    header += f" {'ANLS':>10}"
    header += f" {'WFES':>10}"
    header += f" {'LI-F1':>10}"
    print(header)
    print(f"  {'-'*60}")
    
    # Sort by WFES
    sorted_models = sorted(models, key=lambda m: all_metrics[m]["wfes"], reverse=True)
    
    for model in sorted_models:
        m = all_metrics[model]
        row = f"  {model:<20}"
        row += f" {m['entity_f1']:>10.4f}"
        row += f" {m['anls']:>10.4f}"
        row += f" {m['wfes']:>10.4f}"
        li_f1 = m.get("line_item_f1")
        row += f" {li_f1:>10.4f}" if li_f1 is not None else f" {'--':>10}"
        print(row)
    
    print(f"  {'='*70}\n")