"""
Report Generator for Receipt OCR Evaluation

Produces:
- Console summary tables (per-field, per-tier, overall)
- Per-document detail reports
- JSON export for further analysis
- Multi-model comparison tables

FIX: True negatives (both GT and pred null) are excluded from similarity
averages, exact match rates, and tier/overall score calculations. They are
still tracked separately for transparency (tn_count, tn_rate) but do not
inflate extraction quality metrics. Hallucinations (FP) and misses (FN)
remain fully penalized.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

from schema import FIELD_DEFINITIONS, get_fields_by_tier
from comparators import FieldResult
from line_item_matcher import LineItemEvalResult


@dataclass
class DocumentResult:
    """Evaluation result for a single document."""
    file_name: str
    field_results: Dict[str, FieldResult] = field(default_factory=dict)
    line_item_result: Optional[LineItemEvalResult] = None
    tax_item_result: Optional[LineItemEvalResult] = None
    
    # Aggregate scores
    overall_score: float = 0.0
    tier1_score: float = 0.0
    tier2_score: float = 0.0
    tier3_score: float = 0.0


@dataclass
class EvalReport:
    """Full evaluation report for a model run."""
    model_name: str
    dataset_name: str
    num_documents: int
    
    # Per-document results
    document_results: List[DocumentResult] = field(default_factory=list)
    
    # Aggregate per-field metrics
    field_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Aggregate line item metrics
    line_item_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Overall scores
    overall_weighted_score: float = 0.0
    tier1_score: float = 0.0
    tier2_score: float = 0.0
    tier3_score: float = 0.0
    
    # Grade distribution
    grade_distribution: Dict[str, int] = field(default_factory=dict)


def compute_report(model_name: str, dataset_name: str, 
                   document_results: List[DocumentResult]) -> EvalReport:
    """
    Compute aggregate report from per-document results.
    
    FIX: Renormalizes tier weights so only tiers with actual evaluated
    fields contribute to overall_weighted_score.
    
    FIX: True negatives (both GT and pred null) are excluded from
    similarity_mean, exact_match, and grade percentage calculations.
    They are tracked separately as tn_count / tn_rate for transparency.
    """
    report = EvalReport(
        model_name=model_name,
        dataset_name=dataset_name,
        num_documents=len(document_results),
        document_results=document_results,
    )
    
    if not document_results:
        return report
    
    # ---- Per-field aggregation ----
    # We collect ALL results for presence tracking, but only non-TN
    # results for similarity/grade scoring.
    field_scores = {}       # field_name -> list of scores (excludes TN)
    field_em = {}           # field_name -> list of exact match scores (excludes TN)
    field_grades = {}       # field_name -> list of grades (excludes TN)
    field_all_grades = {}   # field_name -> list of ALL grades (includes TN, for distribution)
    field_presence = {}     # field_name -> null_status counts
    field_total_count = {}  # field_name -> total documents evaluated
    
    for doc in document_results:
        for fname, fresult in doc.field_results.items():
            if fname not in field_scores:
                field_scores[fname] = []
                field_em[fname] = []
                field_grades[fname] = []
                field_all_grades[fname] = []
                field_presence[fname] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
                field_total_count[fname] = 0
            
            field_total_count[fname] += 1
            field_all_grades[fname].append(fresult.grade)
            field_presence[fname][fresult.null_status] += 1
            
            # Only include non-TN results in similarity/grade scoring
            if fresult.null_status != "tn":
                field_scores[fname].append(fresult.similarity_score)
                field_em[fname].append(fresult.exact_match_score)
                field_grades[fname].append(fresult.grade)
    
    for fname in field_scores:
        scores = field_scores[fname]
        em_scores = field_em[fname]
        grades = field_grades[fname]
        all_grades = field_all_grades[fname]
        presence = field_presence[fname]
        n_total = field_total_count[fname]
        n_evaluated = len(scores)  # Excludes TN
        
        report.field_metrics[fname] = {
            "n": n_total,
            "n_evaluated": n_evaluated,  # NEW: count excluding TNs
            # Similarity/EM computed only over non-TN instances
            "exact_match": sum(em_scores) / n_evaluated if n_evaluated > 0 else 0.0,
            "similarity_mean": sum(scores) / n_evaluated if n_evaluated > 0 else 0.0,
            "similarity_min": min(scores) if scores else 0.0,
            "similarity_max": max(scores) if scores else 0.0,
            # Presence rates (computed over all instances)
            "true_positives": presence["tp"],
            "true_negatives": presence["tn"],
            "false_positives": presence["fp"],   # Hallucinations
            "false_negatives": presence["fn"],    # Misses
            "presence_rate": (presence["tp"] + presence["tn"]) / n_total if n_total > 0 else 0.0,
            "hallucination_rate": presence["fp"] / n_total if n_total > 0 else 0.0,
            "miss_rate": presence["fn"] / n_total if n_total > 0 else 0.0,
            "tn_rate": presence["tn"] / n_total if n_total > 0 else 0.0,  # NEW
            # Grade distribution (computed over non-TN instances only)
            "grade_exact": grades.count("exact") / n_evaluated if n_evaluated > 0 else 0.0,
            "grade_acceptable": grades.count("acceptable") / n_evaluated if n_evaluated > 0 else 0.0,
            "grade_partial": grades.count("partial") / n_evaluated if n_evaluated > 0 else 0.0,
            "grade_wrong": grades.count("wrong") / n_evaluated if n_evaluated > 0 else 0.0,
            # Hallucination and miss grades are relative to all instances
            # since they represent real errors
            "grade_null_match": all_grades.count("null_match") / n_total if n_total > 0 else 0.0,
            "grade_hallucination": all_grades.count("hallucination") / n_total if n_total > 0 else 0.0,
            "grade_miss": all_grades.count("miss") / n_total if n_total > 0 else 0.0,
        }
    
    # ---- Line item aggregation ----
    li_results = [doc.line_item_result for doc in document_results if doc.line_item_result]
    if li_results:
        report.line_item_metrics = {
            "count_accuracy": _mean([r.count_accuracy for r in li_results]),
            "item_precision": _mean([r.item_precision for r in li_results]),
            "item_recall": _mean([r.item_recall for r in li_results]),
            "item_f1": _mean([r.item_f1 for r in li_results]),
            "overall_score": _mean([r.overall_score for r in li_results]),
        }
        # Per-field averages across all documents
        all_field_names = set()
        for r in li_results:
            all_field_names.update(r.avg_field_scores.keys())
        for fname in all_field_names:
            vals = [r.avg_field_scores.get(fname, 0.0) for r in li_results if r.avg_field_scores]
            report.line_item_metrics[f"field_{fname}"] = _mean(vals)
    
    # ---- Tier scores ----
    tier_weights_map = {1: 0.60, 2: 0.25, 3: 0.15}
    active_tiers = {}  # tier -> weight (only tiers with evaluated data)

    for tier in [1, 2, 3]:
        tier_fields = get_fields_by_tier(tier)
        tier_scores = []
        tier_field_weights = []
        for fname, fdef in tier_fields.items():
            if fname in report.field_metrics:
                # Only use similarity_mean if there are non-TN instances
                if report.field_metrics[fname]["n_evaluated"] > 0:
                    tier_scores.append(report.field_metrics[fname]["similarity_mean"])
                    tier_field_weights.append(fdef.weight)
                # If all instances are TN for this field, skip it from tier scoring
            elif fname == "line_items" and report.line_item_metrics:
                tier_scores.append(report.line_item_metrics.get("overall_score", 0.0))
                tier_field_weights.append(fdef.weight)
            elif fname == "taxes" and report.line_item_metrics:
                # Check if we have tax-specific metrics via tax_item_result
                tax_results = [doc.tax_item_result for doc in document_results if doc.tax_item_result]
                if tax_results:
                    tax_score = _mean([r.overall_score for r in tax_results])
                    tier_scores.append(tax_score)
                    tier_field_weights.append(fdef.weight)
        
        if tier_field_weights:
            weighted = sum(s * w for s, w in zip(tier_scores, tier_field_weights))
            tier_score = weighted / sum(tier_field_weights)
            active_tiers[tier] = tier_weights_map[tier]
        else:
            tier_score = 0.0
        
        if tier == 1:
            report.tier1_score = tier_score
        elif tier == 2:
            report.tier2_score = tier_score
        else:
            report.tier3_score = tier_score
    
    # ---- Overall weighted score ----
    # Renormalize weights to only include tiers that have data
    if active_tiers:
        total_weight = sum(active_tiers.values())
        tier_score_map = {1: report.tier1_score, 2: report.tier2_score, 3: report.tier3_score}
        report.overall_weighted_score = sum(
            (w / total_weight) * tier_score_map[t]
            for t, w in active_tiers.items()
        )
    else:
        report.overall_weighted_score = 0.0
    
    # ---- Global grade distribution ----
    all_grades = []
    for doc in document_results:
        for fresult in doc.field_results.values():
            all_grades.append(fresult.grade)
    for g in ["exact", "acceptable", "partial", "wrong", "null_match", "hallucination", "miss"]:
        report.grade_distribution[g] = all_grades.count(g)
    
    return report


def print_report(report: EvalReport):
    """Print formatted evaluation report to console."""
    w = 72  # Width
    
    print(f"\n{'='*w}")
    print(f"  RECEIPT OCR EVALUATION REPORT")
    print(f"{'='*w}")
    print(f"  Model:    {report.model_name}")
    print(f"  Dataset:  {report.dataset_name} ({report.num_documents} documents)")
    print(f"{'='*w}")
    
    # ---- Overall Scores ----
    print(f"\n  OVERALL SCORES")
    print(f"  {'-'*40}")
    print(f"  {'Weighted Score:':<25} {report.overall_weighted_score:.4f}")
    print(f"  {'Tier 1 (Critical):':<25} {report.tier1_score:.4f}")
    print(f"  {'Tier 2 (Important):':<25} {report.tier2_score:.4f}")
    print(f"  {'Tier 3 (Supplementary):':<25} {report.tier3_score:.4f}")
    
    # ---- Per-Field Table ----
    for tier in [1, 2, 3]:
        tier_names = {1: "CRITICAL", 2: "IMPORTANT", 3: "SUPPLEMENTARY"}
        tier_fields = get_fields_by_tier(tier)
        
        relevant_fields = {k: v for k, v in tier_fields.items() 
                          if k in report.field_metrics}
        
        if not relevant_fields and tier != 1:
            continue
        
        print(f"\n  TIER {tier}: {tier_names[tier]} FIELDS  (scores exclude true negatives)")
        print(f"  {'-'*74}")
        print(f"  {'Field':<28} {'N_eval':>6} {'EM':>6} {'Sim':>6} {'Exact%':>7} {'Accept%':>8} {'Miss%':>6} {'Hall%':>6}")
        print(f"  {'-'*74}")
        
        for fname, fdef in tier_fields.items():
            if fname in ("line_items", "taxes"):
                continue
            if fname not in report.field_metrics:
                print(f"  {fname:<28} {'--':>6} {'--':>6} {'--':>6} {'--':>7} {'--':>8} {'--':>6} {'--':>6}")
                continue
            
            m = report.field_metrics[fname]
            n_eval = m['n_evaluated']
            print(
                f"  {fname:<28} "
                f"{n_eval:>5} "
                f"{m['exact_match']:>5.2f} "
                f"{m['similarity_mean']:>6.3f} "
                f"{m['grade_exact']*100:>6.1f}% "
                f"{m['grade_acceptable']*100:>7.1f}% "
                f"{m['miss_rate']*100:>5.1f}% "
                f"{m['hallucination_rate']*100:>5.1f}%"
            )
    
    # ---- Line Items ----
    if report.line_item_metrics:
        print(f"\n  LINE ITEMS")
        print(f"  {'-'*50}")
        print(f"  {'Count Accuracy:':<30} {report.line_item_metrics.get('count_accuracy', 0):.3f}")
        print(f"  {'Item Precision:':<30} {report.line_item_metrics.get('item_precision', 0):.3f}")
        print(f"  {'Item Recall:':<30} {report.line_item_metrics.get('item_recall', 0):.3f}")
        print(f"  {'Item F1:':<30} {report.line_item_metrics.get('item_f1', 0):.3f}")
        print(f"  {'Overall Score:':<30} {report.line_item_metrics.get('overall_score', 0):.3f}")
        
        # Per-field in line items
        li_fields = {k: v for k, v in report.line_item_metrics.items() if k.startswith("field_")}
        if li_fields:
            print(f"\n  {'Per-field (matched pairs):'}")
            for k, v in sorted(li_fields.items()):
                field_name = k.replace("field_", "")
                print(f"    {field_name:<25} {v:.3f}")
    
    # ---- Grade Distribution ----
    print(f"\n  GRADE DISTRIBUTION (all fields, including true negatives)")
    print(f"  {'-'*40}")
    total = sum(report.grade_distribution.values())
    if total > 0:
        for grade in ["exact", "acceptable", "partial", "wrong", "null_match", "hallucination", "miss"]:
            count = report.grade_distribution.get(grade, 0)
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {grade:<15} {count:>5} ({pct:>5.1f}%) {bar}")
        
        # Also show the TN-excluded view
        tn_count = report.grade_distribution.get("null_match", 0)
        non_tn_total = total - tn_count
        if non_tn_total > 0 and tn_count > 0:
            print(f"\n  GRADE DISTRIBUTION (excluding {tn_count} true negatives)")
            print(f"  {'-'*40}")
            for grade in ["exact", "acceptable", "partial", "wrong", "hallucination", "miss"]:
                count = report.grade_distribution.get(grade, 0)
                pct = count / non_tn_total * 100
                bar = "█" * int(pct / 2)
                print(f"  {grade:<15} {count:>5} ({pct:>5.1f}%) {bar}")
    
    # ---- Worst-performing documents ----
    print(f"\n  WORST DOCUMENTS (by overall score)")
    print(f"  {'-'*50}")
    sorted_docs = sorted(report.document_results, key=lambda d: d.overall_score)
    for doc in sorted_docs[:5]:
        print(f"  {doc.file_name:<40} {doc.overall_score:.3f}")
    
    print(f"\n{'='*w}\n")


def print_comparison(reports: List[EvalReport]):
    """Print side-by-side comparison of multiple models."""
    if not reports:
        return
    
    w = 72
    print(f"\n{'='*w}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*w}")
    print(f"  Dataset: {reports[0].dataset_name} ({reports[0].num_documents} documents)")
    print()
    
    # Header
    model_names = [r.model_name for r in reports]
    header = f"  {'Metric':<25}"
    for name in model_names:
        header += f" {name[:12]:>12}"
    print(header)
    print(f"  {'-'*(25 + 13*len(model_names))}")
    
    # Overall scores
    row = f"  {'Overall Weighted':<25}"
    for r in reports:
        row += f" {r.overall_weighted_score:>12.4f}"
    print(row)
    
    for tier, label in [(1, "Tier 1 (Critical)"), (2, "Tier 2 (Important)"), (3, "Tier 3 (Supplementary)")]:
        row = f"  {label:<25}"
        for r in reports:
            score = [r.tier1_score, r.tier2_score, r.tier3_score][tier-1]
            row += f" {score:>12.4f}"
        print(row)
    
    # Per-field comparison
    print(f"\n  {'Field Similarity Scores (excl. TNs)':<25}")
    print(f"  {'-'*(25 + 13*len(model_names))}")
    
    all_fields = set()
    for r in reports:
        all_fields.update(r.field_metrics.keys())
    
    for fname in sorted(all_fields):
        if fname in ("line_items", "taxes"):
            continue
        row = f"  {fname:<25}"
        for r in reports:
            if fname in r.field_metrics:
                score = r.field_metrics[fname]["similarity_mean"]
                row += f" {score:>12.3f}"
            else:
                row += f" {'--':>12}"
        print(row)
    
    # Line items comparison
    print(f"\n  {'Line Item Metrics':<25}")
    print(f"  {'-'*(25 + 13*len(model_names))}")
    for metric in ["item_f1", "count_accuracy", "overall_score"]:
        row = f"  {metric:<25}"
        for r in reports:
            val = r.line_item_metrics.get(metric, 0.0)
            row += f" {val:>12.3f}"
        print(row)
    
    # Winner
    print(f"\n  {'WINNER':<25}", end="")
    best_idx = max(range(len(reports)), key=lambda i: reports[i].overall_weighted_score)
    print(f" → {reports[best_idx].model_name} ({reports[best_idx].overall_weighted_score:.4f})")
    print(f"\n{'='*w}\n")


def export_report(report: EvalReport, output_path: str):
    """Export report as JSON for further analysis."""
    
    # Build serializable dict
    export = {
        "model_name": report.model_name,
        "dataset_name": report.dataset_name,
        "num_documents": report.num_documents,
        "overall_weighted_score": report.overall_weighted_score,
        "tier1_score": report.tier1_score,
        "tier2_score": report.tier2_score,
        "tier3_score": report.tier3_score,
        "field_metrics": report.field_metrics,
        "line_item_metrics": report.line_item_metrics,
        "grade_distribution": report.grade_distribution,
        "documents": [],
    }
    
    for doc in report.document_results:
        doc_export = {
            "file_name": doc.file_name,
            "overall_score": doc.overall_score,
            "tier1_score": doc.tier1_score,
            "tier2_score": doc.tier2_score,
            "tier3_score": doc.tier3_score,
            "fields": {},
        }
        
        for fname, fresult in doc.field_results.items():
            doc_export["fields"][fname] = {
                "exact_match": fresult.exact_match_score,
                "similarity": fresult.similarity_score,
                "embedding": fresult.embedding_score,
                "grade": fresult.grade,
                "null_status": fresult.null_status,
                "gt_value": _safe_serialize(fresult.gt_value),
                "pred_value": _safe_serialize(fresult.pred_value),
                "details": fresult.details,
            }
        
        if doc.line_item_result:
            li = doc.line_item_result
            doc_export["line_items"] = {
                "gt_count": li.gt_count,
                "pred_count": li.pred_count,
                "count_accuracy": li.count_accuracy,
                "item_f1": li.item_f1,
                "overall_score": li.overall_score,
                "matched_pairs": len(li.matched_pairs),
                "unmatched_gt": len(li.unmatched_gt),
                "unmatched_pred": len(li.unmatched_pred),
                "avg_field_scores": li.avg_field_scores,
            }
        
        export["documents"].append(doc_export)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Report exported to: {output_path}")


def _safe_serialize(val):
    """Make a value JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, dict)):
        return val
    return str(val)


def _mean(values: list) -> float:
    """Safe mean calculation."""
    if not values:
        return 0.0
    return sum(values) / len(values)