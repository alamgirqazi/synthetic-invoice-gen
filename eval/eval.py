"""
Receipt OCR Evaluation Engine

Main entry point. Loads ground truth and prediction JSON files,
runs field-by-field comparison, produces aggregate reports.

Usage:
    # Evaluate single model
    python -m receipt_eval --gt ground_truth.json --pred predictions.json

    # Compare multiple models
    python -m receipt_eval --gt ground_truth.json --pred model_a.json model_b.json --compare

    # Custom output
    python -m receipt_eval --gt ground_truth.json --pred predictions.json -o results/

    # Skip embeddings (faster, no sentence-transformers dependency)
    python -m receipt_eval --gt ground_truth.json --pred predictions.json --no-embeddings

    # Only evaluate specific tiers
    python -m receipt_eval --gt ground_truth.json --pred predictions.json --tiers 1 2
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from schema import FIELD_DEFINITIONS, get_evaluatable_fields, get_fields_by_tier
from comparators import compare_field, FieldResult
from line_item_matcher import evaluate_line_items, evaluate_tax_items
from report import (
    DocumentResult, EvalReport, 
    compute_report, print_report, print_comparison, export_report
)


class ReceiptEvaluator:
    """
    Main evaluator class.
    
    Compares ground truth and predicted receipts field by field,
    handling line items with Hungarian matching.
    """
    
    def __init__(self, 
                 use_embeddings: bool = True,
                 tiers: Optional[Set[int]] = None,
                 extra_fields: Optional[Dict] = None):
        """
        Args:
            use_embeddings: Whether to use sentence-transformers for semantic similarity
            tiers: Which tiers to evaluate (None = all)
            extra_fields: Additional field definitions to register
        """
        self.use_embeddings = use_embeddings
        self.tiers = tiers or {1, 2, 3}
        
        # Disable embeddings globally if requested
        if not use_embeddings:
            import metrics
            metrics._embedding_model = False  # Mark as unavailable
        
        # Register extra fields if provided
        if extra_fields:
            for fname, fdef in extra_fields.items():
                FIELD_DEFINITIONS[fname] = fdef
    
    def evaluate_document(self, gt_doc: Dict[str, Any], 
                          pred_doc: Dict[str, Any]) -> DocumentResult:
        """
        Evaluate a single document (GT vs prediction).
        
        Args:
            gt_doc: Ground truth document dict
            pred_doc: Predicted document dict
        
        Returns:
            DocumentResult with per-field scores
        """
        file_name = gt_doc.get("file_name", pred_doc.get("file_name", "unknown"))
        result = DocumentResult(file_name=file_name)
        
        evaluatable = get_evaluatable_fields()
        
        for fname, fdef in evaluatable.items():
            # Skip tiers we're not evaluating
            if fdef.tier not in self.tiers:
                continue
            
            # Skip complex types (handled separately)
            if fdef.field_type in ("line_items", "tax_items"):
                continue
            
            gt_value = gt_doc.get(fname)
            pred_value = pred_doc.get(fname)
            
            field_result = compare_field(fdef, gt_value, pred_value)
            result.field_results[fname] = field_result
        
        # ---- Line Items ----
        if "line_items" in evaluatable and evaluatable["line_items"].tier in self.tiers:
            gt_items = gt_doc.get("line_items") or []
            pred_items = pred_doc.get("line_items") or []
            
            if isinstance(gt_items, list) or isinstance(pred_items, list):
                result.line_item_result = evaluate_line_items(
                    gt_items if isinstance(gt_items, list) else [],
                    pred_items if isinstance(pred_items, list) else []
                )
        
        # ---- Tax Items ----
        if "taxes" in evaluatable and evaluatable["taxes"].tier in self.tiers:
            gt_taxes = gt_doc.get("taxes") or []
            pred_taxes = pred_doc.get("taxes") or []
            
            if isinstance(gt_taxes, list) or isinstance(pred_taxes, list):
                result.tax_item_result = evaluate_tax_items(
                    gt_taxes if isinstance(gt_taxes, list) else [],
                    pred_taxes if isinstance(pred_taxes, list) else []
                )
        
        # ---- Compute document-level scores ----
        self._compute_doc_scores(result)
        
        return result
    
    def _compute_doc_scores(self, result: DocumentResult):
        """
        Compute tier-level and overall scores for a document.
        
        FIX: Renormalizes tier weights so that only tiers with evaluated
        fields contribute to the overall score.
        
        FIX: Excludes true negatives (both GT and pred null) from scoring.
        TN fields tell us nothing about extraction quality — they just mean
        the field wasn't on the document and the model didn't invent it.
        Hallucinations (FP) and misses (FN) are still penalized.
        """
        tier_weights = {1: 0.60, 2: 0.25, 3: 0.15}
        tier_scores = {}
        active_tiers = {}  # tier -> weight (only tiers with data)

        for tier in [1, 2, 3]:
            scores = []
            weights = []
            
            for fname, fresult in result.field_results.items():
                if fresult.tier == tier:
                    # SKIP true negatives — they inflate scores without
                    # telling us anything about extraction quality
                    if fresult.null_status == "tn":
                        continue
                    scores.append(fresult.similarity_score)
                    weights.append(fresult.weight)
            
            # Add line item score to tier 1
            if tier == 1 and result.line_item_result:
                li_def = FIELD_DEFINITIONS.get("line_items")
                if li_def:
                    scores.append(result.line_item_result.overall_score)
                    weights.append(li_def.weight)
            
            # Add tax item score to tier 2
            if tier == 2 and result.tax_item_result:
                tax_def = FIELD_DEFINITIONS.get("taxes")
                if tax_def:
                    scores.append(result.tax_item_result.overall_score)
                    weights.append(tax_def.weight)
            
            if weights:
                weighted = sum(s * w for s, w in zip(scores, weights))
                tier_score = weighted / sum(weights)
                active_tiers[tier] = tier_weights[tier]
            else:
                tier_score = 0.0
            
            tier_scores[tier] = tier_score

            if tier == 1:
                result.tier1_score = tier_score
            elif tier == 2:
                result.tier2_score = tier_score
            else:
                result.tier3_score = tier_score
        
        # Overall: weighted combination of ACTIVE tiers only
        # Renormalize weights so they sum to 1.0
        if active_tiers:
            total_weight = sum(active_tiers.values())
            result.overall_score = sum(
                (w / total_weight) * tier_scores[t]
                for t, w in active_tiers.items()
            )
        else:
            result.overall_score = 0.0
    
    def evaluate_dataset(self, gt_docs: List[Dict], pred_docs: List[Dict],
                         model_name: str = "unknown",
                         dataset_name: str = "unknown") -> EvalReport:
        """
        Evaluate an entire dataset.
        
        Matches documents by file_name, then evaluates each pair.
        
        Args:
            gt_docs: List of ground truth documents
            pred_docs: List of predicted documents
            model_name: Name for the report
            dataset_name: Name for the report
        
        Returns:
            EvalReport with all metrics
        """
        # Build lookup by file_name
        gt_lookup = {}
        for doc in gt_docs:
            fname = doc.get("file_name", "")
            if fname:
                gt_lookup[fname] = doc
        
        pred_lookup = {}
        for doc in pred_docs:
            fname = doc.get("file_name", "")
            if fname:
                pred_lookup[fname] = doc
        
        # Find matching documents
        gt_files = set(gt_lookup.keys())
        pred_files = set(pred_lookup.keys())
        matched_files = gt_files & pred_files
        gt_only = gt_files - pred_files
        pred_only = pred_files - gt_files
        
        if gt_only:
            print(f"  WARNING: {len(gt_only)} GT documents not found in predictions:")
            for f in sorted(gt_only)[:5]:
                print(f"    - {f}")
            if len(gt_only) > 5:
                print(f"    ... and {len(gt_only)-5} more")
        
        if pred_only:
            print(f"  WARNING: {len(pred_only)} predicted documents not in GT:")
            for f in sorted(pred_only)[:5]:
                print(f"    - {f}")
            if len(pred_only) > 5:
                print(f"    ... and {len(pred_only)-5} more")
        
        print(f"  Evaluating {len(matched_files)} matched documents...")
        
        # Evaluate each matched pair
        document_results = []
        for i, fname in enumerate(sorted(matched_files)):
            gt_doc = gt_lookup[fname]
            pred_doc = pred_lookup[fname]
            
            doc_result = self.evaluate_document(gt_doc, pred_doc)
            document_results.append(doc_result)
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(matched_files)} documents...")
        
        # Build report
        report = compute_report(model_name, dataset_name, document_results)
        return report


def load_json(path: str) -> List[Dict]:
    """Load and validate a JSON file containing document arrays."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Single document - wrap in list
        data = [data]
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Receipt OCR Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python -m receipt_eval --gt ground_truth.json --pred olmocr2_output.json

  # Compare multiple models
  python -m receipt_eval --gt ground_truth.json \\
      --pred olmocr2.json nanonets.json dots.json --compare

  # Fast mode (no embeddings)
  python -m receipt_eval --gt gt.json --pred pred.json --no-embeddings

  # Evaluate only critical fields
  python -m receipt_eval --gt gt.json --pred pred.json --tiers 1

  # Export JSON report
  python -m receipt_eval --gt gt.json --pred pred.json -o eval_results/
"""
    )
    
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--pred", nargs="+", required=True, help="Path(s) to prediction JSON file(s)")
    parser.add_argument("-o", "--output", default=None, help="Output directory for reports (JSON export)")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models side by side")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding similarity (faster)")
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3], 
                       help="Which tiers to evaluate (default: 1 2 3)")
    parser.add_argument("--dataset-name", default=None, help="Name for the dataset in reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-document details")
    
    args = parser.parse_args()
    
    # Load ground truth
    print(f"\nLoading ground truth: {args.gt}")
    gt_docs = load_json(args.gt)
    print(f"  Loaded {len(gt_docs)} documents")
    
    dataset_name = args.dataset_name or Path(args.gt).stem
    
    # Initialize evaluator
    evaluator = ReceiptEvaluator(
        use_embeddings=not args.no_embeddings,
        tiers=set(args.tiers),
    )
    
    # Process each prediction file
    reports = []
    
    for pred_path in args.pred:
        print(f"\nLoading predictions: {pred_path}")
        pred_docs = load_json(pred_path)
        print(f"  Loaded {len(pred_docs)} documents")
        
        model_name = Path(pred_path).stem
        
        start = time.time()
        report = evaluator.evaluate_dataset(
            gt_docs, pred_docs,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        elapsed = time.time() - start
        print(f"  Evaluation completed in {elapsed:.2f}s")
        
        reports.append(report)
        
        # Print individual report
        print_report(report)
        
        # Export JSON if output dir specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            export_path = output_dir / f"eval_{model_name}.json"
            export_report(report, str(export_path))
        
        # Verbose: per-document details
        if args.verbose:
            _print_verbose(report)
    
    # Multi-model comparison
    if args.compare and len(reports) > 1:
        print_comparison(reports)
    
    print("Done.")


def _print_verbose(report: EvalReport):
    """Print detailed per-document results."""
    print(f"\n  DETAILED RESULTS ({report.model_name})")
    print(f"  {'='*70}")
    
    for doc in sorted(report.document_results, key=lambda d: d.overall_score):
        print(f"\n  📄 {doc.file_name}  (score: {doc.overall_score:.3f})")
        
        for fname, fresult in sorted(doc.field_results.items()):
            icon = {
                "exact": "✅", "acceptable": "🟡", "partial": "🟠",
                "wrong": "❌", "null_match": "⬜", "hallucination": "🔴", "miss": "🟤"
            }.get(fresult.grade, "❓")
            
            gt_display = _truncate(str(fresult.gt_value), 25)
            pred_display = _truncate(str(fresult.pred_value), 25)
            
            print(
                f"    {icon} {fname:<28} "
                f"sim={fresult.similarity_score:.2f}  "
                f"GT={gt_display:<25} "
                f"Pred={pred_display}"
            )
        
        if doc.line_item_result:
            li = doc.line_item_result
            print(f"    📋 Line items: {li.gt_count} GT / {li.pred_count} pred / "
                  f"F1={li.item_f1:.2f} / score={li.overall_score:.2f}")


def _truncate(s: str, max_len: int) -> str:
    """Truncate string for display."""
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."


if __name__ == "__main__":
    main()