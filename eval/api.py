"""
FastAPI Wrapper for Receipt OCR Evaluation Framework

Exposes the evaluation engine via REST endpoints with support for:
- Single model evaluation
- Multi-model comparison
- Single document pair evaluation
- Dynamic field/tier selection
- Schema introspection
- Standard benchmark metrics (Entity F1, ANLS, WFES)

FIX: True negatives (both GT and pred null) are excluded from all scoring
calculations. They are still tracked for transparency (tn_rate, num_tn_skipped).
"""

import json
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from schema import (
    FIELD_DEFINITIONS, LINE_ITEM_FIELDS, TAX_ITEM_FIELDS,
    get_evaluatable_fields, get_fields_by_tier,
)
from comparators import compare_field, FieldResult
from line_item_matcher import evaluate_line_items, evaluate_tax_items
from eval import ReceiptEvaluator
from report import (
    DocumentResult, EvalReport,
    compute_report, export_report,
)
from standard_metrics import (
    compute_standard_metrics,
    compute_entity_f1,
    compute_anls,
)


# ── App setup ────────────────────────────────────────────────

app = FastAPI(
    title="Receipt OCR Evaluation API",
    version="2.1.0",
    description=(
        "Benchmark-style evaluation API for comparing OCR model outputs on "
        "receipt/invoice documents. Supports field-level comparison, "
        "multi-model comparison, dynamic field selection, and standard "
        "benchmark metrics (Entity F1, ANLS, WFES). "
        "True negatives (both GT and pred null) are excluded from all "
        "scoring to prevent inflated metrics."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Metric selection enum ────────────────────────────────────

class MetricType(str, Enum):
    wfes = "wfes"
    entity_f1 = "entity_f1"
    anls = "anls"
    all = "all"


# ── Pydantic request / response models ──────────────────────

class EvalRequest(BaseModel):
    ground_truth: List[Dict[str, Any]] = Field(..., description="Ground truth documents")
    predictions: List[Dict[str, Any]] = Field(..., description="Predicted documents")
    model_name: str = Field("model", description="Label for this model run")
    dataset_name: str = Field("dataset", description="Label for the dataset")
    tiers: Optional[List[int]] = Field(None, description="Tiers to evaluate (default: [1,2,3])")
    fields: Optional[List[str]] = Field(None, description="Specific field names to evaluate")
    use_embeddings: bool = Field(False, description="Use sentence-transformer embeddings")
    metrics: List[MetricType] = Field(
        default=[MetricType.all],
        description="Which metrics to compute: wfes, entity_f1, anls, or all"
    )

class CompareRequest(BaseModel):
    ground_truth: List[Dict[str, Any]]
    predictions: Dict[str, List[Dict[str, Any]]] = Field(
        ..., description="Map of model_name -> predicted documents"
    )
    dataset_name: str = "dataset"
    tiers: Optional[List[int]] = None
    fields: Optional[List[str]] = None
    use_embeddings: bool = False
    metrics: List[MetricType] = Field(
        default=[MetricType.all],
        description="Which metrics to compute: wfes, entity_f1, anls, or all"
    )

class DocumentPairRequest(BaseModel):
    ground_truth: Dict[str, Any]
    prediction: Dict[str, Any]
    tiers: Optional[List[int]] = None
    fields: Optional[List[str]] = None
    use_embeddings: bool = False
    metrics: List[MetricType] = Field(default=[MetricType.all])

class FieldCompareRequest(BaseModel):
    field_name: str
    gt_value: Any
    pred_value: Any


# ── Helpers ──────────────────────────────────────────────────

def _should_compute(metric: MetricType, requested: List[MetricType]) -> bool:
    return MetricType.all in requested or metric in requested


def _build_evaluator(
    use_embeddings: bool = False,
    tiers: Optional[List[int]] = None,
    fields: Optional[List[str]] = None,
) -> ReceiptEvaluator:
    tier_set = set(tiers) if tiers else {1, 2, 3}
    evaluator = ReceiptEvaluator(use_embeddings=use_embeddings, tiers=tier_set)
    evaluator._field_filter = set(fields) if fields else None
    return evaluator


def _evaluate_document_filtered(
    evaluator: ReceiptEvaluator,
    gt_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
) -> DocumentResult:
    file_name = gt_doc.get("file_name", pred_doc.get("file_name", "unknown"))
    result = DocumentResult(file_name=file_name)
    evaluatable = get_evaluatable_fields()
    field_filter: Optional[Set[str]] = getattr(evaluator, "_field_filter", None)

    for fname, fdef in evaluatable.items():
        if fdef.tier not in evaluator.tiers:
            continue
        if fdef.field_type in ("line_items", "tax_items"):
            continue
        if field_filter and fname not in field_filter:
            continue
        gt_value = gt_doc.get(fname)
        pred_value = pred_doc.get(fname)
        result.field_results[fname] = compare_field(fdef, gt_value, pred_value)

    include_li = (field_filter is None or "line_items" in field_filter)
    if include_li and "line_items" in evaluatable and evaluatable["line_items"].tier in evaluator.tiers:
        gt_items = gt_doc.get("line_items") or []
        pred_items = pred_doc.get("line_items") or []
        if isinstance(gt_items, list) or isinstance(pred_items, list):
            result.line_item_result = evaluate_line_items(
                gt_items if isinstance(gt_items, list) else [],
                pred_items if isinstance(pred_items, list) else [],
            )

    include_tax = (field_filter is None or "taxes" in field_filter)
    if include_tax and "taxes" in evaluatable and evaluatable["taxes"].tier in evaluator.tiers:
        gt_taxes = gt_doc.get("taxes") or []
        pred_taxes = pred_doc.get("taxes") or []
        if isinstance(gt_taxes, list) or isinstance(pred_taxes, list):
            result.tax_item_result = evaluate_tax_items(
                gt_taxes if isinstance(gt_taxes, list) else [],
                pred_taxes if isinstance(pred_taxes, list) else [],
            )

    evaluator._compute_doc_scores(result)
    return result


def _run_evaluation(
    evaluator: ReceiptEvaluator,
    gt_docs: List[Dict],
    pred_docs: List[Dict],
    model_name: str,
    dataset_name: str,
) -> EvalReport:
    """Run evaluation and return the raw EvalReport."""
    if evaluator._field_filter:
        gt_lookup = {d.get("file_name", ""): d for d in gt_docs}
        pred_lookup = {d.get("file_name", ""): d for d in pred_docs}
        matched = set(gt_lookup) & set(pred_lookup)
        doc_results = [
            _evaluate_document_filtered(evaluator, gt_lookup[f], pred_lookup[f])
            for f in sorted(matched)
        ]
        return compute_report(model_name, dataset_name, doc_results)
    else:
        return evaluator.evaluate_dataset(
            gt_docs, pred_docs,
            model_name=model_name,
            dataset_name=dataset_name,
        )


def _serialize_report(
    report: EvalReport,
    metrics: List[MetricType] = None,
    include_documents: bool = True,
) -> dict:
    """Convert an EvalReport to a JSON-safe dict, with optional standard metrics."""
    if metrics is None:
        metrics = [MetricType.all]

    out = {
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
        "scoring_note": "True negatives (both GT and pred null) are excluded from similarity and grade scores.",
    }

    # Standard benchmark metrics
    standard = {}

    if _should_compute(MetricType.wfes, metrics):
        standard["wfes"] = {
            "overall": report.overall_weighted_score,
            "tier1": report.tier1_score,
            "tier2": report.tier2_score,
            "tier3": report.tier3_score,
        }

    if _should_compute(MetricType.entity_f1, metrics):
        ef1 = compute_entity_f1(report)
        standard["entity_f1"] = {
            "precision": ef1["entity_precision"],
            "recall": ef1["entity_recall"],
            "f1": ef1["entity_f1"],
            "total_tp": ef1["total_tp"],
            "total_fp": ef1["total_fp"],
            "total_fn": ef1["total_fn"],
            "per_field": ef1["per_field_f1"],
        }

    if _should_compute(MetricType.anls, metrics):
        anls = compute_anls(report)
        standard["anls"] = {
            "score": anls["anls"],
            "threshold": anls["anls_threshold"],
            "num_comparisons": anls["num_comparisons"],
            "num_tn_skipped": anls["num_tn_skipped"],
            "per_field": anls["per_field_anls"],
        }

    if standard:
        out["standard_metrics"] = standard

    if include_documents:
        out["documents"] = _serialize_documents(report)

    return out


def _serialize_documents(report: EvalReport) -> list:
    docs = []
    for doc in report.document_results:
        fields = {}
        for fname, fr in doc.field_results.items():
            fields[fname] = {
                "exact_match": fr.exact_match_score,
                "similarity": fr.similarity_score,
                "embedding": fr.embedding_score,
                "grade": fr.grade,
                "null_status": fr.null_status,
                "gt_value": fr.gt_value if _is_json_safe(fr.gt_value) else str(fr.gt_value),
                "pred_value": fr.pred_value if _is_json_safe(fr.pred_value) else str(fr.pred_value),
                "details": fr.details,
            }
        d = {
            "file_name": doc.file_name,
            "overall_score": doc.overall_score,
            "tier1_score": doc.tier1_score,
            "tier2_score": doc.tier2_score,
            "tier3_score": doc.tier3_score,
            "fields": fields,
        }
        if doc.line_item_result:
            li = doc.line_item_result
            d["line_items"] = {
                "gt_count": li.gt_count, "pred_count": li.pred_count,
                "count_accuracy": li.count_accuracy,
                "item_precision": li.item_precision, "item_recall": li.item_recall,
                "item_f1": li.item_f1, "overall_score": li.overall_score,
                "avg_field_scores": li.avg_field_scores,
                "unmatched_gt": li.unmatched_gt, "unmatched_pred": li.unmatched_pred,
            }
        if doc.tax_item_result:
            tx = doc.tax_item_result
            d["taxes"] = {
                "gt_count": tx.gt_count, "pred_count": tx.pred_count,
                "item_f1": tx.item_f1, "overall_score": tx.overall_score,
                "avg_field_scores": tx.avg_field_scores,
            }
        docs.append(d)
    return docs


def _is_json_safe(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool, list, dict))


# ── Routes ───────────────────────────────────────────────────

# 1. Schema introspection

@app.get("/schema/fields", summary="List all evaluatable fields")
def list_fields():
    out = {}
    for fname, fdef in FIELD_DEFINITIONS.items():
        out[fname] = {
            "field_type": fdef.field_type, "tier": fdef.tier,
            "weight": fdef.weight, "description": fdef.description,
            "numeric_tolerance": fdef.numeric_tolerance,
            "threshold_exact": fdef.threshold_exact,
            "threshold_acceptable": fdef.threshold_acceptable,
            "threshold_partial": fdef.threshold_partial,
            "identity_critical": fdef.identity_critical,
        }
    return out

@app.get("/schema/fields/tier/{tier}", summary="Fields for a given tier")
def list_fields_by_tier(tier: int):
    if tier not in (1, 2, 3):
        raise HTTPException(400, "Tier must be 1, 2, or 3")
    return {
        fname: {"field_type": fd.field_type, "tier": fd.tier,
                "weight": fd.weight, "description": fd.description}
        for fname, fd in get_fields_by_tier(tier).items()
    }

@app.get("/schema/line_item_fields", summary="Line-item sub-field definitions")
def list_line_item_fields():
    return {
        "line_item_fields": {k: {"field_type": v.field_type, "weight": v.weight}
                             for k, v in LINE_ITEM_FIELDS.items()},
        "tax_item_fields": {k: {"field_type": v.field_type, "weight": v.weight}
                            for k, v in TAX_ITEM_FIELDS.items()},
    }

@app.get("/schema/metrics", summary="Available metric types and descriptions")
def list_metrics():
    return {
        "available_metrics": [
            {"name": "wfes", "full_name": "Weighted Field Extraction Score",
             "description": "Type-aware similarity, tiered weighting (60/25/15). TNs excluded.",
             "standard": False},
            {"name": "entity_f1", "full_name": "Entity-level F1",
             "description": "Binary exact match P/R/F1 (SROIE/CORD standard). TNs excluded by definition.",
             "standard": True, "used_in": ["SROIE", "CORD", "KIEval"]},
            {"name": "anls", "full_name": "Average Normalized Levenshtein Similarity",
             "description": "Normalized edit distance with threshold (DocVQA standard). TNs excluded.",
             "standard": True, "used_in": ["DocVQA", "ANLS*", "ICDAR HWD 2024"]},
        ],
        "usage": "Pass metrics=['entity_f1','anls'] or metrics=['all'] in request body",
        "scoring_note": "True negatives (both GT and pred null) are excluded from all metrics to prevent score inflation.",
    }


# 2. Single-field comparison

@app.post("/evaluate/field", summary="Compare a single field value pair")
def evaluate_single_field(req: FieldCompareRequest):
    fdef = FIELD_DEFINITIONS.get(req.field_name)
    if not fdef:
        raise HTTPException(404, f"Unknown field: {req.field_name}")
    fr = compare_field(fdef, req.gt_value, req.pred_value)
    return {
        "field_name": fr.field_name, "field_type": fr.field_type,
        "exact_match": fr.exact_match_score, "similarity": fr.similarity_score,
        "embedding": fr.embedding_score, "grade": fr.grade,
        "null_status": fr.null_status, "details": fr.details,
    }


# 3. Single-document evaluation

@app.post("/evaluate/document", summary="Evaluate one GT / prediction document pair")
def evaluate_document(req: DocumentPairRequest):
    evaluator = _build_evaluator(
        use_embeddings=req.use_embeddings, tiers=req.tiers, fields=req.fields,
    )
    doc_result = _evaluate_document_filtered(evaluator, req.ground_truth, req.prediction)

    fields = {}
    for fname, fr in doc_result.field_results.items():
        fields[fname] = {
            "exact_match": fr.exact_match_score, "similarity": fr.similarity_score,
            "embedding": fr.embedding_score, "grade": fr.grade,
            "null_status": fr.null_status,
            "gt_value": fr.gt_value if _is_json_safe(fr.gt_value) else str(fr.gt_value),
            "pred_value": fr.pred_value if _is_json_safe(fr.pred_value) else str(fr.pred_value),
            "details": fr.details,
        }

    result: Dict[str, Any] = {
        "file_name": doc_result.file_name,
        "overall_score": doc_result.overall_score,
        "tier1_score": doc_result.tier1_score,
        "tier2_score": doc_result.tier2_score,
        "tier3_score": doc_result.tier3_score,
        "fields": fields,
    }
    if doc_result.line_item_result:
        li = doc_result.line_item_result
        result["line_items"] = {
            "gt_count": li.gt_count, "pred_count": li.pred_count,
            "item_f1": li.item_f1, "overall_score": li.overall_score,
            "avg_field_scores": li.avg_field_scores,
        }
    if doc_result.tax_item_result:
        tx = doc_result.tax_item_result
        result["taxes"] = {
            "gt_count": tx.gt_count, "pred_count": tx.pred_count,
            "item_f1": tx.item_f1, "overall_score": tx.overall_score,
            "avg_field_scores": tx.avg_field_scores,
        }
    return result


# 4. Full dataset evaluation

@app.post("/evaluate", summary="Evaluate predictions against ground truth")
def evaluate_dataset(req: EvalRequest):
    try:
        evaluator = _build_evaluator(
            use_embeddings=req.use_embeddings, tiers=req.tiers, fields=req.fields,
        )
        report = _run_evaluation(
            evaluator, req.ground_truth, req.predictions,
            req.model_name, req.dataset_name,
        )
        return _serialize_report(report, metrics=req.metrics)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, detail=str(exc))


# 5. Multi-model comparison

@app.post("/compare", summary="Compare 2+ models against the same ground truth")
def compare_models(req: CompareRequest):
    if len(req.predictions) < 2:
        raise HTTPException(400, "Provide at least 2 prediction sets to compare")

    try:
        evaluator = _build_evaluator(
            use_embeddings=req.use_embeddings, tiers=req.tiers, fields=req.fields,
        )

        # Evaluate all models, keeping raw reports for standard metrics
        raw_reports: Dict[str, EvalReport] = {}
        serialized: Dict[str, dict] = {}

        for model_name, preds in req.predictions.items():
            report = _run_evaluation(
                evaluator, req.ground_truth, preds,
                model_name, req.dataset_name,
            )
            raw_reports[model_name] = report
            serialized[model_name] = _serialize_report(report, metrics=req.metrics)

        # Per-field side-by-side
        all_fields = set()
        for r in serialized.values():
            all_fields.update(r["field_metrics"].keys())
        field_comparison = {
            fname: {mname: r["field_metrics"].get(fname, {}).get("similarity_mean")
                    for mname, r in serialized.items()}
            for fname in sorted(all_fields)
        }

        # Rankings by each requested metric
        rankings = {}

        # WFES ranking (always)
        rankings["wfes"] = sorted(
            [{"model": m, "score": r["overall_weighted_score"]} for m, r in serialized.items()],
            key=lambda x: x["score"], reverse=True,
        )

        if _should_compute(MetricType.entity_f1, req.metrics):
            ef1_data = {m: compute_entity_f1(r) for m, r in raw_reports.items()}
            rankings["entity_f1"] = sorted(
                [{"model": m, "score": d["entity_f1"],
                  "precision": d["entity_precision"], "recall": d["entity_recall"]}
                 for m, d in ef1_data.items()],
                key=lambda x: x["score"], reverse=True,
            )

        if _should_compute(MetricType.anls, req.metrics):
            anls_data = {m: compute_anls(r) for m, r in raw_reports.items()}
            rankings["anls"] = sorted(
                [{"model": m, "score": d["anls"]} for m, d in anls_data.items()],
                key=lambda x: x["score"], reverse=True,
            )

        # Summary table: all metrics per model (easy for frontend tables)
        summary_table = {}
        for mname in serialized:
            row = {"wfes": serialized[mname]["overall_weighted_score"]}
            sm = serialized[mname].get("standard_metrics", {})
            if "entity_f1" in sm:
                row["entity_f1"] = sm["entity_f1"]["f1"]
                row["entity_precision"] = sm["entity_f1"]["precision"]
                row["entity_recall"] = sm["entity_f1"]["recall"]
            if "anls" in sm:
                row["anls"] = sm["anls"]["score"]
            li = serialized[mname].get("line_item_metrics")
            if li:
                row["line_item_f1"] = li.get("item_f1", 0.0)
            summary_table[mname] = row

        return {
            "dataset_name": req.dataset_name,
            "num_models": len(serialized),
            "rankings": rankings,
            "winner": {
                metric: ranking[0]["model"] if ranking else None
                for metric, ranking in rankings.items()
            },
            "summary_table": summary_table,
            "field_comparison": field_comparison,
            "models": serialized,
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, detail=str(exc))


# 6. File-upload variants

@app.post("/evaluate/upload", summary="Evaluate via JSON file uploads")
async def evaluate_upload(
    ground_truth: UploadFile = File(...),
    predictions: UploadFile = File(...),
    model_name: str = Query("model"),
    dataset_name: str = Query("dataset"),
    tiers: Optional[str] = Query(None, description="Comma-separated tiers"),
    fields: Optional[str] = Query(None, description="Comma-separated field names"),
    use_embeddings: bool = Query(False),
    metrics: str = Query("all", description="Comma-separated: wfes,entity_f1,anls,all"),
):
    try:
        gt_docs = json.loads(await ground_truth.read())
        if isinstance(gt_docs, dict): gt_docs = [gt_docs]
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in ground_truth file: {e}")

    try:
        pred_docs = json.loads(await predictions.read())
        if isinstance(pred_docs, dict): pred_docs = [pred_docs]
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in predictions file: {e}")

    req = EvalRequest(
        ground_truth=gt_docs, predictions=pred_docs,
        model_name=model_name, dataset_name=dataset_name,
        tiers=[int(t) for t in tiers.split(",")] if tiers else None,
        fields=[f.strip() for f in fields.split(",")] if fields else None,
        use_embeddings=use_embeddings,
        metrics=[MetricType(m.strip()) for m in metrics.split(",")],
    )
    return evaluate_dataset(req)


@app.post("/compare/upload", summary="Compare 2+ models via file uploads")
async def compare_upload(
    ground_truth: UploadFile = File(...),
    predictions: List[UploadFile] = File(...),
    dataset_name: str = Query("dataset"),
    tiers: Optional[str] = Query(None),
    fields: Optional[str] = Query(None),
    use_embeddings: bool = Query(False),
    metrics: str = Query("all", description="Comma-separated: wfes,entity_f1,anls,all"),
):
    try:
        gt_docs = json.loads(await ground_truth.read())
        if isinstance(gt_docs, dict): gt_docs = [gt_docs]
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in ground_truth file: {e}")

    if len(predictions) < 2:
        raise HTTPException(400, "Upload at least 2 prediction files")

    preds_map: Dict[str, List[Dict[str, Any]]] = {}
    for pf in predictions:
        mname = pf.filename.rsplit(".", 1)[0] if pf.filename else f"model_{len(preds_map)}"
        try:
            docs = json.loads(await pf.read())
            if isinstance(docs, dict): docs = [docs]
            preds_map[mname] = docs
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON in '{pf.filename}': {e}")

    req = CompareRequest(
        ground_truth=gt_docs, predictions=preds_map,
        dataset_name=dataset_name,
        tiers=[int(t) for t in tiers.split(",")] if tiers else None,
        fields=[f.strip() for f in fields.split(",")] if fields else None,
        use_embeddings=use_embeddings,
        metrics=[MetricType(m.strip()) for m in metrics.split(",")],
    )
    return compare_models(req)


# 7. Health check

@app.get("/health")
def health():
    return {
        "status": "ok",
        "fields_registered": len(FIELD_DEFINITIONS),
        "available_metrics": ["wfes", "entity_f1", "anls"],
        "scoring_note": "True negatives excluded from all metrics",
    }