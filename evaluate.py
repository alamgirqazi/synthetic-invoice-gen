#!/usr/bin/env python3
import json
import argparse
import re
import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional


STRING_FIELDS = [
    "supplier_name", "document_date", "document_currency",
    "document_category", "invoice_number", "receipt_number",
    "customer_name",
]

NUMERIC_FIELDS = [
    "document_total_amount", "document_total_net", "document_total_tax",
]

NUMERIC_TOLERANCE = 0.02


def normalize_string(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s


def normalize_date(s: Optional[str]) -> str:
    if not s:
        return ""
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
                "%d %B %Y", "%B %d, %Y", "%d %b %Y"]:
        try:
            return datetime.datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s.strip()


def match_string(pred: Optional[str], gold: Optional[str], field: str) -> bool:
    if gold is None or gold == "":
        return True
    if field == "document_date":
        return normalize_date(pred) == normalize_date(gold)
    return normalize_string(pred) == normalize_string(gold)


def match_numeric(pred, gold, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    if gold is None:
        return True
    try:
        return abs(float(pred or 0) - float(gold)) <= tolerance
    except (ValueError, TypeError):
        return False


def compute_line_item_f1(pred_items: List[Dict], gold_items: List[Dict]) -> Dict:
    if not gold_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gold = set()
    matched_pred = set()

    for pi, p in enumerate(pred_items):
        for gi, g in enumerate(gold_items):
            if gi in matched_gold:
                continue
            p_desc = normalize_string(p.get("description", ""))
            g_desc = normalize_string(g.get("description", ""))
            desc_match = (p_desc == g_desc or
                          (p_desc in g_desc or g_desc in p_desc) and len(p_desc) > 3)
            amount_match = match_numeric(p.get("amount"), g.get("amount"), 0.05)
            if desc_match and amount_match:
                matched_gold.add(gi)
                matched_pred.add(pi)
                break

    prec = len(matched_pred) / len(pred_items) if pred_items else 0.0
    rec = len(matched_gold) / len(gold_items) if gold_items else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    gt_index = {}
    for gt in ground_truth:
        fname = gt.get("file_name", "")
        gt_index[fname] = gt

    results = defaultdict(lambda: {
        "str_ok": defaultdict(int), "str_n": defaultdict(int),
        "num_ok": defaultdict(int), "num_n": defaultdict(int),
        "li_f1s": [], "count": 0,
    })

    for pred in predictions:
        fname = pred.get("file_name", "")
        gt = gt_index.get(fname)
        if not gt:
            continue

        deg = gt.get("degradation", "unknown")
        b = results[deg]
        b["count"] += 1

        for f in STRING_FIELDS:
            gv = gt.get(f)
            if gv not in (None, ""):
                b["str_n"][f] += 1
                if match_string(pred.get(f), gv, f):
                    b["str_ok"][f] += 1

        for f in NUMERIC_FIELDS:
            gv = gt.get(f)
            if gv is not None:
                b["num_n"][f] += 1
                if match_numeric(pred.get(f), gv):
                    b["num_ok"][f] += 1

        pi = pred.get("line_items", [])
        gi = gt.get("line_items", [])
        if isinstance(pi, str):
            pi = json.loads(pi)
        if isinstance(gi, str):
            gi = json.loads(gi)
        b["li_f1s"].append(compute_line_item_f1(pi, gi)["f1"])

    report = {}
    for deg, b in sorted(results.items()):
        r = {"count": b["count"]}

        for f in STRING_FIELDS:
            n = b["str_n"][f]
            r[f"{f}_acc"] = b["str_ok"][f] / n if n > 0 else None

        for f in NUMERIC_FIELDS:
            n = b["num_n"][f]
            r[f"{f}_acc"] = b["num_ok"][f] / n if n > 0 else None

        accs = [r[f"{f}_acc"] for f in STRING_FIELDS + NUMERIC_FIELDS
                if r.get(f"{f}_acc") is not None]
        r["field_accuracy_mean"] = sum(accs) / len(accs) if accs else 0.0

        num_accs = [r[f"{f}_acc"] for f in NUMERIC_FIELDS
                    if r.get(f"{f}_acc") is not None]
        r["amount_accuracy"] = sum(num_accs) / len(num_accs) if num_accs else 0.0

        f1s = b["li_f1s"]
        r["line_item_f1"] = sum(f1s) / len(f1s) if f1s else 0.0

        report[deg] = r

    return report


def print_report(report: Dict, model_name: str = "Model"):
    degs = sorted(report.keys())

    print(f"\n{'=' * 72}")
    print(f"  SynthDegradeBench Results — {model_name}")
    print(f"{'=' * 72}")

    header = f"{'Metric':<25s}"
    for d in degs:
        header += f" {d:>10s}"
    if "clean" in report and len(degs) > 1:
        for d in degs:
            if d != "clean":
                header += f" {'Δ→' + d:>12s}"
    print(header)
    print("-" * len(header))

    for m in ["field_accuracy_mean", "amount_accuracy", "line_item_f1"]:
        row = f"{m:<25s}"
        cv = report.get("clean", {}).get(m, 0)
        for d in degs:
            row += f" {report[d].get(m, 0):>10.3f}"
        if "clean" in report and len(degs) > 1:
            for d in degs:
                if d != "clean":
                    delta = report[d].get(m, 0) - cv
                    row += f" {delta:>+12.3f}"
        print(row)

    row = f"{'n_images':<25s}"
    for d in degs:
        row += f" {report[d]['count']:>10d}"
    print(row)
    print(f"{'=' * 72}\n")

    print("Per-field breakdown:")
    for f in STRING_FIELDS + NUMERIC_FIELDS:
        row = f"  {f:<30s}"
        for d in degs:
            v = report[d].get(f"{f}_acc")
            row += f" {v:>8.3f}" if v is not None else f" {'n/a':>8s}"
        print(row)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against SynthDegradeBench"
    )
    parser.add_argument("--predictions", "-p", required=True)
    parser.add_argument("--ground-truth", "-g", required=True)
    parser.add_argument("--model-name", default="Model")
    args = parser.parse_args()

    def load_json_or_jsonl(path):
        with open(path) as f:
            content = f.read().strip()
            if content.startswith("["):
                return json.loads(content)
            return [json.loads(l) for l in content.split("\n") if l.strip()]

    predictions = load_json_or_jsonl(args.predictions)
    ground_truth = load_json_or_jsonl(args.ground_truth)

    report = evaluate(predictions, ground_truth)
    print_report(report, args.model_name)

    out = args.predictions.replace(".json", "_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out}")
