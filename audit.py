#!/usr/bin/env python3
"""
Audit ground-truth.json for the InvoiceOCR-Synth corpus.

Produces the numbers a Data in Brief reviewer will ask for:
  - distribution counts (template, font, category, currency, type, locale)
  - uniqueness / repetition statistics for generated content
  - arithmetic verification of every identity the paper claims
  - field sparsity (null rates), supporting the "nulls are ground truth" claim
  - degenerate records (empty line items, zero totals)

Standard library only.

Usage:
    python audit_ground_truth.py ground-truth.json
    python audit_ground_truth.py ground-truth.json --tol 0.02 --examples 5
"""

import argparse
import json
import sys
from collections import Counter, defaultdict

MONEY_FIELDS = ("document_total_amount", "document_total_net", "document_total_tax")

SCHEMA_FIELDS = [
    "supplier_name", "supplier_address", "supplier_phone", "supplier_email",
    "supplier_website", "supplier_company_registrations",
    "customer_name", "customer_id", "customer_address", "billing_address",
    "shipping_address", "customer_company_registrations",
    "document_type_extended", "document_number", "invoice_number",
    "receipt_number", "reference_numbers", "po_number",
    "document_date", "document_time", "due_date", "payment_date",
    "document_currency", "document_total_amount", "document_total_net",
    "document_total_tax",
    "document_category", "subcategory", "locale",
    "line_items", "taxes", "notes",
]


def parse_rate(raw):
    """'23%' -> 23.0 ; None/'' -> None"""
    if raw is None:
        return None
    s = str(raw).strip().rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def num(v):
    return None if v is None else float(v)


def close(a, b, tol):
    if a is None or b is None:
        return None
    return abs(a - b) <= tol


def pct(n, total):
    return f"{n} ({100.0 * n / total:.1f}%)" if total else "0"


def header(title):
    print()
    print(title)
    print("-" * len(title))


def section_counts(records, tol, examples):
    header("DISTRIBUTIONS")

    def show(label, key):
            c = Counter()
            for r in records:
                v = r.get(key)


    show("Template (_template_used)", "_template_used")
    show("Font (_font_info)", "_font_info")
    show("Currency", "document_currency")
    show("Locale field", "locale")
    show("Category", "document_category")
    show("Document type (document_type_extended)", "document_type_extended")

    # subcategory is usually high-cardinality: top 15 only
    c = Counter(r.get("subcategory") or "<null>" for r in records)
    print(f"\nSubcategory ({len(c)} distinct, top 15)")
    for val, n in c.most_common(15):
        print(f"  {str(val)[:45]:<45} {pct(n, len(records))}")


def section_arrays(records):
    header("ARRAY FIELDS")
    n = len(records)
    li_counts = [len(r.get("line_items") or []) for r in records]
    tx_counts = [len(r.get("taxes") or []) for r in records]
    notes_filled = sum(1 for r in records if (r.get("notes") or "").strip())

    print(f"Line items   total={sum(li_counts)}  mean={sum(li_counts)/n:.2f}  "
          f"min={min(li_counts)}  max={max(li_counts)}  empty={pct(li_counts.count(0), n)}")
    print(f"Tax entries  total={sum(tx_counts)}  mean={sum(tx_counts)/n:.2f}  "
          f"min={min(tx_counts)}  max={max(tx_counts)}  empty={pct(tx_counts.count(0), n)}")
    print(f"Notes populated: {pct(notes_filled, n)}")
    print("\n-> 'Avg. line items per document' and 'Avg. tax entries per document' "
          "for Table 1 are the means above.")

    dist = Counter(li_counts)
    print("\nLine-item count distribution")
    for k in sorted(dist):
        print(f"  {k:>3} items  {pct(dist[k], n)}")


def section_uniqueness(records, examples):
    header("CONTENT DIVERSITY")
    n = len(records)

    def uniq(label, key):
        vals = [r.get(key) for r in records if r.get(key)]
        c = Counter(vals)
        if not c:
            print(f"\n{label}: no non-null values")
            return
        top = c.most_common(examples)
        print(f"\n{label}: {len(c)} unique / {len(vals)} non-null "
              f"({100.0*len(c)/len(vals):.1f}% unique)")
        for val, k in top:
            print(f"    {k:>4}x  {str(val)[:60]}")

    uniq("Supplier names", "supplier_name")
    uniq("Customer names", "customer_name")
    uniq("Supplier addresses", "supplier_address")
    uniq("Supplier phones", "supplier_phone")
    uniq("Supplier registrations", "supplier_company_registrations")

    descs = [li.get("description") for r in records
             for li in (r.get("line_items") or []) if li.get("description")]
    c = Counter(descs)
    print(f"\nLine-item descriptions: {len(c)} unique / {len(descs)} total "
          f"({100.0*len(c)/len(descs):.1f}% unique)" if descs else "\nNo descriptions")
    for val, k in c.most_common(examples):
        print(f"    {k:>4}x  {str(val)[:60]}")

    # cross-region contamination: registration prefix vs currency
    print("\nRegistration-prefix vs currency mismatches (heuristic)")
    expect = {"EUR": ("IE", "DE", "FR", "ES", "IT"), "GBP": ("GB",),
              "USD": ("US", "CA-", "EIN"), "PLN": ("PL",)}
    hits = []
    for i, r in enumerate(records):
        reg, cur = r.get("supplier_company_registrations"), r.get("document_currency")
        if not reg or cur not in expect:
            continue
        prefix = str(reg).strip()[:2].upper()
        if prefix.isalpha() and not any(prefix == e[:2] for e in expect[cur]):
            hits.append((i, cur, reg))
    print(f"  {len(hits)} record(s) flagged")
    for i, cur, reg in hits[:examples]:
        print(f"    #{i}  currency={cur}  registration={reg}")


def section_arithmetic(records, tol, examples):
    header(f"ARITHMETIC VERIFICATION (tolerance = {tol})")
    n = len(records)
    fails = defaultdict(list)

    for i, r in enumerate(records):
        idx = f"#{i} {r.get('file_name', '?')}"
        items = r.get("line_items") or []
        taxes = r.get("taxes") or []

        net = num(r.get("document_total_net"))
        tax = num(r.get("document_total_tax"))
        gross = num(r.get("document_total_amount"))

        # 1. per line: quantity * unit_price == amount
        for j, li in enumerate(items):
            q, up, amt = num(li.get("quantity")), num(li.get("unit_price")), num(li.get("amount"))
            if q is not None and up is not None and amt is not None:
                if not close(q * up, amt, tol):
                    fails["line: quantity x unit_price != amount"].append(
                        f"{idx} line{j}: {q} x {up} = {q*up:.4f} vs {amt}")

        # 2. per line: amount * rate == tax_amount
        for j, li in enumerate(items):
            amt, ta = num(li.get("amount")), num(li.get("tax_amount"))
            rate = parse_rate(li.get("tax_rate"))
            if None not in (amt, ta, rate):
                if not close(amt * rate / 100.0, ta, tol):
                    fails["line: amount x rate != tax_amount"].append(
                        f"{idx} line{j}: {amt} x {rate}% = {amt*rate/100:.4f} vs {ta}")

        # 3. sum(line amounts) == net
        if items and net is not None:
            s = sum(num(li.get("amount")) or 0.0 for li in items)
            if not close(s, net, tol):
                fails["sum(line amounts) != document_total_net"].append(
                    f"{idx}: {s:.2f} vs {net}")

        # 4. sum(line tax) == total tax
        if items and tax is not None:
            s = sum(num(li.get("tax_amount")) or 0.0 for li in items)
            if not close(s, tax, tol):
                fails["sum(line tax_amount) != document_total_tax"].append(
                    f"{idx}: {s:.2f} vs {tax}")

        # 5. net + tax == gross
        if None not in (net, tax, gross):
            if not close(net + tax, gross, tol):
                fails["net + tax != document_total_amount"].append(
                    f"{idx}: {net} + {tax} = {net+tax:.2f} vs {gross}")

        # 6. taxes array consistency
        if taxes:
            sb = sum(num(t.get("base")) or 0.0 for t in taxes)
            sv = sum(num(t.get("value")) or 0.0 for t in taxes)
            if net is not None and not close(sb, net, tol):
                fails["sum(taxes.base) != document_total_net"].append(
                    f"{idx}: {sb:.2f} vs {net}")
            if tax is not None and not close(sv, tax, tol):
                fails["sum(taxes.value) != document_total_tax"].append(
                    f"{idx}: {sv:.2f} vs {tax}")
            for j, t in enumerate(taxes):
                b, v, rate = num(t.get("base")), num(t.get("value")), parse_rate(t.get("rate"))
                if None not in (b, v, rate) and not close(b * rate / 100.0, v, tol):
                    fails["taxes: base x rate != value"].append(
                        f"{idx} tax{j}: {b} x {rate}% = {b*rate/100:.4f} vs {v}")

    total_fail = sum(len(v) for v in fails.values())
    if not total_fail:
        print(f"All identities hold across {n} records.")
    else:
        print(f"{total_fail} violation(s) across {n} records:\n")
        for k in sorted(fails, key=lambda x: -len(fails[x])):
            print(f"  {k}: {len(fails[k])}")
            for ex in fails[k][:examples]:
                print(f"      {ex}")

    records_clean = n - len({f.split(':')[0] for v in fails.values() for f in v})
    print(f"\nRecords with zero violations: {pct(records_clean, n)}")
    print("-> This is the figure for Table 1 'Arithmetic consistency of labels'.")


def section_sparsity(records):
    header("FIELD SPARSITY (null / empty rate)")
    n = len(records)
    rows = []
    for f in SCHEMA_FIELDS:
        empty = 0
        for r in records:
            v = r.get(f)
            if v is None or v == "" or v == []:
                empty += 1
        rows.append((f, empty))
    for f, empty in sorted(rows, key=lambda x: -x[1]):
        print(f"  {f:<38} {pct(empty, n)}")


def section_integrity(records, examples):
    header("INTEGRITY")
    n = len(records)

    missing = [f for f in SCHEMA_FIELDS if not any(f in r for r in records)]
    print(f"Schema fields present: {len(SCHEMA_FIELDS) - len(missing)}/{len(SCHEMA_FIELDS)}")
    if missing:
        print(f"  MISSING: {', '.join(missing)}")

    names = Counter(r.get("file_name") for r in records)
    dupes = [k for k, v in names.items() if v > 1]
    print(f"Duplicate file_name values: {len(dupes)}")
    for d in dupes[:examples]:
        print(f"    {d}")

    degenerate = [(i, r.get("file_name")) for i, r in enumerate(records)
                  if not (r.get("line_items") or [])
                  or (num(r.get("document_total_amount")) or 0) == 0]
    print(f"Degenerate records (no line items or zero total): {pct(len(degenerate), n)}")
    for i, fn in degenerate[:examples]:
        print(f"    #{i} {fn}")

    bad_variants = []
    for i, r in enumerate(records):
        dv = r.get("degradation_variants") or {}
        if dv and len(set(dv.values())) != 1:
            bad_variants.append((i, dv))
    print(f"Records where the 3 variants disagree on filename: {len(bad_variants)}")
    for i, dv in bad_variants[:examples]:
        print(f"    #{i} {dv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to ground-truth.json")
    ap.add_argument("--tol", type=float, default=0.011,
                    help="currency tolerance; 0.011 absorbs single-cent rounding")
    ap.add_argument("--examples", type=int, default=10,
                    help="how many examples to print per finding")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        sys.exit("Expected a JSON array of records.")

    print(f"Loaded {len(records)} records from {args.path}")

    section_integrity(records, args.examples)
    section_counts(records, args.tol, args.examples)
    section_arrays(records)
    section_uniqueness(records, args.examples)
    section_arithmetic(records, args.tol, args.examples)
    section_sparsity(records)

    print("\nDone.")


if __name__ == "__main__":
    main()