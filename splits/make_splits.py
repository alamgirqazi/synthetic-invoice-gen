#!/usr/bin/env python3
"""Document-level, template-stratified train/val/test split for InvoiceOCR-Synth.

All three image-quality variants share a file_name, so assigning file_names
keeps every variant of a base document in the same split.

Usage: python make_splits.py ground_truth_released.json -o splits.json
"""
import argparse, json, random
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("gt")
p.add_argument("-o", "--output", default="splits.json")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
a = p.parse_args()

records = json.load(open(a.gt))
by_template = defaultdict(list)
for r in records:
    by_template[r["_template_used"]].append(r["file_name"])

rng = random.Random(a.seed)
splits = {}
for template in sorted(by_template):
    names = sorted(by_template[template])
    rng.shuffle(names)
    n = len(names)
    n_val = round(n * a.ratios[1])
    n_test = round(n * a.ratios[2])
    for i, name in enumerate(names):
        splits[name] = "val" if i < n_val else "test" if i < n_val + n_test else "train"

json.dump(dict(sorted(splits.items())), open(a.output, "w"), indent=1)
