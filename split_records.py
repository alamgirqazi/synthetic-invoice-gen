#!/usr/bin/env python3
"""
split_records.py — Split synthetic_records.json into train/val/test.

Usage:
    python split_records.py -i synthetic_records.json --train 700 --val 100 --test 200 --seed 42

Output:
    splits/train.json  (700 records)
    splits/val.json    (100 records)
    splits/test.json   (200 records)
"""

import json
import os
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Split records into train/val/test")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output-dir", "-o", default="./splits", help="Output directory")
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        records = json.load(f)

    random.shuffle(records)

    total_needed = args.train + args.val + args.test
    if len(records) < total_needed:
        print(f"WARNING: {len(records)} records < {total_needed} requested")
        ratio = len(records) / total_needed
        args.train = int(args.train * ratio)
        args.val = int(args.val * ratio)
        args.test = len(records) - args.train - args.val

    splits = {
        "train": records[:args.train],
        "val": records[args.train:args.train + args.val],
        "test": records[args.train + args.val:args.train + args.val + args.test],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for name, recs in splits.items():
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        print(f"  {name}: {len(recs)} records → {path}")

    print(f"\nTotal: {sum(len(r) for r in splits.values())} records")


if __name__ == "__main__":
    main()
