#!/usr/bin/env python3

import json
import os
import shutil
import argparse
from pathlib import Path


def build_metadata_jsonl(labels_path: str, output_path: str):
    """Convert labels.json → metadata.jsonl for HuggingFace ImageFolder."""
    with open(labels_path) as f:
        labels = json.load(f)

    lines = []
    for label in labels:
        base_id = label["file_name"].replace(".png", "")
        variants = label.get("degradation_variants", {})
        if not variants:
            variants = {"clean": label["file_name"]}

        for degradation, filename in variants.items():
            row = {
                "file_name": f"{degradation}/{filename}",
                "degradation": degradation,
                "base_id": base_id,
                "supplier_name": label.get("supplier_name"),
                "document_date": label.get("document_date"),
                "document_currency": label.get("document_currency"),
                "document_total_amount": label.get("document_total_amount"),
                "document_total_net": label.get("document_total_net"),
                "document_total_tax": label.get("document_total_tax"),
                "document_category": label.get("document_category"),
                "subcategory": label.get("subcategory"),
                "document_type_extended": label.get("document_type_extended"),
                "invoice_number": label.get("invoice_number"),
                "receipt_number": label.get("receipt_number"),
                "customer_name": label.get("customer_name"),
                "supplier_address": label.get("supplier_address"),
                "supplier_phone": label.get("supplier_phone"),
                "line_items": json.dumps(label.get("line_items", []),
                                          ensure_ascii=False),
                "taxes": json.dumps(label.get("taxes", []),
                                     ensure_ascii=False),
                "template_used": label.get("_template_used"),
                "num_line_items": len(label.get("line_items", [])),
            }
            lines.append(json.dumps(row, ensure_ascii=False))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {len(lines)} rows → {output_path}")


def prepare_hf_repo(rendered_dir: str, hf_dir: str, presets: list):
    """Reorganize rendered output into HuggingFace-compatible structure."""
    os.makedirs(hf_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_src = os.path.join(rendered_dir, split)
        split_dst = os.path.join(hf_dir, split)

        if not os.path.exists(split_src):
            print(f"  SKIP {split} (not found at {split_src})")
            continue

        print(f"\n  Processing {split}:")
        images_dir = os.path.join(split_src, "images")
        for variant in presets:
            src = os.path.join(images_dir, variant)
            dst = os.path.join(split_dst, variant)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"    {variant}: {len(os.listdir(dst))} images")
            else:
                print(f"    {variant}: NOT FOUND at {src}")

        labels_path = os.path.join(split_src, "labels.json")
        if os.path.exists(labels_path):
            metadata_path = os.path.join(split_dst, "metadata.jsonl")
            build_metadata_jsonl(labels_path, metadata_path)


def verify_dataset(hf_dir: str):
    """Quick verification that the dataset loads."""
    try:
        from datasets import load_dataset
        ds = load_dataset("imagefolder", data_dir=hf_dir)
        print("\nVerification:")
        for split, data in ds.items():
            print(f"  {split}: {len(data)} rows, columns: {data.column_names}")
        print("  ✓ Dataset loads correctly")
    except Exception as e:
        print(f"\n  ⚠ Verification failed: {e}")
        print("  (This is OK — you can still upload manually)")


def upload(hf_dir: str, repo_id: str):
    """Upload to HuggingFace Hub."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=hf_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data",
    )
    print(f"\n✓ Uploaded to https://huggingface.co/datasets/{repo_id}")
    print("  Remember to upload README.md via the HF web interface!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered-dir", default="./rendered")
    parser.add_argument("--hf-dir", default="./hf_data")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--presets", nargs="+", default=["clean", "bad_scan", "faded"])
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing SynthDegradeBench for HuggingFace")
    print("=" * 60)

    prepare_hf_repo(args.rendered_dir, args.hf_dir, args.presets)

    if not args.skip_verify:
        verify_dataset(args.hf_dir)

    if not args.skip_upload:
        upload(args.hf_dir, args.repo_id)
