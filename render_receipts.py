#!/usr/bin/env python3

import json
import os
import random
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import BytesIO

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from PIL import Image
import numpy as np

# Import the new augmentation module
from augmentation import DocumentAugmentor, HAS_AUGRAPHY

TEMPLATE_RULES = {
    "formal_corporate": {
        "file": "formal_corporate.html",
        "match_categories": ["rent", "utilities", "services", "software", "office_supplies"],
        "match_doc_types": ["tax_invoice", "sales_invoice", "proforma_invoice"],
        "match_layout": ["formal_corporate", "traditional_tabular"],
    },
    "freelancer_invoice": {
        "file": "freelancer_invoice.html",
        "match_categories": ["services"],
        "match_doc_types": ["freelancer_invoice", "consulting_invoice"],
        "match_layout": ["freelancer"],
        "match_subcategories": ["consulting", "development", "design", "freelance",
                                "coaching", "writing", "translation"],
    },
    "pos_thermal": {
        "file": "pos_thermal.html",
        "match_categories": ["food"],
        "match_doc_types": ["restaurant_receipt"],
        "match_layout": ["pos_thermal"],
        "match_subcategories": ["restaurant", "cafe", "bar", "pub", "bistro",
                                "fast_food", "takeaway", "bakery", "deli"],
    },
    "retail_receipt": {
        "file": "retail_receipt.html",
        "match_categories": ["retail", "electronics", "office_supplies"],
        "match_doc_types": ["receipt"],
        "match_layout": ["retail_receipt", "retail_receipt"],
        "match_subcategories": [
            "grocery", "supermarket", "pharmacy", "shop", "convenience",
            "hardware", "electronics", "department_store", "bookshop",
            "clothing", "sports", "home_improvement", "pet_store",
            "toy_store", "stationery",
        ],
    },

    "modern_minimal": {
        "file": "modern_minimal.html",
        "match_categories": ["software", "services", "electronics", "shipping"],
        "match_doc_types": ["subscription_invoice", "tax_invoice"],
        "match_layout": ["modern_minimal"],
    },
    "event_ticket": {
        "file": "event_ticket.html",
        "match_categories": ["miscellaneous"],
        "match_doc_types": ["event_ticket", "order_confirmation"],
        "match_layout": ["event_ticket"],
        "match_subcategories": ["event", "ticket", "concert", "conference",
                                "exhibition", "festival", "theatre"],
    },
    "utility_bill": {
        "file": "utility_bill.html",
        "match_categories": ["utilities"],
        "match_doc_types": ["utility_bill", "credit_note"],
        "match_layout": ["utility_bill"],
        "match_subcategories": ["electricity", "gas", "water", "broadband",
                                "phone", "internet", "waste"],
    },
    "delivery_note": {
        "file": "delivery_note.html",
        "match_categories": ["shipping"],
        "match_doc_types": ["delivery_note"],
        "match_layout": ["delivery_note"],
        "match_subcategories": ["delivery", "shipping", "logistics",
                                "warehouse", "fulfilment"],
    },
    "taxi_receipt": {
        "file": "taxi_receipt.html",
        "match_categories": ["transport"],
        "match_doc_types": ["taxi_receipt"],
        "match_layout": ["taxi_receipt"],
        "match_subcategories": ["taxi", "uber", "ride", "rideshare",
                                "cab", "transfer"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# FONT RANDOMIZATION
# ═══════════════════════════════════════════════════════════════════════

FONT_POOLS = {
    "sans": [
        ("'Roboto', sans-serif", 3),
        ("'Open Sans', sans-serif", 3),
        ("'Lato', sans-serif", 3),
        ("'Noto Sans', sans-serif", 2),
        ("'Liberation Sans', sans-serif", 2),
        ("'DejaVu Sans', sans-serif", 2),
        ("'Ubuntu', sans-serif", 1),
        ("'FreeSans', sans-serif", 1),
        ("'Carlito', sans-serif", 1),
    ],
    "serif": [
        ("'Liberation Serif', serif", 3),
        ("'DejaVu Serif', serif", 3),
        ("'FreeSerif', serif", 2),
        ("'Bitstream Charter', serif", 2),
        ("'Caladea', serif", 1),
    ],
    "mono": [
        ("'Courier 10 Pitch', 'Courier New', monospace", 3),
        ("'DejaVu Sans Mono', monospace", 3),
        ("'Liberation Mono', monospace", 2),
        ("'Ubuntu Mono', monospace", 2),
        ("'FreeMono', monospace", 1),
    ],
}

TEMPLATE_FONT_MAP = {
    "formal_corporate": {"body": "sans", "heading": "sans"},
    "freelancer_invoice": {"body": "sans", "heading": "sans"},
    "modern_minimal": {"body": "sans", "heading": "sans"},
    "utility_bill": {"body": "sans", "heading": "sans"},
    "event_ticket": {"body": "sans", "heading": "sans"},
    "delivery_note": {"body": "sans", "heading": "sans"},
    "pos_thermal": {"body": "mono", "heading": "mono"},
    "retail_receipt": {"body": "mono", "heading": "mono"},
    "taxi_receipt": {"body": "sans", "heading": "sans"},
}


def pick_font(pool_name: str) -> str:
    pool = FONT_POOLS.get(pool_name, FONT_POOLS["sans"])
    fonts, weights = zip(*pool)
    return random.choices(fonts, weights=weights, k=1)[0]


def get_font_css_override(template_name: str) -> tuple:
    font_map = TEMPLATE_FONT_MAP.get(template_name, {"body": "sans", "heading": "sans"})
    body_pool = font_map["body"]
    heading_pool = font_map["heading"]

    # 15% serif swap for non-mono templates
    if body_pool == "sans" and random.random() < 0.15:
        body_pool = "serif"
        heading_pool = "serif"

    body_font = pick_font(body_pool)
    heading_font = pick_font(heading_pool)

    if body_pool != "mono" and random.random() < 0.30:
        heading_font = pick_font("sans")

    css = f"""<style>
  body, td, th, div, span, p {{ font-family: {body_font} !important; }}
  h1, h2, h3, .brand-name, .logo-text, .store-name, .doc-title, .hotel-name, .company {{ font-family: {heading_font} !important; }}
</style>"""

    return css, body_font, heading_font


def select_template(record: Dict[str, Any]) -> str:
    """Intelligently select the best template for a record."""
    hints = record.get("_template_hints", {})
    layout_hint = hints.get("layout_style", "")
    doc_type = (record.get("document_type_extended") or "").lower().replace(" ", "_")
    category = (record.get("document_category") or "").lower()
    subcategory = (record.get("subcategory") or "").lower()

    # 1. Layout hint — exact match
    for name, rule in TEMPLATE_RULES.items():
        if layout_hint in rule.get("match_layout", []):
            return name

    # 2. Document type
    for name, rule in TEMPLATE_RULES.items():
        if doc_type in rule.get("match_doc_types", []):
            return name

    # 3. Subcategory
    for name, rule in TEMPLATE_RULES.items():
        if subcategory in rule.get("match_subcategories", []):
            return name

    # 4. Category
    for name, rule in TEMPLATE_RULES.items():
        if category in rule.get("match_categories", []):
            return name

    # 5. Weighted fallback — includes retail_receipt_v2
    templates = [
        "formal_corporate", "freelancer_invoice", "pos_thermal",
        "retail_receipt_v2", "modern_minimal", "event_ticket",
        "utility_bill", "delivery_note", "taxi_receipt",
    ]
    weights = [0.18, 0.12, 0.12, 0.15, 0.15, 0.06, 0.08, 0.06, 0.08]
    return random.choices(templates, weights=weights, k=1)[0]

CURRENCY_SYMBOLS = {
    "EUR": "€", "USD": "$", "GBP": "£", "PLN": "zł",
    "CHF": "CHF ", "SEK": "kr", "NOK": "kr", "DKK": "kr",
    "CZK": "Kč", "JPY": "¥", "CNY": "¥", "AUD": "A$", "CAD": "C$",
}

COLOR_SCHEMES = {
    "blue_corporate": {"primary": "#2c5aa0", "secondary": "#4a90d9"},
    "green_eco": {"primary": "#27ae60", "secondary": "#2ecc71"},
    "red_accent": {"primary": "#c0392b", "secondary": "#e74c3c"},
    "monochrome": {"primary": "#333333", "secondary": "#666666"},
    "teal_modern": {"primary": "#00b4d8", "secondary": "#48cae4"},
    "purple_creative": {"primary": "#6c5ce7", "secondary": "#a29bfe"},
    "orange_warm": {"primary": "#e17055", "secondary": "#fab1a0"},
    "navy_formal": {"primary": "#2c3e50", "secondary": "#34495e"},
    "dark_green": {"primary": "#1a5632", "secondary": "#27ae60"},
    "burgundy": {"primary": "#7b2d3b", "secondary": "#b03a4e"},
}

PAYMENT_METHODS = [
    "Visa", "MasterCard", "Amex", "Bank Transfer",
    "PayPal", "Card", "Direct Debit", "Apple Pay",
    "Google Pay", "Revolut",
]


def format_date(date_str: str, style: str = "dd/mm/yyyy") -> str:
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        formats = {
            "dd/mm/yyyy": "%d/%m/%Y",
            "d_month_yyyy": "%-d %B %Y",
            "mm/dd/yyyy": "%m/%d/%Y",
            "yyyy-mm-dd": "%Y-%m-%d",
        }
        return dt.strftime(formats.get(style, "%d/%m/%Y"))
    except (ValueError, TypeError):
        return date_str or ""


def prepare_template_context(record: Dict[str, Any], template_name: str) -> Dict[str, Any]:
    ctx = dict(record)

    currency = record.get("document_currency", "EUR") or "EUR"
    ctx["currency_symbol"] = CURRENCY_SYMBOLS.get(currency, currency + " ")

    locale = (record.get("locale") or "en").lower()
    region = record.get("_generation_params", {}).get("region", "IE")
    date_style = "mm/dd/yyyy" if region == "US" else "dd/mm/yyyy"

    ctx["document_date_formatted"] = format_date(record.get("document_date"), date_style)
    ctx["due_date_formatted"] = format_date(record.get("due_date"), date_style)
    ctx["payment_date_formatted"] = format_date(record.get("payment_date"), date_style)

    try:
        dt = datetime.strptime(record.get("document_date", ""), "%Y-%m-%d")
        ctx["document_date_display"] = dt.strftime("%-d %B %Y")
    except (ValueError, TypeError):
        ctx["document_date_display"] = ctx["document_date_formatted"]

    hints = record.get("_template_hints", {})
    color_name = hints.get("color_scheme", random.choice(list(COLOR_SCHEMES.keys())))
    colors = COLOR_SCHEMES.get(color_name, COLOR_SCHEMES["blue_corporate"])
    ctx["color_primary"] = colors["primary"]
    ctx["color_secondary"] = colors["secondary"]

    ctx["payment_method"] = random.choice(PAYMENT_METHODS)

    if not isinstance(ctx.get("line_items"), list):
        ctx["line_items"] = []
    if not isinstance(ctx.get("taxes"), list):
        ctx["taxes"] = []

    for field in ["document_total_net", "document_total_tax", "document_total_amount"]:
        try:
            ctx[field] = float(ctx.get(field, 0) or 0)
        except (ValueError, TypeError):
            ctx[field] = 0.0

    for item in ctx["line_items"]:
        for nf in ["quantity", "unit_price", "amount", "tax_amount"]:
            try:
                item[nf] = float(item.get(nf, 0) or 0)
            except (ValueError, TypeError):
                item[nf] = 0.0

    for tax in ctx["taxes"]:
        for nf in ["base", "value"]:
            try:
                tax[nf] = float(tax.get(nf, 0) or 0)
            except (ValueError, TypeError):
                tax[nf] = 0.0

    # Payment details for some templates
    if template_name in ("formal_corporate", "utility_bill", "modern_minimal",
                         "delivery_note"):
        if random.random() > 0.4:
            ctx["payment_details"] = _generate_payment_details(region)
        else:
            ctx["payment_details"] = None
    else:
        ctx["payment_details"] = None

    ctx["max"] = max
    ctx["random"] = random.random

    return ctx


def _generate_payment_details(region: str) -> Dict[str, str]:
    r = random.randint(1000, 9999)
    if region == "IE":
        return {
            "BIC": f"BOFI{random.choice(['IE', 'XX'])}2D",
            "IBAN": f"IE{random.randint(10,99)}BOFI{random.randint(10000000,99999999)}{random.randint(100000,999999)}",
            "Account Name": "Acme Payments Ltd",
        }
    elif region == "GB":
        return {
            "Sort Code": f"{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}",
            "Account No": f"{random.randint(10000000,99999999)}",
        }
    elif region == "DE":
        return {
            "BIC": f"DEUT{random.choice(['DE', 'FF'])}XXX",
            "IBAN": f"DE{random.randint(10,99)}{random.randint(10000000,99999999)}{random.randint(10000000,99999999)}",
        }
    else:
        return {"Bank": f"National Bank #{r}"}


# ═══════════════════════════════════════════════════════════════════════
# GROUND TRUTH LABELS
# ═══════════════════════════════════════════════════════════════════════

def build_ground_truth_label(
    record: Dict[str, Any],
    file_name: str,
    mime_type: str,
) -> Dict[str, Any]:
    """Build ground-truth label matching OCR pipeline schema."""
    label = {
        "file_name": file_name,
        "supplier_name": record.get("supplier_name"),
        "document_date": record.get("document_date"),
        "document_time": record.get("document_time"),
        "document_currency": record.get("document_currency"),
        "document_total_amount": float(record.get("document_total_amount", 0) or 0),
        "document_total_net": float(record.get("document_total_net", 0) or 0),
        "document_total_tax": float(record.get("document_total_tax", 0) or 0),
        "document_category": record.get("document_category", "other"),
        "subcategory": record.get("subcategory"),
        "due_date": record.get("due_date"),
        "payment_date": record.get("payment_date"),
        "document_type_extended": record.get("document_type_extended"),
        "document_number": record.get("document_number"),
        "receipt_number": record.get("receipt_number"),
        "invoice_number": record.get("invoice_number"),
        "reference_numbers": record.get("reference_numbers"),
        "po_number": record.get("po_number"),
        "locale": record.get("locale"),
        "customer_name": record.get("customer_name"),
        "customer_id": record.get("customer_id"),
        "customer_address": record.get("customer_address"),
        "billing_address": record.get("billing_address"),
        "shipping_address": record.get("shipping_address"),
        "customer_company_registrations": record.get("customer_company_registrations"),
        "supplier_address": record.get("supplier_address"),
        "supplier_phone": record.get("supplier_phone"),
        "supplier_email": record.get("supplier_email"),
        "supplier_website": record.get("supplier_website"),
        "supplier_company_registrations": record.get("supplier_company_registrations"),
        "line_items": [],
        "taxes": [],
        "notes": record.get("notes", ""),
        "mime_type": mime_type,
        "labeled_at": datetime.now().isoformat(),
    }

    for item in (record.get("line_items") or []):
        label["line_items"].append({
            "description": item.get("description", ""),
            "quantity": float(item.get("quantity", 1) or 1),
            "unit_price": float(item.get("unit_price", 0) or 0),
            "amount": float(item.get("amount", 0) or 0),
            "tax_rate": str(item.get("tax_rate", "")),
            "tax_amount": float(item.get("tax_amount", 0) or 0),
        })

    for tax in (record.get("taxes") or []):
        label["taxes"].append({
            "base": float(tax.get("base", 0) or 0),
            "rate": str(tax.get("rate", "")),
            "code": str(tax.get("code", "")),
            "value": float(tax.get("value", 0) or 0),
        })

    return label

class ReceiptRenderer:
    """Renders JSON records into PDF/PNG using HTML templates."""

    def __init__(self, templates_dir: str = "./templates"):
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=False,
        )
        self.env.filters["commaformat"] = lambda v: f"{v:,.2f}"
        self.templates_dir = templates_dir

        available = [name for name, rule in TEMPLATE_RULES.items()
                     if os.path.exists(os.path.join(templates_dir, rule["file"]))]
        print(f"Templates dir: {templates_dir}")
        print(f"Available templates ({len(available)}): {', '.join(available)}")

    def _render_html_to_image(
        self,
        record: Dict[str, Any],
        template_name: str,
        dpi: int,
        randomize_fonts: bool,
    ) -> tuple:
        """Render a record to a PIL Image (the 'clean' version). Returns (image, font_info)."""
        template = self.env.get_template(TEMPLATE_RULES[template_name]["file"])
        ctx = prepare_template_context(record, template_name)
        html_content = template.render(**ctx)

        font_info = ""
        if randomize_fonts:
            font_css, body_font, heading_font = get_font_css_override(template_name)
            if "</head>" in html_content:
                html_content = html_content.replace("</head>", f"{font_css}\n</head>")
            else:
                html_content = f"{font_css}\n{html_content}"
            font_info = body_font.split(",")[0].strip("' ")

        html_doc = HTML(string=html_content)
        pdf_bytes = html_doc.write_pdf()

        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        return img, font_info, pdf_bytes

    def render_multi_degrade(
        self,
        record: Dict[str, Any],
        output_dir: str,
        index: int,
        degradation_presets: List[str],
        dpi: int = 150,
        randomize_fonts: bool = True,
        save_pdf: bool = False,
    ) -> Dict[str, Any]:
        """
        Render a single record at multiple degradation levels.

        Returns ONE label dict with a `degradation_variants` field listing
        the file names per preset.
        """
        template_name = select_template(record)
        record_hash = hashlib.md5(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()[:8]
        base_name = f"synth_{index:04d}_{record_hash}"

        clean_img, font_info, pdf_bytes = self._render_html_to_image(
            record, template_name, dpi, randomize_fonts
        )

        if save_pdf:
            pdf_path = os.path.join(output_dir, "pdfs", f"{base_name}.pdf")
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

        variants = {}
        for preset_name in degradation_presets:
            preset_dir = os.path.join(output_dir, "images", preset_name)
            os.makedirs(preset_dir, exist_ok=True)
            png_path = os.path.join(preset_dir, f"{base_name}.png")

            if preset_name == "clean":
                clean_img.save(png_path, format="PNG")
            else:
                aug = DocumentAugmentor(preset=preset_name)
                degraded = aug(clean_img)
                degraded.save(png_path, format="PNG")

            variants[preset_name] = f"{base_name}.png"

        label = build_ground_truth_label(record, f"{base_name}.png", "image/png")
        label["_template_used"] = template_name
        label["_synthetic"] = True
        label["_font_info"] = font_info or None
        label["degradation_variants"] = variants

        return label

    def render_single(
        self,
        record: Dict[str, Any],
        output_dir: str,
        index: int,
        output_format: str = "both",
        noise_preset: Optional[str] = None,
        dpi: int = 150,
        randomize_fonts: bool = True,
    ) -> Dict[str, Any]:
        """Render a single record (classic mode — one output per record)."""
        template_name = select_template(record)
        record_hash = hashlib.md5(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()[:8]
        base_name = f"synth_{index:04d}_{record_hash}"

        clean_img, font_info, pdf_bytes = self._render_html_to_image(
            record, template_name, dpi, randomize_fonts
        )

        if output_format in ("pdf", "both"):
            pdf_path = os.path.join(output_dir, "pdfs", f"{base_name}.pdf")
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

        if output_format in ("png", "both"):
            png_path = os.path.join(output_dir, "images", f"{base_name}.png")
            os.makedirs(os.path.dirname(png_path), exist_ok=True)

            if noise_preset:
                aug = DocumentAugmentor(preset=noise_preset)
                img = aug(clean_img)
            else:
                img = clean_img

            img.save(png_path, format="PNG")

        file_name = f"{base_name}.png" if output_format in ("png", "both") else f"{base_name}.pdf"
        mime_type = "image/png" if output_format in ("png", "both") else "application/pdf"

        label = build_ground_truth_label(record, file_name, mime_type)
        label["_template_used"] = template_name
        label["_synthetic"] = True
        label["_font_info"] = font_info or None
        if noise_preset:
            label["_degradation_preset"] = noise_preset

        return label

    def render_dataset(
        self,
        records: List[Dict[str, Any]],
        output_dir: str,
        multi_degrade: bool = False,
        degradation_presets: Optional[List[str]] = None,
        output_format: str = "both",
        noise_preset: Optional[str] = None,
        dpi: int = 150,
        randomize_fonts: bool = True,
        save_pdf: bool = False,
    ) -> List[Dict[str, Any]]:
        """Render all records and produce labels.json."""
        os.makedirs(output_dir, exist_ok=True)
        labels = []
        template_counts = {}

        if multi_degrade:
            if degradation_presets is None:
                degradation_presets = ["clean", "bad_scan", "faded"]
            print(f"Multi-degradation mode: {', '.join(degradation_presets)}")
            print(f"Will produce {len(records)} × {len(degradation_presets)} = "
                  f"{len(records) * len(degradation_presets)} images")
        print()

        for i, record in enumerate(records):
            print(f"[{i+1}/{len(records)}] ", end="", flush=True)

            try:
                if multi_degrade:
                    label = self.render_multi_degrade(
                        record, output_dir, i,
                        degradation_presets=degradation_presets,
                        dpi=dpi,
                        randomize_fonts=randomize_fonts,
                        save_pdf=save_pdf,
                    )
                else:
                    label = self.render_single(
                        record, output_dir, i,
                        output_format=output_format,
                        noise_preset=noise_preset,
                        dpi=dpi,
                        randomize_fonts=randomize_fonts,
                    )

                labels.append(label)

                tmpl = label.get("_template_used", "unknown")
                template_counts[tmpl] = template_counts.get(tmpl, 0) + 1

                font_str = f" [{label.get('_font_info', '')}]" if label.get('_font_info') else ""
                presets_str = ""
                if multi_degrade:
                    n_variants = len(label.get("degradation_variants", {}))
                    presets_str = f" ({n_variants} variants)"

                print(f"{label['file_name']} → {tmpl}{font_str}{presets_str} | "
                      f"{label.get('supplier_name', '?')[:25]} | "
                      f"{label.get('document_currency', '?')} {label.get('document_total_amount', 0)}")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Save labels
        labels_path = os.path.join(output_dir, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

        # Summary
        total_images = len(labels)
        if multi_degrade:
            total_images = len(labels) * len(degradation_presets)

        print(f"\n{'='*60}")
        print(f"Rendered: {len(labels)} records → {total_images} images")
        print(f"Labels:   {labels_path}")
        print(f"\nTemplate distribution:")
        for tmpl, count in sorted(template_counts.items(), key=lambda x: -x[1]):
            print(f"  {tmpl:25s} {count:4d} ({100*count/len(labels):.1f}%)")
        if multi_degrade:
            print(f"\nDegradation presets: {', '.join(degradation_presets)}")
        print(f"Augmentation engine: {'Augraphy + OpenCV' if HAS_AUGRAPHY else 'OpenCV + PIL'}")
        print(f"{'='*60}")

        return labels


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Render synthetic receipt JSON records into PDF/PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test — 5 records, phone photo look
  python render_receipts.py -i records.json -o ./test --limit 5 --preset phone_photo --format png

""",
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output-dir", "-o", default="./rendered", help="Output directory")
    parser.add_argument("--templates", "-t", default="./templates", help="Templates directory")
    parser.add_argument("--format", "-f", default="both", choices=["pdf", "png", "both"])
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI (default: 150)")
    parser.add_argument("--limit", type=int, default=None, help="Only render first N records")
    parser.add_argument("--no-fonts", action="store_true", help="Disable font randomization")

    # Multi-degradation mode
    parser.add_argument("--multi-degrade", action="store_true",
                        help="Render each record at multiple degradation levels")
    parser.add_argument("--presets", nargs="+", default=None,
                        help="Degradation presets for --multi-degrade "
                             "(default: clean light medium heavy phone_photo thermal_aged)")
    parser.add_argument("--save-pdf", action="store_true",
                        help="Also save PDFs in multi-degrade mode")

    # Single-preset mode
    parser.add_argument("--preset", default=None,
                        choices=DocumentAugmentor.list_presets(),
                        help="Apply a single degradation preset to all images")

    # Legacy noise interface
    parser.add_argument("--noise", action="store_true", help="Apply noise (legacy flag)")
    parser.add_argument("--noise-level", default="light",
                        choices=["light", "medium", "heavy"],
                        help="Noise level (legacy, maps to preset)")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.limit:
        records = records[:args.limit]

    noise_preset = args.preset
    if not noise_preset and args.noise:
        noise_preset = args.noise_level

    print(f"{'='*60}")
    print(f"Synthetic Receipt Renderer v2.1")
    print(f"{'='*60}")
    print(f"Records: {len(records)} from {args.input}")
    print(f"Output:  {args.output_dir}")
    print(f"DPI:     {args.dpi}")
    print(f"Fonts:   {'disabled' if args.no_fonts else 'randomized'}")
    if args.multi_degrade:
        presets = args.presets or ["clean", "light", "medium", "heavy",
                                    "phone_photo", "thermal_aged"]
        print(f"Mode:    MULTI-DEGRADE ({', '.join(presets)})")
    elif noise_preset:
        print(f"Mode:    Single preset: {noise_preset}")
    else:
        print(f"Mode:    Clean render ({args.format})")
    print(f"Augraphy: {'available' if HAS_AUGRAPHY else 'not installed (using PIL/OpenCV)'}")
    print(f"{'='*60}\n")

    renderer = ReceiptRenderer(templates_dir=args.templates)
    renderer.render_dataset(
        records,
        output_dir=args.output_dir,
        multi_degrade=args.multi_degrade,
        degradation_presets=args.presets,
        output_format=args.format,
        noise_preset=noise_preset,
        dpi=args.dpi,
        randomize_fonts=not args.no_fonts,
        save_pdf=args.save_pdf or args.format in ("pdf", "both"),
    )


if __name__ == "__main__":
    main()