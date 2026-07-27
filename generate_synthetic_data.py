#!/usr/bin/env python3

import json
import os
import random
import time
import argparse
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# (region, language, tax_regime, weight)
REGIONS = [
    ("IE", "en", "irish_vat", 0.45),    # 45% Irish (your primary use case)
    ("GB", "en", "uk_vat", 0.25),        # 25% UK
    ("US", "en", "us_sales_tax", 0.20),  # 20% US
    ("AU", "en", "australian_gst", 0.05),# 5% Australian
    ("CA", "en", "canadian_gst", 0.05),  # 5% Canadian
]

DOCUMENT_TYPES = [
    "tax_invoice", "proforma_invoice", "credit_note", "receipt",
    "order_confirmation", "freelancer_invoice", "utility_bill",
    "event_ticket", "hotel_receipt", "restaurant_receipt",
    "taxi_receipt", "subscription_invoice", "consulting_invoice",
    "rent_invoice", "delivery_note",
]

# (complexity, weight)
COMPLEXITIES = [
    ("low", 0.35),     # 1-2 line items
    ("medium", 0.45),  # 3-5 line items
    ("high", 0.20),    # 6-15 line items
]

# Sparsity profiles — controls how many fields get filled
# This makes generated data more realistic (real receipts rarely have ALL fields)
SPARSITY_PROFILES = {
    "receipt": {
        "always": ["supplier_name", "document_date", "document_currency",
                    "document_total_amount", "line_items"],
        "common": ["document_time", "receipt_number", "supplier_address",
                    "supplier_phone", "taxes", "document_total_tax",
                    "document_total_net"],
        "rare": ["customer_name", "supplier_email", "supplier_website",
                 "supplier_company_registrations", "notes"],
        "skip": ["customer_address", "billing_address", "shipping_address",
                 "customer_company_registrations", "customer_id",
                 "invoice_number", "po_number", "due_date", "payment_date"],
    },
    "invoice": {
        "always": ["supplier_name", "document_date", "document_currency",
                    "document_total_amount", "document_total_net",
                    "document_total_tax", "line_items", "taxes",
                    "invoice_number"],
        "common": ["supplier_address", "supplier_company_registrations",
                    "customer_name", "customer_address", "due_date",
                    "supplier_phone", "document_number"],
        "rare": ["supplier_email", "supplier_website", "customer_id",
                 "customer_company_registrations", "po_number",
                 "reference_numbers", "billing_address", "payment_date",
                 "notes"],
        "skip": ["shipping_address", "document_time"],
    },
    "event": {
        "always": ["supplier_name", "document_date", "document_currency",
                    "document_total_amount", "line_items", "document_number"],
        "common": ["customer_name", "receipt_number", "document_time"],
        "rare": ["supplier_address", "supplier_email", "supplier_website",
                 "notes"],
        "skip": ["supplier_phone", "supplier_company_registrations",
                 "customer_address", "customer_id", "billing_address",
                 "shipping_address", "customer_company_registrations",
                 "invoice_number", "po_number", "due_date", "payment_date",
                 "taxes", "document_total_tax", "document_total_net"],
    },
    "freelancer": {
        "always": ["supplier_name", "supplier_address", "document_date",
                    "document_currency", "document_total_amount",
                    "document_total_net", "line_items", "invoice_number",
                    "customer_name"],
        "common": ["supplier_phone", "supplier_email", "customer_address",
                    "due_date", "taxes", "document_total_tax", "notes"],
        "rare": ["supplier_website", "supplier_company_registrations",
                 "customer_company_registrations", "po_number",
                 "reference_numbers"],
        "skip": ["customer_id", "billing_address", "shipping_address",
                 "document_time", "receipt_number"],
    },
}

# Map document types to sparsity profiles
DOC_TYPE_TO_PROFILE = {
    "receipt": "receipt", "restaurant_receipt": "receipt",
    "taxi_receipt": "receipt", "hotel_receipt": "receipt",
    "tax_invoice": "invoice", "proforma_invoice": "invoice",
    "credit_note": "invoice", "subscription_invoice": "invoice",
    "rent_invoice": "invoice", "utility_bill": "invoice",
    "delivery_note": "invoice", "consulting_invoice": "invoice",
    "order_confirmation": "event", "event_ticket": "event",
    "freelancer_invoice": "freelancer",
}

LAYOUT_STYLES = [
    "formal_corporate", "modern_minimal", "traditional_tabular",
    "pos_thermal", "freelancer", "utility_bill", "event_ticket",
]

COLOR_SCHEMES = [
    "blue_corporate", "green_eco", "red_accent", "monochrome",
    "teal_modern", "purple_creative", "orange_warm", "navy_formal",
    "dark_green", "burgundy",
]


# ═══════════════════════════════════════════════════════════════════════
# PARAMETER GENERATION
# ═══════════════════════════════════════════════════════════════════════

def weighted_choice(items_with_weights: list) -> tuple:
    """Pick from list of (*values, weight) tuples, return values."""
    values = [item[:-1] for item in items_with_weights]
    weights = [item[-1] for item in items_with_weights]
    chosen = random.choices(values, weights=weights, k=1)[0]
    return chosen


def generate_params() -> Dict[str, Any]:
    """Generate randomized parameters for one synthetic record."""
    region, language, tax_regime = weighted_choice(REGIONS)
    (complexity,) = weighted_choice(COMPLEXITIES)
    doc_type = random.choice(DOCUMENT_TYPES)
    profile = DOC_TYPE_TO_PROFILE.get(doc_type, "invoice")

    return {
        "region": region,
        "language": language,
        "tax_regime": tax_regime,
        "complexity": complexity,
        "document_type": doc_type,
        "sparsity_profile": profile,
    }


def get_fields_to_fill(params: Dict[str, Any]) -> List[str]:
    """
    Determine which fields the LLM should fill based on sparsity profile.
    
    - 'always' fields: 100% chance
    - 'common' fields: 70% chance each
    - 'rare' fields: 25% chance each
    - 'skip' fields: 0% chance (set to null)
    """
    profile_name = params["sparsity_profile"]
    profile = SPARSITY_PROFILES.get(profile_name, SPARSITY_PROFILES["invoice"])

    fields = list(profile["always"])

    for f in profile["common"]:
        if random.random() < 0.70:
            fields.append(f)

    for f in profile["rare"]:
        if random.random() < 0.25:
            fields.append(f)

    # skip fields are never included
    return fields


# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(params: Dict[str, Any], fields_to_fill: List[str]) -> str:
    """
    Build a compact prompt for local LLMs.
    
    Key design choices for local models:
    - Shorter prompts = faster inference
    - Explicit field list = model knows exactly what to output
    - Math rules stated clearly = fewer validation failures
    - "null for anything not listed" = clean sparse output
    """
    region = params["region"]
    doc_type = params["document_type"]
    complexity = params["complexity"]
    language = params["language"]
    tax_regime = params["tax_regime"]

    # Compact complexity description
    complexity_desc = {
        "low": "1-2 line items",
        "medium": "3-5 line items",
        "high": "6-15 line items",
    }[complexity]

    # Tax regime hints
    tax_hints = {
        "irish_vat": "Irish VAT rates: 23%, 13.5%, 9%, 0%. VAT number format: IE + 7 digits + 1-2 letters. Eircode postcodes (e.g. T12 AB34).",
        "uk_vat": "UK VAT: 20%, 5%, 0%. VAT format: GB + 9 digits. UK postcodes (e.g. SW1A 1AA).",
        "us_sales_tax": "US state sales tax varies 0-10.25%. No federal VAT. Use realistic state tax. ZIP codes.",
        "australian_gst": "Australian GST: 10% flat rate. ABN format: 11 digits. Postcodes: 4 digits.",
        "canadian_gst": "Canadian GST: 5%. HST varies by province (13-15%). GST/HST number: 9 digits + RT0001. Postcodes: A1A 1A1.",
    }.get(tax_regime, "Use appropriate local tax rates.")

    # Build the field list
    fields_str = ", ".join(fields_to_fill)

    # Determine which array fields are needed
    has_line_items = "line_items" in fields_to_fill
    has_taxes = "taxes" in fields_to_fill

    prompt = f"""Generate a fictional {doc_type.replace('_', ' ')} as JSON for region {region}.
All content in English.

RULES:
- ALL data must be FICTIONAL. No real companies, people, addresses, or tax numbers.
- Data must be plausible for {region}.
- {tax_hints}
- Complexity: {complexity_desc}
- Date range: 2022-01-01 to 2024-12-31, format YYYY-MM-DD.
- Currency: use the standard currency for {region}.

STRICT FORMAT RULES:
- tax_rate must be a STRING like "23%" or "10%" or "0%" — never a decimal or integer.
- supplier_company_registrations must be a STRING like "IE 3298401F" — never an array.
- All amounts must be numbers (not strings).
- All text fields must be strings (not arrays).

FIELDS TO FILL (set everything else to null):
{fields_str}

"""

    if has_line_items:
        prompt += """LINE ITEMS array — each item has:
  description (string), quantity (number), unit_price (number), amount (number), tax_rate (string like "23%"), tax_amount (number)
"""

    if has_taxes:
        prompt += """TAXES array — each entry has:
  base (number), rate (string like "23%"), code (string like "VAT"), value (number)
"""

    prompt += f"""
MATH (critical):
- amount = quantity × unit_price
- document_total_net = sum of all line item amounts
- document_total_tax = sum of all line item tax_amounts
- document_total_amount = document_total_net + document_total_tax
- Round all amounts to 2 decimal places.

Also include:
- document_category: one of (services, software, electronics, travel, food, office_supplies, utilities, rent, shipping, retail, transport, miscellaneous, other)
- subcategory: more specific (e.g. "taxi", "hotel", "consulting", "coworking")
- locale: "en"

Return ONLY valid JSON. No markdown fences, no explanation, no commentary."""

    return prompt

# Default endpoints for each backend
BACKEND_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "gpt-oss:20b",
        "api_key": "ollama",  # Ollama ignores this but openai client requires it
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "openai/gpt-oss-20b",
        "api_key": "dummy",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key": None,  # Must be set via OPENAI_API_KEY env var
    },
    "gemini": {
        "base_url": None,  # Uses google-genai client
        "model": "gemini-2.5-flash",
        "api_key": None,  # Must be set via GEMINI_API_KEY env var
    },
}


class LLMClient:

    def __init__(self, backend: str, base_url: str = None,
                 model: str = None, api_key: str = None,
                 temperature: float = 0.9, max_tokens: int = 4096,
                 reasoning_effort: str = "low"):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        defaults = BACKEND_DEFAULTS.get(backend, {})
        self.base_url = base_url or defaults.get("base_url")
        self.model = model or defaults.get("model")
        self.api_key = api_key or defaults.get("api_key")

        if backend == "gemini":
            self._init_gemini()
        else:
            self._init_openai_compat()

    def _init_openai_compat(self):
        """Initialize OpenAI-compatible client (Ollama, vLLM, OpenAI)."""
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package required. Install with: pip install openai")
            sys.exit(1)

        # Resolve API key
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key and self.backend == "openai":
                print("ERROR: Set OPENAI_API_KEY environment variable")
                sys.exit(1)
            elif not self.api_key:
                self.api_key = "not-needed"

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        print(f"Backend: {self.backend} @ {self.base_url}")
        print(f"Model: {self.model}")

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            from google import genai
        except ImportError:
            print("ERROR: google-genai required. Install with: pip install google-genai")
            sys.exit(1)

        api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: Set GEMINI_API_KEY environment variable")
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        print(f"Backend: gemini")
        print(f"Model: {self.model}")

    def generate(self, prompt: str) -> str:
        """Generate text from prompt. Returns raw text response."""
        if self.backend == "gemini":
            return self._generate_gemini(prompt)
        else:
            return self._generate_openai_compat(prompt)

    def _generate_openai_compat(self, prompt: str) -> str:
        """Call OpenAI-compatible endpoint (Ollama, vLLM, OpenAI)."""
        # Build system message — for gpt-oss, include reasoning effort
        system_msg = "You are a synthetic financial document data generator. Generate realistic but entirely fictional invoice/receipt data as JSON."

        if "gpt-oss" in self.model:
            system_msg += f"\n\nReasoning effort: {self.reasoning_effort}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        return response.text


# ═══════════════════════════════════════════════════════════════════════
# PARSING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def extract_json(text: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response, handling common issues:
    - Markdown fences
    - Leading/trailing text
    - Thinking tokens (gpt-oss reasoning output)
    """
    if not text:
        return None

    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first line (```json or ```)
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in response (handles thinking tokens / preamble)
    import re
    # Find the outermost {...} block
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None  # Try next block

    return None


def validate_record(record: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a generated record. Returns (is_valid, list_of_issues).
    Lenient — we don't require all fields, just check math consistency.
    """
    issues = []

    # Must have at minimum these fields
    if not record.get("supplier_name"):
        issues.append("missing supplier_name")

    if not record.get("document_date"):
        issues.append("missing document_date")

    # Validate math if amounts are present
    total = record.get("document_total_amount")
    net = record.get("document_total_net")
    tax = record.get("document_total_tax")

    if total is not None and net is not None and tax is not None:
        try:
            total_f = float(total)
            net_f = float(net)
            tax_f = float(tax)
            if abs((net_f + tax_f) - total_f) > 0.10:
                issues.append(
                    f"total mismatch: net({net_f}) + tax({tax_f}) = "
                    f"{net_f + tax_f:.2f} != total({total_f})"
                )
        except (ValueError, TypeError):
            issues.append("non-numeric amount fields")

    # Validate line items math (lenient)
    line_items = record.get("line_items")
    if isinstance(line_items, list) and line_items and net is not None:
        try:
            items_sum = sum(float(item.get("amount", 0) or 0) for item in line_items)
            net_f = float(net)
            if items_sum > 0 and abs(items_sum - net_f) > 1.0:
                issues.append(
                    f"line items sum({items_sum:.2f}) != net({net_f:.2f})"
                )
        except (ValueError, TypeError):
            pass  # Don't fail on this

    is_valid = len([i for i in issues if "missing" in i]) == 0
    return is_valid, issues


def fix_math(record: Dict) -> Dict:
    """
    Attempt to fix common math inconsistencies in generated records.
    Runs AFTER LLM generation as a post-processing step.
    """
    line_items = record.get("line_items")
    if not isinstance(line_items, list):
        return record

    # Fix individual line items
    for item in line_items:
        try:
            qty = float(item.get("quantity", 1) or 1)
            price = float(item.get("unit_price", 0) or 0)
            amount = float(item.get("amount", 0) or 0)

            # Recalculate amount if it doesn't match
            expected = round(qty * price, 2)
            if price > 0 and abs(expected - amount) > 0.02:
                item["amount"] = expected

            # Fix tax_amount
            tax_rate_str = str(item.get("tax_rate", "0") or "0")
            tax_rate_num = float(tax_rate_str.replace("%", "").strip() or "0")
            if tax_rate_num > 1:
                tax_rate_num = tax_rate_num / 100  # Convert 23 → 0.23
            item["tax_amount"] = round(float(item.get("amount", 0) or 0) * tax_rate_num, 2)

        except (ValueError, TypeError):
            continue

    # Recalculate totals
    try:
        net = sum(float(item.get("amount", 0) or 0) for item in line_items)
        tax = sum(float(item.get("tax_amount", 0) or 0) for item in line_items)
        record["document_total_net"] = round(net, 2)
        record["document_total_tax"] = round(tax, 2)
        record["document_total_amount"] = round(net + tax, 2)

        # Fix taxes array too
        taxes = record.get("taxes")
        if isinstance(taxes, list) and len(taxes) == 1:
            taxes[0]["base"] = round(net, 2)
            taxes[0]["value"] = round(tax, 2)
    except (ValueError, TypeError):
        pass

    return record


# ═══════════════════════════════════════════════════════════════════════
# POST-PROCESSING — Normalize LLM output & inject template hints
# ═══════════════════════════════════════════════════════════════════════

# Deterministic mapping: document_type → layout_style
# (The LLM was picking wrong templates, so we do this ourselves)
DOC_TYPE_TO_LAYOUT = {
    "receipt": "pos_thermal",
    "restaurant_receipt": "pos_thermal",
    "taxi_receipt": "pos_thermal",
    "hotel_receipt": "modern_minimal",
    "tax_invoice": "formal_corporate",
    "proforma_invoice": "formal_corporate",
    "credit_note": "utility_bill",
    "rent_invoice": "formal_corporate",
    "utility_bill": "utility_bill",
    "delivery_note": "utility_bill",
    "subscription_invoice": "modern_minimal",
    "consulting_invoice": "freelancer_invoice",
    "freelancer_invoice": "freelancer_invoice",
    "order_confirmation": "event_ticket",
    "event_ticket": "event_ticket",
}

COLOR_SCHEMES = [
    "blue_corporate", "green_eco", "red_accent", "monochrome",
    "teal_modern", "purple_creative", "orange_warm", "navy_formal",
    "dark_green", "burgundy",
]


def inject_template_hints(record: Dict, params: Dict) -> Dict:
    """
    Inject _template_hints based on document_type.
    Overrides whatever the LLM may have generated.
    """
    doc_type = params.get("document_type", "tax_invoice")
    layout = DOC_TYPE_TO_LAYOUT.get(doc_type, "modern_minimal")
    color = random.choice(COLOR_SCHEMES)

    record["_template_hints"] = {
        "layout_style": layout,
        "color_scheme": color,
    }
    return record


def normalize_record(record: Dict) -> Dict:
    """
    Fix common LLM output issues:
    - tax_rate: convert decimal/int to string percentage (0.23 → "23%", 10 → "10%")
    - supplier_company_registrations: convert array to string
    - Ensure locale is "en"
    """
    # Fix tax_rate in line items
    for item in (record.get("line_items") or []):
        tax_rate = item.get("tax_rate")
        if tax_rate is not None:
            item["tax_rate"] = _normalize_tax_rate(tax_rate)

    # Fix tax rate in taxes array
    for tax in (record.get("taxes") or []):
        rate = tax.get("rate")
        if rate is not None:
            tax["rate"] = _normalize_tax_rate(rate)

    # Fix supplier_company_registrations: array → string
    scr = record.get("supplier_company_registrations")
    if isinstance(scr, list):
        record["supplier_company_registrations"] = ", ".join(str(x) for x in scr)

    # Same for customer_company_registrations
    ccr = record.get("customer_company_registrations")
    if isinstance(ccr, list):
        record["customer_company_registrations"] = ", ".join(str(x) for x in ccr)

    # Ensure locale is "en"
    record["locale"] = "en"

    return record


def _normalize_tax_rate(value) -> str:
    """
    Normalize any tax_rate representation to a string percentage.
    
    Input → Output:
      0.23   → "23%"
      0.055  → "5.5%"
      23     → "23%"
      10     → "10%"
      "23%"  → "23%"
      "10"   → "10%"
      0      → "0%"
      None   → "0%"
    """
    if value is None:
        return "0%"

    s = str(value).strip()

    # Already formatted with %
    if s.endswith("%"):
        return s

    try:
        num = float(s)
    except (ValueError, TypeError):
        return "0%"

    # Detect if it's a decimal fraction (< 1 means it's like 0.23 = 23%)
    if 0 < num < 1:
        pct = round(num * 100, 2)
        # Clean up: 23.0 → 23, 5.5 stays 5.5
        if pct == int(pct):
            return f"{int(pct)}%"
        return f"{pct}%"
    elif num == 0:
        return "0%"
    else:
        # Already a percentage number (23, 10, etc.)
        if num == int(num):
            return f"{int(num)}%"
        return f"{num}%"

def generate_dataset(
    client: LLMClient,
    count: int,
    output_path: str,
    max_retries: int = 2,
    fix_math_flag: bool = True,
    delay: float = 0.0,
    dry_run: bool = False,
):
    """Generate the full synthetic dataset."""

    records = []
    failed = 0
    math_fixed = 0

    for i in range(count):
        params = generate_params()
        fields_to_fill = get_fields_to_fill(params)
        prompt = build_prompt(params, fields_to_fill)

        status = (
            f"[{i + 1}/{count}] "
            f"{params['document_type']:25s} "
            f"{params['region']} "
            f"{params['complexity']:6s} "
            f"({len(fields_to_fill)} fields)"
        )

        if dry_run:
            print(f"{status} → DRY RUN")
            if i == 0:
                print(f"\n--- SAMPLE PROMPT ---\n{prompt}\n--- END ---\n")
            continue

        print(f"{status} ", end="", flush=True)

        # Try generation with retries
        record = None
        for attempt in range(max_retries + 1):
            try:
                raw = client.generate(prompt)
                record = extract_json(raw)

                if record is None:
                    print(f"[parse fail, attempt {attempt + 1}] ", end="", flush=True)
                    continue

                # Validate
                is_valid, issues = validate_record(record)
                if not is_valid and attempt < max_retries:
                    print(f"[invalid: {issues[0]}, retry] ", end="", flush=True)
                    record = None
                    continue

                break

            except Exception as e:
                print(f"[error: {e}] ", end="", flush=True)
                if attempt < max_retries:
                    time.sleep(2)

        if record is None:
            print("FAILED")
            failed += 1
            continue

        # Post-process: fix math
        if fix_math_flag:
            _, issues_before = validate_record(record)
            record = fix_math(record)
            _, issues_after = validate_record(record)
            if len(issues_after) < len(issues_before):
                math_fixed += 1

        # Post-process: normalize formats (tax_rate strings, array→string, etc.)
        record = normalize_record(record)

        # Post-process: inject template hints (deterministic, not LLM-chosen)
        record = inject_template_hints(record, params)

        # Set null for any fields not in fields_to_fill
        all_possible_fields = [
            "supplier_name", "supplier_address", "supplier_phone",
            "supplier_email", "supplier_website", "supplier_company_registrations",
            "customer_name", "customer_id", "customer_address",
            "billing_address", "shipping_address", "customer_company_registrations",
            "document_type_extended", "document_number", "invoice_number",
            "receipt_number", "reference_numbers", "po_number",
            "document_date", "document_time", "due_date", "payment_date",
            "document_currency", "document_total_amount", "document_total_net",
            "document_total_tax",
        ]
        for f in all_possible_fields:
            if f not in fields_to_fill and f not in record:
                record[f] = None

        # Add metadata
        record["_generation_params"] = params
        record["_generation_index"] = i
        record["_fields_requested"] = fields_to_fill

        records.append(record)

        supplier = record.get("supplier_name", "?")
        total = record.get("document_total_amount", 0)
        currency = record.get("document_currency", "?")
        layout = record.get("_template_hints", {}).get("layout_style", "?")
        _, remaining_issues = validate_record(record)
        status_emoji = "OK" if not remaining_issues else f"WARN({len(remaining_issues)})"
        print(f"→ {supplier[:30]} | {currency} {total} | {layout} | {status_emoji}")

        if delay > 0:
            time.sleep(delay)

    if dry_run:
        print(f"\nDry run complete. Would generate {count} records.")
        return []

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated: {len(records)} / {count}")
    print(f"Failed:    {failed}")
    print(f"Math auto-fixed: {math_fixed}")
    print(f"Saved to:  {output_path}")

    # Distribution stats
    regions = {}
    doc_types = {}
    profiles = {}
    for r in records:
        p = r.get("_generation_params", {})
        reg = p.get("region", "?")
        dt = p.get("document_type", "?")
        sp = p.get("sparsity_profile", "?")
        regions[reg] = regions.get(reg, 0) + 1
        doc_types[dt] = doc_types.get(dt, 0) + 1
        profiles[sp] = profiles.get(sp, 0) + 1

    print(f"\nRegion distribution:")
    for k, v in sorted(regions.items(), key=lambda x: -x[1]):
        print(f"  {k:5s} {v:4d} ({100 * v / len(records):.1f}%)")

    print(f"\nTemplate distribution:")
    templates = {}
    for r in records:
        tmpl = r.get("_template_hints", {}).get("layout_style", "?")
        templates[tmpl] = templates.get(tmpl, 0) + 1
    for k, v in sorted(templates.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v:4d} ({100 * v / len(records):.1f}%)")

    print(f"\nSparsity profile distribution:")
    for k, v in sorted(profiles.items(), key=lambda x: -x[1]):
        print(f"  {k:15s} {v:4d} ({100 * v / len(records):.1f}%)")

    avg_fields = sum(
        len(r.get("_fields_requested", []))
        for r in records
    ) / max(len(records), 1)
    print(f"\nAvg fields per record: {avg_fields:.1f}")
    print(f"{'=' * 60}")

    return records


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic receipt/invoice JSON data using local or remote LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ollama with gpt-oss:20b (default)
  python generate_synthetic_data.py --count 200
""",
    )

    # Backend selection
    parser.add_argument(
        "--backend", "-b",
        default="ollama",
        choices=["ollama", "vllm", "openai", "gemini"],
        help="LLM backend (default: ollama)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override base URL for the API endpoint",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (default: gpt-oss:20b for ollama)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set OPENAI_API_KEY / GEMINI_API_KEY env var)",
    )

    # Generation params
    parser.add_argument("--count", "-n", type=int, default=200, help="Number of records (default: 200)")
    parser.add_argument("--output", "-o", default="synthetic_records.json", help="Output JSON file")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens (default: 4096)")
    parser.add_argument("--retries", type=int, default=2, help="Max retries per record (default: 2)")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests in seconds (default: 0)")
    parser.add_argument("--no-fix-math", action="store_true", help="Disable automatic math correction")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for gpt-oss models (default: low — faster)",
    )

    # Utility
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling LLM")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)

    # Print config
    print(f"{'=' * 60}")
    print("Synthetic Receipt Data Generator")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("MODE: DRY RUN (no LLM calls)")
        client = None
    else:
        client = LLMClient(
            backend=args.backend,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )

    print(f"Count: {args.count}")
    print(f"Output: {args.output}")
    print(f"Temperature: {args.temperature}")
    print(f"Math fix: {'disabled' if args.no_fix_math else 'enabled'}")
    print(f"{'=' * 60}\n")

    generate_dataset(
        client=client,
        count=args.count,
        output_path=args.output,
        max_retries=args.retries,
        fix_math_flag=not args.no_fix_math,
        delay=args.delay,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()