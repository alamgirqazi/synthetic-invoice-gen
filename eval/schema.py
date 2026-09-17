"""
Receipt OCR Evaluation Schema

Defines field types, tiers, weights, and evaluation strategies.
The schema drives the entire evaluation pipeline.

Field Types:
    numeric       - Amounts, quantities. Exact match with tolerance.
    date          - Dates. Parsed comparison (format-agnostic).
    time          - Times. Parsed comparison.
    short_string  - 1-10 words. Normalized exact + embedding similarity.
    medium_string - 10-100 words. Token F1 + embedding similarity.
    long_text     - 100+ words. Token F1 + key info extraction.
    category      - Classification from a fixed taxonomy. Exact match primary.
    line_items    - Nested array. Hungarian matching + per-field scoring.
    tax_items     - Nested array of tax breakdowns.

Tiers:
    1 - Critical   (accounting/identification fields)
    2 - Important   (classification/reference fields)
    3 - Supplementary (contact/metadata fields)

FIXES:
    - Added identity_critical flag to FieldDefinition. When True, the
      comparator weights lexical metrics (Levenshtein, token F1) higher
      than embedding similarity to prevent false positives on entity names.
    - Changed document_category and subcategory to field_type="category"
      so they use exact-match-first comparison instead of fuzzy similarity.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class FieldDefinition:
    """Definition of a single field in the evaluation schema."""
    name: str
    field_type: str
    tier: int
    weight: float
    description: str = ""
    numeric_tolerance: float = 0.01
    threshold_exact: float = 0.90
    threshold_acceptable: float = 0.75
    threshold_partial: float = 0.55
    identity_critical: bool = False  # NEW: when True, lexical > embedding


@dataclass
class LineItemFieldDef:
    """Definition of a field within a line item."""
    name: str
    field_type: str
    weight: float = 1.0


# ============================================================
# MAIN FIELD DEFINITIONS
# ============================================================

FIELD_DEFINITIONS: Dict[str, FieldDefinition] = {
    # --- Tier 1: Critical ---
    "supplier_name": FieldDefinition(
        name="supplier_name", field_type="short_string", tier=1, weight=1.0,
        description="Company/merchant name",
        identity_critical=True,  # FIX: "Cork Chamber" ≠ "Cork City Council"
    ),
    "document_date": FieldDefinition(
        name="document_date", field_type="date", tier=1, weight=1.0,
        description="Document issue date"
    ),
    "document_currency": FieldDefinition(
        name="document_currency", field_type="short_string", tier=1, weight=0.8,
        description="3-letter currency code", threshold_exact=0.95,
        identity_critical=True,  # "EUR" ≠ "GBP" even if semantically close
    ),
    "document_total_amount": FieldDefinition(
        name="document_total_amount", field_type="numeric", tier=1, weight=1.5,
        description="Final total amount"
    ),
    "document_total_net": FieldDefinition(
        name="document_total_net", field_type="numeric", tier=1, weight=1.0,
        description="Amount before tax (subtotal)"
    ),
    "document_total_tax": FieldDefinition(
        name="document_total_tax", field_type="numeric", tier=1, weight=1.0,
        description="Tax amount"
    ),
    "line_items": FieldDefinition(
        name="line_items", field_type="line_items", tier=1, weight=2.0,
        description="Array of line items"
    ),

    "invoice_number": FieldDefinition(
        name="invoice_number", field_type="short_string", tier=1, weight=0.8,
        description="Invoice number",
        identity_critical=True,  # FIX: invoice numbers must match exactly
    ),
    "document_category": FieldDefinition(
        name="document_category", field_type="category", tier=1, weight=0.7,
        description="Document category",
        # FIX: changed from short_string to category
    ),
    
    "payment_date": FieldDefinition(
        name="payment_date", field_type="date", tier=1, weight=0.3,
        description="Payment date"
    ),
    # --- Tier 2: Important ---
    "subcategory": FieldDefinition(
        name="subcategory", field_type="category", tier=2, weight=0.5,
        description="Document subcategory",
        # FIX: changed from short_string to category
    ),
    "document_number": FieldDefinition(
        name="document_number", field_type="short_string", tier=2, weight=0.8,
        description="General document number",
        identity_critical=True,
    ),

    "receipt_number": FieldDefinition(
        name="receipt_number", field_type="short_string", tier=3, weight=0.5,
        description="Receipt number",
        identity_critical=True,
    ),
    "document_type_extended": FieldDefinition(
        name="document_type_extended", field_type="category", tier=2, weight=0.5,
        description="Document type (invoice, receipt, credit_note, etc.)",
        # FIX: changed from short_string to category
    ),
    "taxes": FieldDefinition(
        name="taxes", field_type="tax_items", tier=2, weight=1.0,
        description="Structured tax breakdown"
    ),
    "supplier_company_registrations": FieldDefinition(
        name="supplier_company_registrations", field_type="short_string", tier=2, weight=0.6,
        description="Supplier VAT/registration numbers",
        identity_critical=True,
    ),
    "due_date": FieldDefinition(
        name="due_date", field_type="date", tier=2, weight=0.5,
        description="Payment due date"
    ),
    "reference_numbers": FieldDefinition(
        name="reference_numbers", field_type="short_string", tier=3, weight=0.3,
        description="Reference numbers",
        identity_critical=True,
    ),
    "po_number": FieldDefinition(
        name="po_number", field_type="short_string", tier=3, weight=0.3,
        description="Purchase order number",
        identity_critical=True,
    ),

    # --- Tier 3: Supplementary ---
    "document_time": FieldDefinition(
        name="document_time", field_type="time", tier=3, weight=0.3,
        description="Document time"
    ),

    "customer_name": FieldDefinition(
        name="customer_name", field_type="short_string", tier=3, weight=0.5,
        description="Customer/recipient name",
        identity_critical=True,
    ),
    "customer_id": FieldDefinition(
        name="customer_id", field_type="short_string", tier=3, weight=0.3,
        description="Customer ID or email",
        identity_critical=True,
    ),
    "customer_address": FieldDefinition(
        name="customer_address", field_type="medium_string", tier=3, weight=0.4,
        description="Customer address"
    ),
    "billing_address": FieldDefinition(
        name="billing_address", field_type="medium_string", tier=3, weight=0.3,
        description="Billing address"
    ),
    "shipping_address": FieldDefinition(
        name="shipping_address", field_type="medium_string", tier=3, weight=0.3,
        description="Shipping address"
    ),
    "supplier_address": FieldDefinition(
        name="supplier_address", field_type="medium_string", tier=3, weight=0.4,
        description="Supplier address"
    ),
    "supplier_email": FieldDefinition(
        name="supplier_email", field_type="short_string", tier=3, weight=0.3,
        description="Supplier email",
        identity_critical=True,
    ),
    "supplier_phone": FieldDefinition(
        name="supplier_phone", field_type="short_string", tier=3, weight=0.3,
        description="Supplier phone",
        identity_critical=True,
    ),
    "supplier_website": FieldDefinition(
        name="supplier_website", field_type="short_string", tier=3, weight=0.3,
        description="Supplier website",
        identity_critical=True,
    ),
    "customer_company_registrations": FieldDefinition(
        name="customer_company_registrations", field_type="short_string", tier=3, weight=0.3,
        description="Customer registration numbers",
        identity_critical=True,
    ),
    "locale": FieldDefinition(
        name="locale", field_type="short_string", tier=3, weight=0.2,
        description="Document locale/language"
    ),
    "notes": FieldDefinition(
        name="notes", field_type="long_text", tier=3, weight=0.3,
        description="Additional notes, terms, references"
    ),
}

# ============================================================
# LINE ITEM FIELD DEFINITIONS
# ============================================================

LINE_ITEM_FIELDS: Dict[str, LineItemFieldDef] = {
    "description": LineItemFieldDef(name="description", field_type="medium_string", weight=1.5),
    "quantity": LineItemFieldDef(name="quantity", field_type="numeric", weight=0.5),
    "unit_price": LineItemFieldDef(name="unit_price", field_type="numeric", weight=1.0),
    "amount": LineItemFieldDef(name="amount", field_type="numeric", weight=1.2),
    "tax_rate": LineItemFieldDef(name="tax_rate", field_type="short_string", weight=0.5),
    "tax_amount": LineItemFieldDef(name="tax_amount", field_type="numeric", weight=0.8),
}

TAX_ITEM_FIELDS: Dict[str, LineItemFieldDef] = {
    "base": LineItemFieldDef(name="base", field_type="numeric", weight=1.0),
    "rate": LineItemFieldDef(name="rate", field_type="numeric", weight=1.0),
    "code": LineItemFieldDef(name="code", field_type="short_string", weight=0.5),
    "value": LineItemFieldDef(name="value", field_type="numeric", weight=1.2),
}


def get_fields_by_tier(tier: int) -> Dict[str, FieldDefinition]:
    return {k: v for k, v in FIELD_DEFINITIONS.items() if v.tier == tier}


def get_fields_by_type(field_type: str) -> Dict[str, FieldDefinition]:
    return {k: v for k, v in FIELD_DEFINITIONS.items() if v.field_type == field_type}


def get_evaluatable_fields() -> Dict[str, FieldDefinition]:
    exclude = {"file_name", "mime_type", "labeled_at"}
    return {k: v for k, v in FIELD_DEFINITIONS.items() if k not in exclude}