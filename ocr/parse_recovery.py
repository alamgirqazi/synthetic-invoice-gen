#!/usr/bin/env python3
"""
Parse recovery utilities for OCR pipeline.

Handles the common failure mode where VLMs output verbose reasoning
(markdown headers, bullet points, step-by-step analysis) instead of
JSON, then hit the token limit before producing the actual JSON object.

Three recovery strategies:
1. Regex extraction from reasoning text (fast, no model call)
2. Retry with text-only condensed prompt (cheap, high success rate)
3. JSON mode for vLLM backends (prevents the problem entirely)
"""

import json
import re
from typing import Dict, Any, Optional, List


def extract_from_reasoning(raw_text: str) -> Dict[str, Any]:
    """
    Extract structured fields from verbose model reasoning output.
    
    When models ignore "Return ONLY JSON" and output markdown analysis like:
        **Supplier Name**: "ACME Corp"
        **Total Amount**: "€222.75"
        **Date**: "15 July 2023"
    
    This function scrapes those key-value pairs and maps them to schema fields.
    
    Returns:
        Dict with whatever fields could be recovered (may be partial).
        Returns empty dict if nothing useful found.
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return {}

    recovered = {}

    # ── Field extraction patterns ──
    # Each tuple: (schema_field, list_of_regex_patterns)
    # Patterns look for common reasoning formats:
    #   **Key**: "value"
    #   - Key: value
    #   * Key: value  
    #   "Key": "value"
    field_patterns = [
        ("supplier_name", [
            r'(?:supplier|merchant|company|vendor|seller)\s*(?:name)?[:\s]*["\']?([A-Z][A-Za-z0-9\s&.,\'-]+?)(?:["\']?\s*(?:\n|\.|,\s*[a-z]|$))',
            r'["\']?(?:Merchant Name|Supplier)["\']?\s*:\s*["\']([^"\']+)["\']',
        ]),
        ("document_date", [
            r'(?:date|document.?date|invoice.?date|order.?date)[:\s]*["\']?(\d{1,2}[\s/.-]\w+[\s/.-]\d{2,4})["\']?',
            r'(?:date|document.?date|invoice.?date)[:\s]*["\']?(\d{4}[-/]\d{2}[-/]\d{2})["\']?',
        ]),
        ("document_total_amount", [
            r'(?:total|final.?total|total.?amount|amount.?due|grand.?total)[:\s]*["\']?[€$£¥]?\s*(\d[\d,]*\.?\d*)',
            r'[€$£¥]\s*(\d[\d,]*\.\d{2})\s*(?:total|$)',
        ]),
        ("document_total_net", [
            r'(?:subtotal|sub.?total|net|before.?tax|excl\.?.?(?:vat|tax))[:\s]*["\']?[€$£¥]?\s*(\d[\d,]*\.?\d*)',
        ]),
        ("document_total_tax", [
            r'(?:tax|vat|gst|tax.?amount|total.?tax)[:\s]*["\']?[€$£¥]?\s*(\d[\d,]*\.?\d*)',
        ]),
        ("document_currency", [
            r'(?:currency)[:\s]*["\']?([A-Z]{3})["\']?',
            r'([€$£¥])',  # symbol fallback
        ]),
        ("document_number", [
            r'(?:order|document|invoice|receipt)\s*(?:number|no\.?|#)[:\s]*["\']?([A-Z0-9#-]+)["\']?',
            r'#([A-Z]{2,4}-\d{4,}[-\d]*)',
        ]),
        ("invoice_number", [
            r'(?:invoice)\s*(?:number|no\.?|#)[:\s]*["\']?([A-Z0-9#-]+)["\']?',
        ]),
        ("receipt_number", [
            r'(?:receipt)\s*(?:number|no\.?|#)[:\s]*["\']?([A-Z0-9#-]+)["\']?',
        ]),
        ("customer_name", [
            r'(?:customer|buyer|client|bill.?to)\s*(?:name)?[:\s]*["\']?([A-Z][A-Za-z\s]+?)(?:["\']?\s*(?:\n|\.|$))',
        ]),
        ("supplier_address", [
            r'(?:supplier|merchant|company)\s*(?:address)[:\s]*["\']?(.+?)(?:["\']?\s*\n)',
        ]),
        ("supplier_phone", [
            r'(?:phone|tel|telephone)[:\s]*["\']?([\d+().\s-]{7,20})["\']?',
        ]),
        ("supplier_email", [
            r'(?:email)[:\s]*["\']?([\w.+-]+@[\w-]+\.[\w.-]+)["\']?',
            r'([\w.+-]+@[\w-]+\.[\w.-]+)',  # bare email
        ]),
        ("supplier_website", [
            r'(?:website|url|web)[:\s]*["\']?(https?://\S+|www\.\S+)["\']?',
        ]),
    ]

    # Currency symbol to code mapping
    symbol_to_code = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY"}

    for field, patterns in field_patterns:
        for pattern in patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip().rstrip(".,;:")
                
                # Convert currency symbols
                if field == "document_currency" and value in symbol_to_code:
                    value = symbol_to_code[value]
                
                # Convert numeric fields
                if field in ("document_total_amount", "document_total_net", "document_total_tax"):
                    try:
                        value = float(value.replace(",", ""))
                    except ValueError:
                        continue
                
                recovered[field] = value
                break

    # ── Line items extraction ──
    # Look for patterns like: "2.0 x SOMETHING ... €71.00" or "Description | Qty | Price | Amount"
    line_item_patterns = [
        # "2.0 x ITEM_NAME ... €71.00"
        r'(\d+\.?\d*)\s*x\s+(.+?)\s+[€$£¥]?\s*(\d[\d,]*\.\d{2})',
        # "ITEM_NAME ... qty: 2 ... €71.00"
        r'["\'](.+?)["\'].*?(?:qty|quantity)[:\s]*(\d+).*?[€$£¥]?\s*(\d[\d,]*\.\d{2})',
    ]

    items = []
    for pattern in line_item_patterns:
        for match in re.finditer(pattern, raw_text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 3:
                try:
                    # Pattern 1: qty, description, amount
                    if groups[0].replace(".", "").isdigit():
                        items.append({
                            "description": groups[1].strip(),
                            "quantity": float(groups[0]),
                            "unit_price": 0.0,
                            "amount": float(groups[2].replace(",", "")),
                            "tax_rate": "",
                            "tax_amount": 0.0,
                        })
                    else:
                        items.append({
                            "description": groups[0].strip(),
                            "quantity": float(groups[1]),
                            "unit_price": 0.0,
                            "amount": float(groups[2].replace(",", "")),
                            "tax_rate": "",
                            "tax_amount": 0.0,
                        })
                except (ValueError, IndexError):
                    continue

    if items:
        recovered["line_items"] = items

    return recovered


def build_retry_prompt(raw_reasoning: str, schema: Dict) -> str:
    """
    Build a condensed text-only retry prompt.
    
    Takes the model's verbose reasoning output and asks it to just
    convert that into the target JSON schema. This is cheap because
    it's text-only (no image re-encoding).
    """
    # Truncate reasoning to avoid exceeding context
    max_reasoning_chars = 3000
    truncated = raw_reasoning[:max_reasoning_chars]
    if len(raw_reasoning) > max_reasoning_chars:
        truncated += "\n... [truncated]"

    # Minimal schema (just field names, no descriptions)
    minimal_schema = {}
    for key, val in schema.items():
        if isinstance(val, list):
            minimal_schema[key] = val
        elif isinstance(val, (int, float)):
            minimal_schema[key] = val
        else:
            minimal_schema[key] = ""

    return f"""Convert the following document analysis into a JSON object.
Return ONLY the raw JSON — no markdown, no explanation, no ```json``` fences.

Analysis:
{truncated}

Target JSON schema:
{json.dumps(minimal_schema, indent=2)}

JSON:"""


def try_parse_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Try multiple strategies to extract JSON from a model response.
    
    1. Direct parse (response is pure JSON)
    2. Find outermost {...} block
    3. Strip markdown fences then parse
    4. Find JSON after common prefixes like "JSON:" or "```json"
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find outermost balanced braces
    # (handles cases where there's text before/after the JSON)
    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 4: greedy regex (original approach, as last resort)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None