#!/usr/bin/env python3
"""
Generic Receipt/Invoice OCR Pipeline - Extended Schema
Extracts structured data from PDF receipts using vision-language models.
Designed to be model-agnostic - easily swap OCR backends.

Supports multi-page PDFs with intelligent result merging.

Extended schema includes:
- Core document details (supplier, date, time, amounts, category)
- Reference numbers (invoice, receipt, document, PO, reference numbers)
- Customer information (name, ID, addresses, company registrations)
- Supplier information (address, phone, email, website, registrations)
- Line items with tax details (description, qty, price, amount, tax_rate, tax_amount)
- Taxes breakdown (base, rate, code, value)
"""

import json
import mimetypes
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import torch
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig

from parse_recovery import try_parse_json, extract_from_reasoning, build_retry_prompt


class OCRBackend(ABC):
    """Abstract base class for OCR backends - easily swap models"""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from image/PDF page"""
        pass

    @abstractmethod
    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        """Extract structured data according to schema"""
        pass

    def extract_text_only(self, prompt: str) -> Optional[str]:
        """
        Text-only model call (no image). Used for retry on parse failure.
        Override in backends that support it. Returns None if unsupported.
        """
        return None


class OlmOCRBackend(OCRBackend):
    """olmOCR backend - supports both v1 and v2 models
    
    Supported models:
    - allenai/olmOCR-7B-0225-preview (v1, uses Qwen2-VL)
    - allenai/olmOCR-2-7B-1025 (v2, uses Qwen2.5-VL)
    - allenai/olmOCR-2-7B-1025-FP8 (v2 FP8 quantized)
    """

    def __init__(self, model_name: str = "allenai/olmOCR-7B-0225-preview", quantization: str = None):
        print(f"Loading olmOCR model: {model_name}")


        self.torch = torch
        self.process_vision_info = process_vision_info
        self.model_name = model_name


        load_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        # olmOCR-2 uses Qwen2.5-VL, older versions use Qwen2-VL
        if "olmOCR-2" in model_name:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                **load_kwargs
            )
            # olmOCR-2 uses Qwen2.5-VL processor
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True
            )
            print("Using olmOCR-2 (Qwen2.5-VL based)")
        else:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            print("Using olmOCR v1 (Qwen2-VL based)")

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")

    def _call_model(self, image_path: str, prompt: str) -> str:
        """Call the VLM with image and prompt"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def extract_text(self, image_path: str) -> str:
        """Extract raw text from document"""
        prompt = "Extract all text from this document exactly as shown."
        return self._call_model(image_path, prompt)

    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        """Extract structured data using LLM intelligence"""
        prompt = self._build_extraction_prompt(schema)
        response = self._call_model(image_path, prompt)
        return self._parse_json_response(response)

    def _build_extraction_prompt(self, schema: Dict) -> str:
        """Build prompt for structured extraction with extended schema"""
        return f"""Analyze this invoice/receipt document and extract ALL available information.
Return ONLY valid JSON matching this structure:

{json.dumps(schema, indent=2)}

FIELD INSTRUCTIONS:

CORE DOCUMENT DETAILS:
- supplier_name: Company/merchant name that issued this document
- document_date: Date in YYYY-MM-DD format
- document_time: Time if shown (e.g., "13:51", "2:30 PM")
- document_currency: 3-letter currency code (EUR, USD, GBP, etc.)
- document_total_amount: Final total as a number
- document_total_net: Subtotal before tax as a number
- document_total_tax: Total tax amount as a number
- document_category: Main category (services, software, electronics, travel, food, office_supplies, utilities, rent, shipping, retail, transport, miscellaneous, other)
- subcategory: More specific category (e.g., "taxi", "hotel", "restaurant")

DATES:
- due_date: Payment due date in YYYY-MM-DD format if shown
- payment_date: Actual payment date in YYYY-MM-DD format if shown

REFERENCE NUMBERS:
- document_type_extended: Specific document type (e.g., "ride receipt", "tax invoice", "credit note")
- document_number: Primary document/order number
- receipt_number: Receipt number if different from document number
- invoice_number: Invoice number
- reference_numbers: Comma-separated list of any other reference numbers
- po_number: Purchase order number
- locale: Language/locale code (e.g., "en", "de", "fr")

CUSTOMER INFORMATION:
- customer_name: Customer/buyer name
- customer_id: Customer ID or account number
- customer_address: Customer's main address
- billing_address: Billing address if different
- shipping_address: Shipping/delivery address if shown
- customer_company_registrations: Customer's VAT/tax IDs

SUPPLIER INFORMATION:
- supplier_address: Supplier's full address
- supplier_phone: Supplier's phone number
- supplier_email: Supplier's email address
- supplier_website: Supplier's website URL
- supplier_company_registrations: Supplier's VAT/tax registration numbers

LINE ITEMS (array):
Each item should have:
- description: Item/service description
- quantity: Number of units (default 1)
- unit_price: Price per unit
- amount: Line total (quantity × unit_price)
- tax_rate: Tax rate as percentage (e.g., "23%" or 23)
- tax_amount: Tax amount for this line

TAXES (array):
Each tax entry should have:
- base: Taxable base amount
- rate: Tax rate (percentage or decimal)
- code: Tax code if shown (e.g., "VAT", "GST")
- value: Calculated tax value

NOTES:
- notes: Any other relevant information (payment terms, special instructions, etc.)

Extract as much information as possible. Use null for fields not visible in the document.
Return ONLY the JSON, no other text or markdown."""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response using improved parser"""
        parsed = try_parse_json(response)
        if parsed is not None:
            return parsed
        # Return raw for pipeline-level recovery
        return {"error": "parse_failed", "raw_response": response}


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type for a file based on its extension.
    """
    mimetypes.init()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type is None:
        ext = Path(file_path).suffix.lower()
        fallback_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.bmp': 'image/bmp',
        }
        mime_type = fallback_types.get(ext, 'application/octet-stream')
    
    return mime_type


class ReceiptPipeline:
    """Main pipeline for batch processing receipts with multi-page PDF support"""

    # Extended output schema with all fields from the UI
    OUTPUT_SCHEMA = {
        # Core document details
        "file_name": "",
        "supplier_name": "",
        "document_date": "",
        "document_time": "",
        "document_currency": "",
        "document_total_amount": 0.0,
        "document_total_net": 0.0,
        "document_total_tax": 0.0,
        "document_category": "",
        "subcategory": "",
        
        # Dates
        "due_date": "",
        "payment_date": "",
        
        # Reference numbers
        "document_type_extended": "",
        "document_number": "",
        "receipt_number": "",
        "invoice_number": "",
        "reference_numbers": "",
        "po_number": "",
        "locale": "",
        
        # Customer information
        "customer_name": "",
        "customer_id": "",
        "customer_address": "",
        "billing_address": "",
        "shipping_address": "",
        "customer_company_registrations": "",
        
        # Supplier information
        "supplier_address": "",
        "supplier_phone": "",
        "supplier_email": "",
        "supplier_website": "",
        "supplier_company_registrations": "",
        
        # Line items with tax details
        "line_items": [],
        
        # Taxes breakdown
        "taxes": [],
        
        # Notes and metadata
        "notes": "",
        "mime_type": "",
        "labeled_at": ""
    }

    # Line item schema with tax fields
    LINE_ITEM_SCHEMA = {
        "description": "",
        "quantity": 1,
        "unit_price": 0.0,
        "amount": 0.0,
        "tax_rate": "",
        "tax_amount": 0.0
    }

    # Tax entry schema
    TAX_SCHEMA = {
        "base": 0.0,
        "rate": "",
        "code": "",
        "value": 0.0
    }

    def __init__(self, backend: OCRBackend, pdf_dpi: int = 150, max_pages: Optional[int] = None,
                 include_debug_fields: bool = False, retry_on_parse_fail: bool = True):
        """
        Initialize pipeline with an OCR backend

        Args:
            backend: OCR backend instance (OlmOCRBackend, etc.)
            pdf_dpi: DPI for PDF to image conversion (default: 150, lower = less memory)
            max_pages: Maximum number of pages to process (None = all pages)
            include_debug_fields: Include _pages_processed in output (default: False)
            retry_on_parse_fail: Attempt recovery when JSON parsing fails (default: True)
        """
        self.backend = backend
        self.pdf_dpi = pdf_dpi
        self.max_pages = max_pages
        self.include_debug_fields = include_debug_fields
        self.retry_on_parse_fail = retry_on_parse_fail

    def _recover_from_parse_failure(self, data: Dict[str, Any], image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt to recover structured data when initial JSON parse fails.
        
        Strategy order:
        1. Extract fields from reasoning text via regex (free, no model call)
        2. Retry with text-only prompt if backend supports it (cheap)
        3. Return best-effort partial data
        """
        raw = data.get("raw_response", "") or data.get("raw", "")
        if not raw:
            return data

        recovery_method = None

        # Strategy 1: Regex extraction from reasoning text
        recovered = extract_from_reasoning(raw)
        if recovered and any(v for v in recovered.values() if v):
            recovery_method = "regex_extraction"
            print(f"    ↳ Recovered {len(recovered)} fields via regex from reasoning text")

        # Strategy 2: Text-only retry (if backend supports it)
        if not recovered or not recovered.get("supplier_name"):
            retry_prompt = build_retry_prompt(raw, self.OUTPUT_SCHEMA)
            retry_response = self.backend.extract_text_only(retry_prompt)
            
            if retry_response:
                parsed = try_parse_json(retry_response)
                if parsed and "error" not in parsed:
                    recovery_method = "text_retry"
                    print(f"    ↳ Recovered via text-only retry")
                    return parsed
                # If retry also produced reasoning, try regex on that too
                retry_recovered = extract_from_reasoning(retry_response)
                if retry_recovered:
                    for k, v in retry_recovered.items():
                        if v and not recovered.get(k):
                            recovered[k] = v

        if recovered and any(v for v in recovered.values() if v):
            # Merge recovered fields into data, keeping error marker for transparency
            result = dict(data)
            result["error"] = f"parse_failed_recovered_via_{recovery_method or 'regex'}"
            result["raw"] = raw[:500]  # Keep truncated raw for debugging
            # Remove raw_response to avoid duplication
            result.pop("raw_response", None)
            for k, v in recovered.items():
                if v is not None:
                    result[k] = v
            return result

        return data

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single PDF/image file (supports multi-page PDFs)
        """
        file_path = Path(file_path)
        print(f"\nProcessing: {file_path.name}")

        mime_type = get_mime_type(str(file_path))

        if file_path.suffix.lower() == '.pdf':
            image_paths = self._pdf_to_images(str(file_path))

            if len(image_paths) == 1:
                data = self.backend.extract_structured(image_paths[0], self.OUTPUT_SCHEMA)
            else:
                print(f"  Processing {len(image_paths)} pages...")
                page_results = []
                for i, img_path in enumerate(image_paths):
                    print(f"    Page {i + 1}/{len(image_paths)}...")
                    try:
                        page_data = self.backend.extract_structured(img_path, self.OUTPUT_SCHEMA)
                        page_data["_page_number"] = i + 1
                        page_results.append(page_data)
                    except Exception as e:
                        print(f"    Page {i + 1} error: {e}")
                        page_results.append({"error": str(e), "_page_number": i + 1})

                data = self._merge_page_results(page_results)
                if self.include_debug_fields:
                    data["_pages_processed"] = len(image_paths)

            self._cleanup_temp_images(image_paths)
        else:
            data = self.backend.extract_structured(str(file_path), self.OUTPUT_SCHEMA)

        # ── Parse failure recovery ──
        if self.retry_on_parse_fail and "error" in data and "parse_failed" in str(data.get("error", "")):
            data = self._recover_from_parse_failure(data, str(file_path))

        data = self._normalize_output(data)
        data["file_name"] = file_path.name
        data["mime_type"] = mime_type
        data["labeled_at"] = datetime.now().isoformat()

        return data

    def _normalize_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize output to ensure consistent snake_case field names
        and proper data structure.
        """
        # Extended field mapping including new fields
        field_mapping = {
            # camelCase -> snake_case
            "supplierName": "supplier_name",
            "documentDate": "document_date",
            "documentTime": "document_time",
            "invoiceNumber": "invoice_number",
            "currency": "document_currency",
            "totalAmount": "document_total_amount",
            "subtotal": "document_total_net",
            "taxAmount": "document_total_tax",
            "category": "document_category",
            "lineItems": "line_items",
            "filename": "file_name",
            "labeledAt": "labeled_at",
            "mimeType": "mime_type",
            "dueDate": "due_date",
            "paymentDate": "payment_date",
            "documentTypeExtended": "document_type_extended",
            "documentNumber": "document_number",
            "receiptNumber": "receipt_number",
            "referenceNumbers": "reference_numbers",
            "poNumber": "po_number",
            "customerName": "customer_name",
            "customerId": "customer_id",
            "customerAddress": "customer_address",
            "billingAddress": "billing_address",
            "shippingAddress": "shipping_address",
            "customerCompanyRegistrations": "customer_company_registrations",
            "supplierAddress": "supplier_address",
            "supplierPhone": "supplier_phone",
            "supplierEmail": "supplier_email",
            "supplierWebsite": "supplier_website",
            "supplierCompanyRegistrations": "supplier_company_registrations",
            # snake_case passthrough
            "supplier_name": "supplier_name",
            "document_date": "document_date",
            "document_time": "document_time",
            "invoice_number": "invoice_number",
            "document_currency": "document_currency",
            "document_total_amount": "document_total_amount",
            "document_total_net": "document_total_net",
            "document_total_tax": "document_total_tax",
            "document_category": "document_category",
            "subcategory": "subcategory",
            "line_items": "line_items",
            "file_name": "file_name",
            "labeled_at": "labeled_at",
            "mime_type": "mime_type",
            "notes": "notes",
            "taxes": "taxes",
            "due_date": "due_date",
            "payment_date": "payment_date",
            "document_type_extended": "document_type_extended",
            "document_number": "document_number",
            "receipt_number": "receipt_number",
            "reference_numbers": "reference_numbers",
            "po_number": "po_number",
            "locale": "locale",
            "customer_name": "customer_name",
            "customer_id": "customer_id",
            "customer_address": "customer_address",
            "billing_address": "billing_address",
            "shipping_address": "shipping_address",
            "customer_company_registrations": "customer_company_registrations",
            "supplier_address": "supplier_address",
            "supplier_phone": "supplier_phone",
            "supplier_email": "supplier_email",
            "supplier_website": "supplier_website",
            "supplier_company_registrations": "supplier_company_registrations",
        }
        
        normalized = {}
        
        for old_key, value in data.items():
            if old_key.startswith("_"):
                normalized[old_key] = value
            elif old_key in field_mapping:
                new_key = field_mapping[old_key]
                normalized[new_key] = value
            else:
                normalized[old_key] = value
        
        # Normalize line_items
        if "line_items" in normalized and isinstance(normalized["line_items"], list):
            normalized["line_items"] = self._normalize_line_items(normalized["line_items"])
        
        # Normalize taxes
        if "taxes" in normalized and isinstance(normalized["taxes"], list):
            normalized["taxes"] = self._normalize_taxes(normalized["taxes"])
        
        # Ensure all required fields exist with defaults
        defaults = {
            # Core
            "supplier_name": None,
            "document_date": None,
            "document_time": None,
            "document_currency": None,
            "document_total_amount": 0.0,
            "document_total_net": 0.0,
            "document_total_tax": 0.0,
            "document_category": "other",
            "subcategory": None,
            # Dates
            "due_date": None,
            "payment_date": None,
            # Reference numbers
            "document_type_extended": None,
            "document_number": None,
            "receipt_number": None,
            "invoice_number": None,
            "reference_numbers": None,
            "po_number": None,
            "locale": None,
            # Customer
            "customer_name": None,
            "customer_id": None,
            "customer_address": None,
            "billing_address": None,
            "shipping_address": None,
            "customer_company_registrations": None,
            # Supplier
            "supplier_address": None,
            "supplier_phone": None,
            "supplier_email": None,
            "supplier_website": None,
            "supplier_company_registrations": None,
            # Arrays
            "line_items": [],
            "taxes": [],
            # Other
            "notes": "",
            "file_name": "",
            "mime_type": "",
            "labeled_at": "",
        }
        
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        return normalized

    def _normalize_line_items(self, line_items: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize line items to consistent structure including tax fields.
        """
        normalized_items = []
        
        for item in line_items:
            if not isinstance(item, dict):
                continue
                
            normalized_item = {
                "description": item.get("description", ""),
                "quantity": item.get("quantity", 1),
                "unit_price": 0.0,
                "amount": 0.0,
                "tax_rate": item.get("tax_rate") or item.get("taxRate") or "",
                "tax_amount": 0.0,
            }
            
            # Handle various price field names
            price_fields = ["unit_price", "unitPrice", "price", "rate"]
            for field in price_fields:
                if field in item and item[field] is not None:
                    try:
                        normalized_item["unit_price"] = float(item[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Handle various amount field names
            amount_fields = ["amount", "total", "line_total", "lineTotal"]
            for field in amount_fields:
                if field in item and item[field] is not None:
                    try:
                        normalized_item["amount"] = float(item[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Handle tax amount
            tax_amount_fields = ["tax_amount", "taxAmount", "tax"]
            for field in tax_amount_fields:
                if field in item and item[field] is not None:
                    try:
                        normalized_item["tax_amount"] = float(item[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Calculate amount if missing
            if normalized_item["amount"] == 0.0 and normalized_item["unit_price"] > 0:
                try:
                    qty = float(normalized_item["quantity"]) if normalized_item["quantity"] else 1
                    normalized_item["amount"] = normalized_item["unit_price"] * qty
                except (ValueError, TypeError):
                    pass
            
            # Calculate unit_price if missing
            if normalized_item["unit_price"] == 0.0 and normalized_item["amount"] > 0:
                try:
                    qty = float(normalized_item["quantity"]) if normalized_item["quantity"] else 1
                    if qty > 0:
                        normalized_item["unit_price"] = normalized_item["amount"] / qty
                except (ValueError, TypeError):
                    pass
            
            normalized_items.append(normalized_item)
        
        return normalized_items

    def _normalize_taxes(self, taxes: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize tax entries to consistent structure.
        """
        normalized_taxes = []
        
        for tax in taxes:
            if not isinstance(tax, dict):
                continue
            
            normalized_tax = {
                "base": 0.0,
                "rate": tax.get("rate") or "",
                "code": tax.get("code") or "",
                "value": 0.0,
            }
            
            # Handle base amount
            base_fields = ["base", "taxableBase", "taxable_base", "net"]
            for field in base_fields:
                if field in tax and tax[field] is not None:
                    try:
                        normalized_tax["base"] = float(tax[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Handle tax value
            value_fields = ["value", "amount", "tax", "taxAmount", "tax_amount"]
            for field in value_fields:
                if field in tax and tax[field] is not None:
                    try:
                        normalized_tax["value"] = float(tax[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            normalized_taxes.append(normalized_tax)
        
        return normalized_taxes

    def _merge_page_results(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Intelligently merge structured data from multiple pages.
        """
        merged = {
            # Core
            "supplier_name": None,
            "document_date": None,
            "document_time": None,
            "document_currency": None,
            "document_total_amount": 0.0,
            "document_total_net": 0.0,
            "document_total_tax": 0.0,
            "document_category": None,
            "subcategory": None,
            # Dates
            "due_date": None,
            "payment_date": None,
            # Reference numbers
            "document_type_extended": None,
            "document_number": None,
            "receipt_number": None,
            "invoice_number": None,
            "reference_numbers": None,
            "po_number": None,
            "locale": None,
            # Customer
            "customer_name": None,
            "customer_id": None,
            "customer_address": None,
            "billing_address": None,
            "shipping_address": None,
            "customer_company_registrations": None,
            # Supplier
            "supplier_address": None,
            "supplier_phone": None,
            "supplier_email": None,
            "supplier_website": None,
            "supplier_company_registrations": None,
            # Arrays
            "line_items": [],
            "taxes": [],
            "notes": "",
        }

        # All string fields where we take first non-empty value
        first_value_fields = [
            "supplier_name", "document_date", "document_time", "document_currency", 
            "document_category", "subcategory", "due_date", "payment_date",
            "document_type_extended", "document_number", "receipt_number", 
            "invoice_number", "reference_numbers", "po_number", "locale",
            "customer_name", "customer_id", "customer_address", "billing_address",
            "shipping_address", "customer_company_registrations",
            "supplier_address", "supplier_phone", "supplier_email", 
            "supplier_website", "supplier_company_registrations"
        ]
        
        # camelCase alternatives
        camel_case_mapping = {
            "supplier_name": "supplierName",
            "document_date": "documentDate",
            "document_time": "documentTime",
            "document_currency": "currency",
            "document_category": "category",
            "due_date": "dueDate",
            "payment_date": "paymentDate",
            "document_type_extended": "documentTypeExtended",
            "document_number": "documentNumber",
            "receipt_number": "receiptNumber",
            "invoice_number": "invoiceNumber",
            "reference_numbers": "referenceNumbers",
            "po_number": "poNumber",
            "customer_name": "customerName",
            "customer_id": "customerId",
            "customer_address": "customerAddress",
            "billing_address": "billingAddress",
            "shipping_address": "shippingAddress",
            "customer_company_registrations": "customerCompanyRegistrations",
            "supplier_address": "supplierAddress",
            "supplier_phone": "supplierPhone",
            "supplier_email": "supplierEmail",
            "supplier_website": "supplierWebsite",
            "supplier_company_registrations": "supplierCompanyRegistrations",
        }

        # Numeric fields where we take maximum value
        max_value_fields = ["document_total_amount", "document_total_net", "document_total_tax"]
        camel_max_mapping = {
            "document_total_amount": "totalAmount",
            "document_total_net": "subtotal",
            "document_total_tax": "taxAmount",
        }

        all_notes = []
        seen_line_items = set()
        seen_taxes = set()

        for page_data in page_results:
            if "error" in page_data and len(page_data) <= 3:
                continue

            # First non-empty value for string fields
            for field in first_value_fields:
                if merged[field] is None or merged[field] == "":
                    value = page_data.get(field)
                    if (value is None or value == "") and field in camel_case_mapping:
                        value = page_data.get(camel_case_mapping[field])
                    if value and value != "" and value != "null":
                        merged[field] = value

            # Maximum value for amounts
            for field in max_value_fields:
                value = page_data.get(field)
                if value is None and field in camel_max_mapping:
                    value = page_data.get(camel_max_mapping[field])
                if value is not None:
                    try:
                        num_value = float(value) if not isinstance(value, (int, float)) else value
                        if num_value > merged[field]:
                            merged[field] = num_value
                    except (ValueError, TypeError):
                        pass

            # Concatenate line items with deduplication
            line_items = page_data.get("line_items") or page_data.get("lineItems", [])
            if isinstance(line_items, list):
                for item in line_items:
                    if isinstance(item, dict):
                        normalized_item = {
                            "description": item.get("description", ""),
                            "quantity": item.get("quantity", 1),
                            "unit_price": item.get("unit_price") or item.get("unitPrice") or item.get("price") or 0,
                            "amount": item.get("amount") or item.get("total") or 0,
                            "tax_rate": item.get("tax_rate") or item.get("taxRate") or "",
                            "tax_amount": item.get("tax_amount") or item.get("taxAmount") or 0
                        }
                        item_key = (
                            normalized_item["description"],
                            normalized_item["quantity"],
                            normalized_item["unit_price"]
                        )
                        if item_key not in seen_line_items and item_key[0]:
                            seen_line_items.add(item_key)
                            merged["line_items"].append(normalized_item)

            # Concatenate taxes with deduplication
            taxes = page_data.get("taxes", [])
            if isinstance(taxes, list):
                for tax in taxes:
                    if isinstance(tax, dict):
                        normalized_tax = {
                            "base": tax.get("base") or 0,
                            "rate": tax.get("rate") or "",
                            "code": tax.get("code") or "",
                            "value": tax.get("value") or tax.get("amount") or 0
                        }
                        tax_key = (normalized_tax["base"], normalized_tax["rate"], normalized_tax["code"])
                        if tax_key not in seen_taxes:
                            seen_taxes.add(tax_key)
                            merged["taxes"].append(normalized_tax)

            # Collect notes
            notes = page_data.get("notes", "")
            if notes and notes.strip():
                all_notes.append(notes.strip())

        # Combine notes
        seen_notes = set()
        unique_notes = []
        for note in all_notes:
            if note not in seen_notes:
                seen_notes.add(note)
                unique_notes.append(note)
        merged["notes"] = " | ".join(unique_notes) if unique_notes else ""

        # Set defaults
        if merged["document_category"] is None:
            merged["document_category"] = "other"

        return merged

    def process_folder(self, folder_path: str, output_path: str = "results.json") -> List[Dict[str, Any]]:
        """Process all receipts in a folder"""
        folder = Path(folder_path)
        results = []

        files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
        files += list(folder.glob("*.png")) + list(folder.glob("*.PNG"))
        files += list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
        files += list(folder.glob("*.jpeg")) + list(folder.glob("*.JPEG"))

        print(f"Found {len(files)} files to process")

        for file_path in files:
            try:
                data = self.process_file(str(file_path))
                results.append(data)
                print(f"  {file_path.name}: {data.get('supplier_name', 'Unknown')} - {data.get('document_currency', '')} {data.get('document_total_amount', 0)}")
            except Exception as e:
                print(f"  {file_path.name}: ERROR - {e}")
                results.append({
                    "file_name": file_path.name,
                    "mime_type": get_mime_type(str(file_path)),
                    "error": str(e),
                    "labeled_at": datetime.now().isoformat()
                })

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")
        return results

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert all pages of a PDF to images for OCR"""
        try:
            import fitz

            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            pages_to_process = total_pages
            if self.max_pages is not None:
                pages_to_process = min(total_pages, self.max_pages)

            print(f"  PDF has {total_pages} pages, processing {pages_to_process}")

            image_paths = []
            for page_num in range(pages_to_process):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.pdf_dpi)

                temp_path = f"/tmp/{Path(pdf_path).stem}_page{page_num}.png"
                pix.save(temp_path)
                image_paths.append(temp_path)

                print(f"    Page {page_num + 1}: {pix.width}x{pix.height}px")

            doc.close()
            return image_paths

        except ImportError:
            print("PyMuPDF not installed. Install with: pip install PyMuPDF")
            raise

    def _cleanup_temp_images(self, image_paths: List[str]):
        """Remove temporary image files"""
        import os
        for path in image_paths:
            try:
                if os.path.exists(path) and path.startswith("/tmp/"):
                    os.remove(path)
            except OSError:
                pass


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Receipt OCR Pipeline")
    parser.add_argument("input", help="Input file or folder path")
    parser.add_argument("-o", "--output", default="receipts_output.json", help="Output JSON file")
    parser.add_argument("--model", default="olmocr", choices=["olmocr"], help="OCR model to use")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to process per PDF (default: all)")

    args = parser.parse_args()

    if args.model == "olmocr":
        backend = OlmOCRBackend()

    pipeline = ReceiptPipeline(backend, max_pages=args.max_pages)

    input_path = Path(args.input)
    if input_path.is_dir():
        results = pipeline.process_folder(str(input_path), args.output)
    else:
        result = pipeline.process_file(str(input_path))
        with open(args.output, 'w') as f:
            json.dump([result], f, indent=2)
        print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()