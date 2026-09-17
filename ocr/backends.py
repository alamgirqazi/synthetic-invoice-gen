#!/usr/bin/env python3
"""
OCR Backend implementations for Receipt Pipeline - Extended Schema

Supports extraction of:
- Core document details (supplier, date, time, amounts, category)
- Reference numbers (invoice, receipt, document, PO, reference numbers)
- Customer information (name, ID, addresses, company registrations)
- Supplier information (address, phone, email, website, registrations)
- Line items with tax details (description, qty, price, amount, tax_rate, tax_amount)
- Taxes breakdown (base, rate, code, value)

Supported backends:
- Nanonets OCR2-3B (nanonets/Nanonets-OCR2-3B) — supports quantization
- GLM-OCR (zai-org/GLM-OCR) — HF + vLLM
- HunyuanOCR (tencent/HunyuanOCR) — HF + vLLM
- Qwen2.5-VL-7B (Qwen/Qwen2.5-VL-7B-Instruct) — supports quantization
- Qwen3.5-VL (Qwen/Qwen3.5-9B, Qwen3.5-0.8B, etc.) — via vLLM
- SmolVLM (HuggingFaceTB/SmolVLM2-2.2B-Instruct, etc.) — via vLLM

Quantization support (4bit/8bit via bitsandbytes):
    pip install bitsandbytes
    python run.py /path --backend qwen --quantization 4bit
"""

import json
import re
import base64
from typing import Dict, Any, List, Optional
from ocr_pipeline import OCRBackend
from parse_recovery import try_parse_json


def get_quantization_config(quantization: Optional[str] = None):
    """
    Create BitsAndBytesConfig for 4-bit or 8-bit quantization.
    
    Args:
        quantization: "4bit", "8bit", or None (no quantization / default bf16)
    
    Returns:
        BitsAndBytesConfig or None
    
    Requirements:
        pip install bitsandbytes
    """
    if quantization is None:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        import torch
    except ImportError:
        raise RuntimeError("bitsandbytes required: pip install bitsandbytes")
    
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}. Use '4bit' or '8bit'.")


def build_extended_extraction_prompt(schema: Dict) -> str:
    """Build comprehensive extraction prompt for all backends"""
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
- supplier_company_registrations: Supplier's VAT/tax registration numbers (e.g., "IE 98240108")

LINE ITEMS (array):
Each item should have:
- description: Item/service description
- quantity: Number of units (default 1)
- unit_price: Price per unit
- amount: Line total (quantity × unit_price)
- tax_rate: Tax rate as string (e.g., "23%", "0%")
- tax_amount: Tax amount for this line

TAXES (array):
Each tax entry should have:
- base: Taxable base amount
- rate: Tax rate (e.g., "0", "23%")
- code: Tax code if shown (e.g., "VAT", "GST", "S1")
- value: Calculated tax value

NOTES:
- notes: Any other relevant information (payment terms, special instructions, etc.)

Extract as much information as possible from the document.
Use null for fields not visible in the document.
Return ONLY the JSON, no other text or markdown."""


class NanonetsBackend(OCRBackend):
    """
    Nanonets OCR2-3B backend (nanonets/Nanonets-OCR2-3B)
    
    Based on Qwen2.5-VL, good for structured document extraction.
    """

    def __init__(self, model_name: str = "nanonets/Nanonets-OCR2-3B", max_new_tokens: int = 4096, max_image_size: int = 1024, quantization: str = None):
        print(f"Loading Nanonets model: {model_name}")
        if quantization:
            print(f"  Quantization: {quantization}")

        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens
        self.max_image_size = max_image_size
        self.model_name = model_name

        attn_impl = None
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
            
            quant_config = get_quantization_config(quantization)
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16
            
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                )
                attn_impl = "flash_attention_2"
            except Exception as e:
                print(f"Flash attention not available: {e}")
                print("Falling back to default attention...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    **load_kwargs,
                )
                attn_impl = "default"
        except Exception as e:
            raise RuntimeError(f"Failed to load Nanonets model: {e}")

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device} with {attn_impl} attention")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Max image size: {self.max_image_size}px")

    def _resize_image(self, image: 'Image.Image') -> 'Image.Image':
        """Resize PIL Image if too large to prevent OOM"""
        width, height = image.size
        
        if max(width, height) <= self.max_image_size:
            return image
        
        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))
        
        print(f"  Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), self.Image.LANCZOS)

    def _call_model(self, image_path: str, prompt: str) -> str:
        """Call the model following official Nanonets example"""
        
        image = self.Image.open(image_path).convert("RGB")
        image = self._resize_image(image)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts information from documents."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False
            )

        generated_ids = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        del inputs, output_ids, generated_ids
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        
        return output_text[0]

    def _call_model_text_only(self, prompt: str) -> str:
        """Text-only model call for retry (no image)"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        del inputs, output_ids, generated_ids
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        return output_text[0]

    def extract_text_only(self, prompt: str) -> Optional[str]:
        """Text-only retry for parse failure recovery"""
        try:
            return self._call_model_text_only(prompt)
        except Exception as e:
            print(f"    ↳ Text-only retry failed: {e}")
            return None

    def extract_text(self, image_path: str) -> str:
        """Extract raw text using official Nanonets prompt"""
        prompt = """Extract the text from the above document as if you were reading it naturally.
Return the tables in html format. Return the equations in LaTeX representation.
If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.
Watermarks should be wrapped in brackets."""
        return self._call_model(image_path, prompt)

    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        """Extract structured invoice data with extended schema"""
        prompt = build_extended_extraction_prompt(schema)
        response = self._call_model(image_path, prompt)
        
        parsed = try_parse_json(response)
        if parsed is not None:
            return parsed
        
        return {"error": "parse_failed", "raw": response[:2000]}


# ---------------------------------------------------------------------------
# Qwen2.5-VL-7B Backend (Qwen/Qwen2.5-VL-7B-Instruct)
# ---------------------------------------------------------------------------

class Qwen25VLBackend(OCRBackend):
    """
    Qwen2.5-VL-7B-Instruct backend (Qwen/Qwen2.5-VL-7B-Instruct)
    
    The 7B general-purpose vision-language model that many OCR models
    (olmOCR, Nanonets) are fine-tuned from. Excellent for structured
    data extraction from invoices/receipts out of the box.
    
    Good for fair 7B-vs-7B comparisons against olmOCR.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", max_new_tokens: int = 4096, max_image_size: int = 1024, quantization: str = None):
        print(f"Loading Qwen2.5-VL model: {model_name}")
        if quantization:
            print(f"  Quantization: {quantization}")

        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens
        self.max_image_size = max_image_size
        self.model_name = model_name

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            quant_config = get_quantization_config(quantization)
            load_kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
                print(f"  Using {quantization} quantization via bitsandbytes")
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16

            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                )
                print("Using flash_attention_2")
            except Exception as e:
                print(f"Flash attention not available: {e}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **load_kwargs,
                )

            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen2.5-VL model: {e}\n"
                "Install: pip install transformers qwen-vl-utils accelerate"
            )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
        print(f"Max new tokens: {self.max_new_tokens}")

    def _resize_image(self, image: 'Image.Image') -> 'Image.Image':
        """Resize PIL Image if too large to prevent OOM"""
        width, height = image.size
        if max(width, height) <= self.max_image_size:
            return image

        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))

        print(f"  Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), self.Image.LANCZOS)

    def _call_model(self, image_path: str, prompt: str) -> str:
        """Call model using official Qwen2.5-VL pattern with qwen_vl_utils"""
        from qwen_vl_utils import process_vision_info
        import tempfile
        import os

        image = self.Image.open(image_path).convert("RGB")
        image = self._resize_image(image)

        # Save resized image to temp file for qwen_vl_utils
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

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
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            del inputs, generated_ids, generated_ids_trimmed
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

            return output_text[0]

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def extract_text(self, image_path: str) -> str:
        """Extract raw text from document"""
        return self._call_model(image_path, "Extract all text from this document exactly as shown.")

    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        """Extract structured data with extended schema"""
        prompt = build_extended_extraction_prompt(schema)
        response = self._call_model(image_path, prompt)

        parsed = try_parse_json(response)
        if parsed is not None:
            return parsed

        return {"error": "parse_failed", "raw": response[:2000]}


# ---------------------------------------------------------------------------
# GLM-OCR Backend (zai-org/GLM-OCR)
# ---------------------------------------------------------------------------

class GLMOCRBackend(OCRBackend):
    """
    GLM-OCR backend (zai-org/GLM-OCR)
    
    Lightweight 0.9B parameter model, #1 on OmniDocBench V1.5.
    Built on GLM-V encoder-decoder architecture with CogViT visual encoder.
    Natively supports JSON-schema information extraction prompts.
    
    Requirements:
        pip install git+https://github.com/huggingface/transformers.git
    
    Note: Requires latest transformers from git (GlmOcr model class).
    """

    def __init__(self, model_name: str = "zai-org/GLM-OCR", max_new_tokens: int = 8192, max_image_size: int = 1024):
        print(f"Loading GLM-OCR model: {model_name}")

        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens
        self.max_image_size = max_image_size
        self.model_name = model_name

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GLM-OCR model: {e}\n"
                "GLM-OCR requires latest transformers from git:\n"
                "  pip install git+https://github.com/huggingface/transformers.git"
            )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
        print(f"Max new tokens: {self.max_new_tokens}")

    def _resize_image(self, image: 'Image.Image') -> 'Image.Image':
        """Resize PIL Image if too large to prevent OOM"""
        width, height = image.size
        if max(width, height) <= self.max_image_size:
            return image
        
        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))
        
        print(f"  Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), self.Image.LANCZOS)

    def _save_temp_image(self, image: 'Image.Image') -> str:
        """Save PIL image to temp file and return the path"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            return tmp.name

    def _call_model(self, image_path: str, prompt: str) -> str:
        """
        Call GLM-OCR model following official transformers usage.
        """
        import os

        image = self.Image.open(image_path).convert("RGB")
        image = self._resize_image(image)
        
        tmp_path = self._save_temp_image(image)
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": tmp_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            inputs.pop("token_type_ids", None)

            with self.torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )

            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            del inputs, generated_ids
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

            return output_text
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def extract_text(self, image_path: str) -> str:
        return self._call_model(image_path, "Text Recognition:")

    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        prompt = build_extended_extraction_prompt(schema)
        response = self._call_model(image_path, prompt)
        
        parsed = try_parse_json(response)
        if parsed is not None:
            return parsed
        
        return {"error": "parse_failed", "raw": response[:2000]}


# ---------------------------------------------------------------------------
# Generic vLLM Backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

class VLLMBackend(OCRBackend):
    """
    Generic vLLM backend for any vision-language model served via vLLM.
    
    Works with any model that vLLM can serve with OpenAI-compatible API:
    - Qwen3.5 VL (Qwen/Qwen3.5-9B, Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-27B)
    - SmolVLM (HuggingFaceTB/SmolVLM2-2.2B-Instruct, etc.)
    - GLM-OCR (zai-org/GLM-OCR)
    - HunyuanOCR (tencent/HunyuanOCR)
    - Qwen3-VL-8B-Thinking
    - Any other vLLM-served VLM
    
    Start server examples:
        # Qwen3.5-9B
        vllm serve Qwen/Qwen3.5-9B --max-model-len 8192 --gpu-memory-utilization 0.85
        
        # Qwen3.5-0.8B (fits on smaller GPUs)
        vllm serve Qwen/Qwen3.5-0.8B --max-model-len 8192
        
        # SmolVLM2
        vllm serve HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-model-len 8192
        
        # HunyuanOCR
        vllm serve tencent/HunyuanOCR --no-enable-prefix-caching --mm-processor-cache-gb 0
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "default",
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        text_extraction_prompt: Optional[str] = None,
        clean_repeated: bool = False,
        use_json_mode: bool = True,
    ):
        """
        Args:
            base_url: vLLM server URL (OpenAI-compatible endpoint)
            model_name: Model name as registered in vLLM
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt (some models like HunyuanOCR need "")
            text_extraction_prompt: Custom prompt for raw text extraction
            clean_repeated: Apply repeated-substring cleaning (needed for HunyuanOCR)
            use_json_mode: Use response_format=json_object for structured extraction (default: True)
        """
        print(f"Using vLLM backend at {base_url}")
        print(f"  Model: {model_name}")
        
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")
        
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.text_extraction_prompt = text_extraction_prompt or "Extract all text from this document exactly as shown."
        self.clean_repeated = clean_repeated
        self.use_json_mode = use_json_mode
        
        # Test if JSON mode is actually supported by this vLLM server
        if use_json_mode:
            self._json_mode_available = self._test_json_mode()
            if self._json_mode_available:
                print(f"  JSON mode: enabled (structured extraction will use guided decoding)")
            else:
                print(f"  JSON mode: not available (falling back to standard prompting)")
        else:
            self._json_mode_available = False
            
        print(f"  Max tokens: {max_tokens}")

    def _test_json_mode(self) -> bool:
        """Test if the vLLM server supports response_format=json_object"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": 'Return {"test": true}'}],
                max_tokens=20,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return True
        except Exception:
            return False

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        ext = image_path.lower().split('.')[-1]
        mime_map = {
            'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp',
        }
        return mime_map.get(ext, 'image/png')

    def _call_model(self, image_path: str, prompt: str, force_json: bool = False) -> str:
        base64_image = self._image_to_base64(image_path)
        mime_type = self._get_mime_type(image_path)
        
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ]
        })
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        
        # Use JSON mode for structured extraction if available
        if force_json and self._json_mode_available:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        result = response.choices[0].message.content
        
        if self.clean_repeated:
            result = _clean_repeated_substrings(result)
        
        return result

    def _call_text_only(self, prompt: str) -> str:
        """Text-only model call (no image) for retry on parse failure"""
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        
        if self._json_mode_available:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def extract_text_only(self, prompt: str) -> Optional[str]:
        """Text-only retry for parse failure recovery"""
        try:
            return self._call_text_only(prompt)
        except Exception as e:
            print(f"    ↳ Text-only retry failed: {e}")
            return None

    def extract_text(self, image_path: str) -> str:
        return self._call_model(image_path, self.text_extraction_prompt)

    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        prompt = build_extended_extraction_prompt(schema)
        response = self._call_model(image_path, prompt, force_json=True)
        
        parsed = try_parse_json(response)
        if parsed is not None:
            return parsed

        return {"error": "parse_failed", "raw": response[:2000]}


# ---------------------------------------------------------------------------
# Legacy aliases (kept for backward compat, delegate to VLLMBackend)
# ---------------------------------------------------------------------------

class GLMOCRVLLMBackend(VLLMBackend):
    """GLM-OCR via vLLM server (legacy alias)"""

    def __init__(self, base_url: str = "http://localhost:8080/v1", model_name: str = "zai-org/GLM-OCR", max_tokens: int = 8192):
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
            text_extraction_prompt="Text Recognition:",
        )


class HunyuanVLLMBackend(VLLMBackend):
    """HunyuanOCR via vLLM server (legacy alias)"""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = "tencent/HunyuanOCR", max_tokens: int = 16384):
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
            system_prompt="",
            text_extraction_prompt=(
                "Extract all information from the document image in markdown format, "
                "ignoring headers and footers. Tables should be expressed in HTML format, "
                "formulas in the document should be represented using LaTeX, "
                "parsed in reading order."
            ),
            clean_repeated=True,
        )


# ---------------------------------------------------------------------------
# HunyuanOCR HF Backend (kept for completeness)
# ---------------------------------------------------------------------------

def _clean_repeated_substrings(text: str) -> str:
    """Clean repeated substrings in HunyuanOCR output."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


class HunyuanOCRBackend(OCRBackend):
    """
    HunyuanOCR backend (tencent/HunyuanOCR) — HuggingFace direct loading.
    
    WARNING: Requires a specific pinned transformers commit.
    For most cases, use hunyuan-vllm instead.
    """

    def __init__(self, model_name: str = "tencent/HunyuanOCR", max_new_tokens: int = 16384, max_image_size: int = 1024):
        print(f"Loading HunyuanOCR model: {model_name}")

        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens
        self.max_image_size = max_image_size
        self.model_name = model_name

        try:
            from transformers import AutoProcessor
            from transformers import HunYuanVLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="eager",
                dtype=torch.bfloat16,
                device_map="auto"
            )
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import HunyuanOCR classes: {e}\n"
                "HunyuanOCR requires a pinned transformers commit:\n"
                "  pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4\n"
                "Alternatively, use the hunyuan-vllm backend."
            )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")

    def _resize_image(self, image: 'Image.Image') -> 'Image.Image':
        width, height = image.size
        if max(width, height) <= self.max_image_size:
            return image
        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))
        print(f"  Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), self.Image.LANCZOS)

    def _call_model(self, image_path: str, prompt: str) -> str:
        image = self.Image.open(image_path).convert("RGB")
        image = self._resize_image(image)

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]}
        ]

        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        inputs = self.processor(text=texts, images=image, padding=True, return_tensors="pt")

        with self.torch.no_grad():
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        result = output_text[0] if isinstance(output_text, list) else output_text
        result = _clean_repeated_substrings(result)

        del inputs, generated_ids, generated_ids_trimmed
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return result

    def extract_text(self, image_path: str) -> str:
        return self._call_model(image_path,
            "Extract all information from the document image in markdown format, "
            "ignoring headers and footers. Tables should be expressed in HTML format, "
            "formulas in the document should be represented using LaTeX, parsed in reading order.")

    def extract_structured(self, image_path: str, schema: Dict) -> Dict[str, Any]:
        prompt = build_extended_extraction_prompt(schema)
        try:
            response = self._call_model(image_path, prompt)
        except Exception as e:
            return {"error": f"model_call_failed: {e}", "raw": ""}
        if not response or not response.strip():
            return {"error": "empty_response", "raw": ""}
        
        parsed = try_parse_json(response)
        if parsed is not None:
            return parsed
        
        return {"error": "parse_failed", "raw": response[:2000]}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

# Default model names for each backend key
BACKEND_DEFAULTS = {
    "nanonets":       {"model": "nanonets/Nanonets-OCR2-3B"},
    "qwen":           {"model": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "glm":            {"model": "zai-org/GLM-OCR"},
    "glm-vllm":       {"model": "zai-org/GLM-OCR"},
    "hunyuan":        {"model": "tencent/HunyuanOCR"},
    "hunyuan-vllm":   {"model": "tencent/HunyuanOCR"},
    # --- new vLLM-based backends ---
    "qwen3-vllm":     {"model": "Qwen/Qwen3-VL-8B-Thinking"},
    "qwen3.5-9b":     {"model": "Qwen/Qwen3.5-9B"},
    "qwen3.5-4b":     {"model": "Qwen/Qwen3.5-4B"},
    "qwen3.5-0.8b":   {"model": "Qwen/Qwen3.5-0.8B"},
    "qwen3.5-27b":    {"model": "Qwen/Qwen3.5-27B"},
    "smolvlm":        {"model": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"},
}


def get_backend(name: str, **kwargs) -> OCRBackend:
    """Get OCR backend by name.
    
    Available backends:
        nanonets         - Nanonets OCR2-3B (HF, supports quantization)
        qwen             - Qwen2.5-VL-7B-Instruct (HF, supports quantization)
        glm              - GLM-OCR (HF, 0.9B)
        glm-vllm         - GLM-OCR via vLLM
        hunyuan          - HunyuanOCR (HF, pinned transformers)
        hunyuan-vllm     - HunyuanOCR via vLLM
        qwen3-vllm       - Qwen3-VL-8B-Thinking via vLLM
        qwen3.5-9b       - Qwen3.5-9B via vLLM
        qwen3.5-4b       - Qwen3.5-4B via vLLM
        qwen3.5-0.8b     - Qwen3.5-0.8B via vLLM
        qwen3.5-27b      - Qwen3.5-27B via vLLM
        smolvlm          - SmolVLM2-2.2B-Instruct via vLLM
    """
    # HF-loaded backends
    hf_backends = {
        "nanonets": lambda: NanonetsBackend(**kwargs),
        "qwen": lambda: Qwen25VLBackend(**kwargs),
        "glm": lambda: GLMOCRBackend(**kwargs),
        "hunyuan": lambda: HunyuanOCRBackend(**kwargs),
    }
    
    if name in hf_backends:
        return hf_backends[name]()
    
    # Legacy vLLM aliases
    if name == "glm-vllm":
        return GLMOCRVLLMBackend(**kwargs)
    if name == "hunyuan-vllm":
        return HunyuanVLLMBackend(**kwargs)
    
    # Generic vLLM backends (qwen3-vllm, qwen3.5-*, smolvlm)
    if name in BACKEND_DEFAULTS:
        defaults = BACKEND_DEFAULTS[name]
        model_name = kwargs.pop("model_name", None) or defaults["model"]
        base_url = kwargs.pop("base_url", "http://localhost:8000/v1")
        max_tokens = kwargs.pop("max_tokens", 4096)
        return VLLMBackend(
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
        )
    
    raise ValueError(f"Unknown backend: {name}. Available: {list(BACKEND_DEFAULTS.keys())}")