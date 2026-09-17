#!/usr/bin/env python3
"""
Simple runner script for the receipt OCR pipeline - Extended Schema
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_pipeline import ReceiptPipeline, OlmOCRBackend


# Backend to model mapping
BACKEND_MODELS = {
    "olmocr": "allenai/olmOCR-7B-0225-preview",
    "olmocr2": "allenai/olmOCR-2-7B-1025",
    "nanonets": "nanonets/Nanonets-OCR2-3B",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "glm": "zai-org/GLM-OCR",
    "glm-vllm": "zai-org/GLM-OCR",
    "hunyuan": "tencent/HunyuanOCR",
    "hunyuan-vllm": "tencent/HunyuanOCR",
    "qwen3-vllm": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

# Backends that go through vLLM (no HF loading)
VLLM_BACKENDS = {
    "glm-vllm", "hunyuan-vllm", "qwen3-vllm",
    "qwen3.5-9b", "qwen3.5-4b", "qwen3.5-0.8b", "qwen3.5-27b",
    "smolvlm",
}


def get_default_output_filename(backend: str, quantization: str = None) -> str:
    """Generate output filename with backend name, optional quantization tag, and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_tag = ""
    if quantization:
        # Map CLI names to descriptive tags for the filename
        quant_map = {"4bit": "fp4", "8bit": "fp8"}
        quant_tag = f"_{quant_map.get(quantization, quantization)}"
    return f"receipts_output_{backend}{quant_tag}_{timestamp}.json"


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in human-readable format (hours/mins/secs)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs:.2f}s")
    
    return " ".join(parts)


def main():
    import argparse

    all_backends = sorted(BACKEND_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="Receipt OCR Pipeline - Extract structured data from invoices/receipts (Extended Schema)",
        epilog=f"Available backends: {', '.join(all_backends)}",
    )
    parser.add_argument("input", help="Input PDF/image file or folder containing receipts")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path (default: auto-generated with timestamp)")
    parser.add_argument(
        "-b", "--backend",
        default="olmocr",
        choices=all_backends,
        help="OCR backend to use (default: olmocr)"
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens")
    parser.add_argument("--max-image-size", type=int, default=1024, help="Max image dimension")
    parser.add_argument("--pdf-dpi", type=int, default=150, help="DPI for PDF")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages")
    parser.add_argument("--quantization", default=None, choices=["4bit", "8bit"], help="Quantization (HF backends only)")

    args = parser.parse_args()

    # Set default output filename with backend name, quantization tag, AND run ID
    if args.output is None:
        args.output = get_default_output_filename(args.backend, args.quantization)

    # Set default max_tokens per backend if not specified
    if args.max_tokens is None:
        default_tokens = {
            "olmocr": 4096, "olmocr2": 4096,
            "nanonets": 4096, "qwen": 4096,
            "glm": 2048, "glm-vllm": 2048,
            "hunyuan": 2048, "hunyuan-vllm": 2048,
            "qwen3-vllm": 4096,
            "qwen3.5-9b": 4096, "qwen3.5-4b": 4096,
            "qwen3.5-0.8b": 4096, "qwen3.5-27b": 4096,
            "smolvlm": 4096,
        }
        args.max_tokens = default_tokens.get(args.backend, 4096)

    # Warn if quantization used with vLLM backend
    if args.quantization and args.backend in VLLM_BACKENDS:
        print(f"WARNING: --quantization is ignored for vLLM backend '{args.backend}'.")
        print("         Quantization must be configured on the vLLM server side.")

    # Start timing
    start_time = time.time()

    # Get model name
    model_name = args.model or BACKEND_MODELS.get(args.backend)

    # Print header
    print(f"\n{'='*60}")
    print("Receipt OCR Pipeline - Extended Schema")
    print(f"{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"Model: {model_name}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max pages: {args.max_pages or 'all'}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    print(f"Output format: snake_case fields (extended schema)")
    print(f"{'='*60}\n")

    # Initialize backend
    if args.backend in ["olmocr", "olmocr2"]:
        if args.quantization:
            try:
                backend = OlmOCRBackend(model_name=model_name, quantization=args.quantization)
            except TypeError:
                backend = OlmOCRBackend(model_name=model_name)
        else:
            backend = OlmOCRBackend(model_name=model_name)

    elif args.backend == "nanonets":
        from backends import NanonetsBackend
        backend = NanonetsBackend(
            model_name=model_name, max_new_tokens=args.max_tokens,
            max_image_size=args.max_image_size, quantization=args.quantization,
        )

    elif args.backend == "qwen":
        from backends import Qwen25VLBackend
        backend = Qwen25VLBackend(
            model_name=model_name, max_new_tokens=args.max_tokens,
            max_image_size=args.max_image_size, quantization=args.quantization,
        )

    elif args.backend == "glm":
        from backends import GLMOCRBackend
        backend = GLMOCRBackend(
            model_name=model_name, max_new_tokens=args.max_tokens,
            max_image_size=args.max_image_size,
        )

    elif args.backend == "glm-vllm":
        from backends import GLMOCRVLLMBackend
        backend = GLMOCRVLLMBackend(
            base_url=args.vllm_url, model_name=model_name, max_tokens=args.max_tokens,
        )

    elif args.backend == "hunyuan":
        from backends import HunyuanOCRBackend
        backend = HunyuanOCRBackend(
            model_name=model_name, max_new_tokens=args.max_tokens,
            max_image_size=args.max_image_size,
        )

    elif args.backend == "hunyuan-vllm":
        from backends import HunyuanVLLMBackend
        backend = HunyuanVLLMBackend(
            base_url=args.vllm_url, model_name=model_name, max_tokens=args.max_tokens,
        )

    elif args.backend in VLLM_BACKENDS:
        # Generic vLLM path: qwen3-vllm, qwen3.5-*, smolvlm
        from backends import VLLMBackend
        backend = VLLMBackend(
            base_url=args.vllm_url, model_name=model_name, max_tokens=args.max_tokens,
        )

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Create pipeline with configured DPI and max pages
    pipeline = ReceiptPipeline(backend, pdf_dpi=args.pdf_dpi, max_pages=args.max_pages)

    # Process input
    input_path = Path(args.input)

    if input_path.is_dir():
        # Find all valid image/PDF files in the directory
        valid_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
        files_to_process = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        total_files = len(files_to_process)
        
        if total_files == 0:
            print(f"No valid documents found in {input_path}")
            sys.exit(0)

        print(f"Found {total_files} documents. Starting batch processing...\n")
        
        results = []
        for i, file_path in enumerate(files_to_process, 1):
            file_start_time = time.time()
            print(f"[{i}/{total_files}] Processing '{file_path.name}' using model: {model_name}...")
            
            try:
                result = pipeline.process_file(str(file_path))
                
                if isinstance(result, dict):
                    result['_source_file'] = file_path.name
                
                results.append(result)
                
                # Incremental Save
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
                file_elapsed = time.time() - file_start_time
                print(f"        └─ Done in {file_elapsed:.1f}s")
                
            except Exception as e:
                print(f"        └─ ERROR processing {file_path.name}: {e}")

        # Final Summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Successfully processed {len(results)}/{total_files} documents")
        print(f"Results saved to: {args.output}")
        print(f"Total time: {format_elapsed_time(elapsed_time)}")
        print(f"Average time per document: {elapsed_time/max(1, len(results)):.2f}s")
        print(f"{'='*60}")

    else:
        print(f"Processing single file '{input_path.name}' using model: {model_name}...")
        result = pipeline.process_file(str(input_path))
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([result], f, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Result Preview:")
        preview = json.dumps(result, indent=2, ensure_ascii=False)
        print(preview[:500] + ("\n... [truncated]" if len(preview) > 500 else ""))
        print(f"\nSaved full result to: {args.output}")
        print(f"Total time: {format_elapsed_time(elapsed_time)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()