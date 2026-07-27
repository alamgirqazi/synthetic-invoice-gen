SYNTHETIC RECEIPT GENERATION + OCR TRAINING PIPELINE
=====================================================


  ┌─────────────────────────────────────────────────────────────────────┐
  │                    STEP 1: GENERATE JSON DATA                      │
  │                    generate_synthetic_data.py                      │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
      ▼                          ▼                          ▼
  ┌─────────┐            ┌──────────────┐          ┌──────────────┐
  │ Ollama  │            │    vLLM      │          │ OpenAI /     │
  │gpt-oss  │            │  gpt-oss-20b │          │ Gemini API   │
  │  :20b   │            │              │          │              │
  └────┬────┘            └──────┬───────┘          └──────┬───────┘
       │                        │                         │
       └────────────────────────┼─────────────────────────┘
                                │
                  All use OpenAI-compatible API
                   (single unified LLM client)
                                │
                                ▼
                   ┌────────────────────────┐
                   │   Parameter Generator  │
                   │                        │
                   │  • Pick region (IE/GB/ │
                   │    US/AU/CA weighted)  │
                   │  • Pick doc type       │
                   │    (15 types)          │
                   │  • Pick complexity     │
                   │    (low/med/high)      │
                   │  • Pick sparsity       │
                   │    profile             │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Sparsity Engine      │
                   │                        │
                   │  receipt  → ~10 fields │
                   │  invoice  → ~17 fields │
                   │  event    → ~9 fields  │
                   │  freelance→ ~15 fields │
                   │                        │
                   │  always:  100% filled  │
                   │  common:  70% chance   │
                   │  rare:    25% chance   │
                   │  skip:    never filled │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Compact Prompt       │
                   │                        │
                   │  • Region + tax rules  │
                   │  • Explicit field list │
                   │  • Strict format rules │
                   │    (tax_rate as "23%") │
                   │  • Math constraints    │
                   │  • English only        │
                   └───────────┬────────────┘
                               │
                       LLM generates JSON
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Post-Processing      │
                   │                        │
                   │  1. extract_json()     │
                   │     (handles thinking  │
                   │      tokens, fences)   │
                   │                        │
                   │  2. fix_math()         │
                   │     (recalculate from  │
                   │      line items)       │
                   │                        │
                   │  3. normalize_record() │
                   │     • 0.23 → "23%"    │
                   │     • [arr] → "string" │
                   │     • locale = "en"    │
                   │                        │
                   │  4. inject_template_   │
                   │     hints()            │
                   │     (deterministic,    │
                   │      not LLM-chosen)   │
                   │                        │
                   │  5. validate_record()  │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │  synthetic_records.json │
                   │                        │
                   │  200 JSON records with │
                   │  _template_hints,      │
                   │  _generation_params    │
                   └───────────┬────────────┘
                               │
                               │
  ┌────────────────────────────┼────────────────────────────────────────┐
  │                    STEP 2: RENDER TO IMAGES                        │
  │                    render_receipts.py                               │
  └────────────────────────────┼───────────────────────────────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Template Selector    │
                   │                        │
                   │  _template_hints       │
                   │    .layout_style →     │
                   │                        │
                   │  ┌──────────────────┐  │
                   │  │formal_corporate  │  │ ← tax_invoice, rent
                   │  │freelancer_invoice│  │ ← consulting, freelance
                   │  │pos_thermal       │  │ ← receipt, restaurant
                   │  │modern_minimal    │  │ ← subscription, hotel
                   │  │event_ticket      │  │ ← event, order
                   │  │utility_bill      │  │ ← utility, credit_note
                   │  └──────────────────┘  │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Font Randomizer      │
                   │                        │
                   │  sans pool (9 fonts):  │
                   │    Roboto, Open Sans,  │
                   │    Lato, Noto, etc.    │
                   │                        │
                   │  serif pool (5 fonts): │
                   │    Liberation, DejaVu  │
                   │    (15% chance)        │
                   │                        │
                   │  mono pool (5 fonts):  │
                   │    Courier, Ubuntu Mono│
                   │    (POS thermal only)  │
                   │                        │
                   │  Injects CSS override  │
                   │  before </head>        │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Jinja2 + WeasyPrint  │
                   │                        │
                   │  JSON → HTML template  │
                   │  HTML → PDF            │
                   │  PDF  → PNG (PyMuPDF)  │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   Augraphy Degradation │
                   │   (or PIL fallback)    │
                   │                        │
                   │  light:  paper tint,   │
                   │          JPEG only     │
                   │                        │
                   │  light+: + ink bleed,  │  ← RECOMMENDED
                   │          ±1° rotation, │
                   │          subtle noise  │
                   │                        │
                   │  medium: + dirty       │
                   │          rollers,      │
                   │          noise texture │
                   │                        │
                   │  heavy:  + photocopy,  │
                   │          folding,      │
                   │          bleed-through │
                   │                        │
                   │  Auto-scales for small │
                   │  images (POS receipts  │
                   │  get 1 level gentler)  │
                   └───────────┬────────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌──────────────────┐              ┌──────────────────┐
   │  rendered/images/ │              │  rendered/        │
   │                   │              │  labels.json      │
   │  synth_0000.png   │              │                   │
   │  synth_0001.png   │              │  Ground-truth     │
   │  synth_0002.png   │              │  labels matching  │
   │  ...              │              │  OCR pipeline     │
   │  synth_0199.png   │              │  OUTPUT_SCHEMA    │
   │                   │              │                   │
   │  200 images       │              │  FREE — no manual │
   │                   │              │  annotation!      │
   └────────┬─────────┘              └────────┬──────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │
                           │
  ┌────────────────────────┼────────────────────────────────────────────┐
  │                 STEP 3: COMBINE WITH REAL DATA                     │
  └────────────────────────┼───────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │ 200 REAL images  │      │ 200 SYNTHETIC    │
   │                  │      │ images           │
   │ Manually labeled │      │                  │
   │ (human annotated │      │ Auto-labeled     │
   │  ground truth)   │      │ (from JSON)      │
   └────────┬─────────┘      └────────┬─────────┘
            │                         │
            └────────────┬────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  400 total documents │
              │                     │
              │  Combined dataset:  │
              │  all_labeled_data   │
              │  .json              │
              └──────────┬──────────┘
                         │
                         │
  ┌──────────────────────┼──────────────────────────────────────────────┐
  │              STEP 4: FINE-TUNE OCR MODEL                           │
  │              finetune/finetune.py                                   │
  └──────────────────────┼─────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Progressive         │
              │  Fine-tuning         │
              │                      │
              │  20 docs → LoRA v1   │
              │  40 docs → LoRA v2   │
              │  80 docs → LoRA v3   │
              │  400 docs → LoRA v4  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Base model:         │
              │  olmOCR-2-7B         │
              │  (Qwen2.5-VL)       │
              │                      │
              │  + LoRA adapters     │
              │  (r=16, alpha=32)    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Evaluate against    │
              │  held-out test set   │
              │                      │
              │  Compare:            │
              │  • Base olmOCR       │
              │  • + 20 docs LoRA    │
              │  • + 80 docs LoRA    │
              │  • + 400 docs LoRA   │
              │  • vs Qwen2.5-VL    │
              │  • vs Nanonets      │
              │  • vs GLM-OCR       │
              └──────────────────────┘


QUICK START COMMANDS
====================

# Step 1: Generate 200 records locally
ollama pull gpt-oss:20b
python generate_synthetic_data.py --count 200 -o synthetic_records.json

# Step 2: Render to images
pip install jinja2 weasyprint PyMuPDF augraphy
python render_receipts.py -i synthetic_records.json -o ./rendered \
    --format png --noise --noise-level light+ --dpi 200

# Output:
#   rendered/images/synth_0000_xxxx.png  (200 images)
#   rendered/labels.json                 (ground truth)


FILE STRUCTURE
==============

synth_receipt_renderer/
├── generate_synthetic_data.py   # Step 1: JSON generation
├── render_receipts.py           # Step 2: JSON → image rendering
├── templates/
│   ├── formal_corporate.html    # Rubicon-style VAT invoice
│   ├── freelancer_invoice.html  # Hourly-rate contractor invoice
│   ├── pos_thermal.html         # Narrow thermal receipt
│   ├── modern_minimal.html      # Clean modern invoice
│   ├── event_ticket.html        # Eventbrite-style order summary
│   └── utility_bill.html        # Colored-banner service bill
└── test_records.json            # 5 sample records for testing