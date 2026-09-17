# synthetic-invoice-gen

Generation pipeline for **InvoiceOCR-Synth**, an annotation-noise-free synthetic dataset of receipt
and invoice images for document information extraction.

- **Dataset:** [huggingface.co/datasets/alamgirqazi/invoice-ocr-synthetic](https://huggingface.co/datasets/alamgirqazi/invoice-ocr-synthetic) · DOI [10.57967/hf/9733](https://doi.org/10.57967/hf/9733) · CC BY 4.0
- **Code (this repository):** MIT Licence

## What this does

Conventional document datasets annotate images that already exist. When an annotator misreads a
faded digit, the ground truth itself violates arithmetic identities such as
`amount = quantity × unit_price`, and a model that extracts the correct value is penalised against a
noisy label.

This pipeline inverts that order — *generate, then render*. A structured JSON record is sampled
first, and the document image is rendered **from** that record. Because the record is both the data
source and the label, the ground truth is correct by construction. Image degradation is applied
afterwards, as a controlled and independent variable.

## Pipeline

| Stage | File | What it does |
| --- | --- | --- |
| 1. Record generation | `generate_synthetic_data.py` | Samples per-record parameters, builds a prompt, calls a local LLM, parses and validates the JSON, and applies deterministic math correction. |
| 2. Rendering | `render_receipts.py`, `templates/` | Renders each record through a Jinja2 HTML/CSS template, converts to PDF with WeasyPrint, and rasterises to PNG at 150 DPI with PyMuPDF. |
| 3. Degradation | `augmentation.py` | Produces the `clean`, `faded` and `bad_scan` variants of each render. |
| 4. Audit | `audit.py` | Verifies every arithmetic identity across the finished corpus and reports the records that fail. |

Supporting scripts: `split_records.py` (shard a record file), `upload_to_hf.py` (publish to Hugging
Face), `evaluate.py` (quick sanity checks).

### Also in this repository

| Directory | Contents |
| --- | --- |
| `eval/` | Schema-driven evaluation framework used to produce the baseline results reported in the paper. Scores a prediction file against the ground truth field by field, with type-aware comparators, Hungarian matching for line items, and Entity F1 / ANLS. CLI and FastAPI entry points; see `eval/README.md`. |
| `ocr/` | Model-agnostic extraction harness. Runs a vision–language model over the images with a schema-driven structured-output prompt, with parse recovery for models that emit reasoning text instead of JSON. See `ocr/readme.md`. |

Neither directory is needed to generate the dataset — they are released so that the reported
baseline can be reproduced.

To score a prediction file against the released ground truth:

```bash
cd eval
python eval.py --gt ground_truth_released.json --pred my_model.json
```

## Installation

```bash
# Fonts used by the renderer (Debian/Ubuntu)
apt install fonts-roboto fonts-open-sans fonts-lato fonts-liberation fonts-ubuntu fonts-noto-core

# Python dependencies
uv pip install openai jinja2 weasyprint numpy PyMuPDF augraphy opencv-python pillow
```

Stage 1 needs an OpenAI-compatible endpoint. The released corpus used `openai/gpt-oss-20b` served
locally through vLLM; Ollama, the OpenAI API and Gemini are also supported via `--backend`.

## Prompt design

This is the part that determines the content distribution, so it is documented in full here.

Each record is generated from a **single user message**. There is no system message and there are no
few-shot examples. The message is assembled by `build_prompt()` in `generate_synthetic_data.py` from
the parameters sampled for that record, and has seven blocks:

1. An instruction naming the document type and region.
2. A rules block — English only, all content fictional, plausible for the region, a complexity band
   expressed as a line-item count, and dates constrained to 2022-01-01 … 2024-12-31 in ISO format.
3. A tax-regime hint selected by region, giving that jurisdiction's legal rates plus its
   registration-number and postcode formats.
4. Four format constraints that exist to suppress recurring type errors: `tax_rate` must be a string
   like `"23%"`; `supplier_company_registrations` must be a string, not an array; amounts must be
   numbers; text fields must be strings.
5. The explicit list of fields to populate for this record, computed from its sparsity profile, with
   an instruction to set everything else to `null`.
6. Sub-schemas for the `line_items` and `taxes` arrays, included only when requested.
7. A math block stating the four identities, a two-decimal rounding rule, the closed
   `document_category` vocabulary, a free-text `subcategory`, and an instruction to return JSON alone.

### Sampling parameters

| Parameter | Values |
| --- | --- |
| Region | IE 0.45, GB 0.25, US 0.20, AU 0.05, CA 0.05 |
| Document type | Uniform over 15 types (`DOCUMENT_TYPES`) |
| Complexity | low 0.35 (1–2 items), medium 0.45 (3–5), high 0.20 (6–15) |
| Field sparsity | Four profiles (`receipt`, `invoice`, `event`, `freelancer`); fields are *always* requested, requested at p=0.70, requested at p=0.25, or never |

These weights are design decisions, not measurements of a real document population. Document type is
sampled uniformly, which over-represents rare types on purpose: for an evaluation corpus, even
coverage is more useful than realistic frequency.

To see a fully instantiated prompt without calling a model:

```bash
python generate_synthetic_data.py --dry-run --count 3
```

## Validation and math correction

Two distinct steps, which make different guarantees.

**`validate_record()` is a gate.** It rejects a record — triggering regeneration, up to
`--retries` times — only when `supplier_name` or `document_date` is missing. Arithmetic
discrepancies are detected and logged here but do **not** cause rejection, because the next step is
expected to repair them.

**`fix_math()` is a deterministic rewrite**, working strictly upward from quantities and unit
prices. It **never alters a quantity, a unit price, or a tax rate** — those are taken as generated.
It recomputes each line `amount` as `quantity × unit_price` when the generated value differs by more
than 0.02, recomputes each `tax_amount` unconditionally, and then recomputes `document_total_net`,
`document_total_tax` and `document_total_amount`. The `taxes` array is corrected **only when it
contains exactly one entry**, because reassigning line items to tax bands is not a deterministic
operation.

Pass `--no-fix-math` to disable the rewrite and inspect the raw model output.

## Consistency audit

`audit.py` checks every identity the dataset claims, across the whole corpus:

```bash
python audit.py ground-truth.json --tol 0.02
```

It reports distribution counts, uniqueness statistics, field sparsity, degenerate records, and each
arithmetic failure with the offending values. For the tax summary it tests three things
independently: `sum(taxes.base) == document_total_net`, `sum(taxes.value) == document_total_tax`,
and per band `base × rate == value`.

Running this over the 1,000 generated records found 62 failures — 57 tax summaries that did not
reconcile against their own line items, and 5 degenerate records with no line items. Because six
templates print the tax summary on the page, repairing those records after rendering would have
desynchronised the label from the image, so all 62 were **excluded rather than repaired**. The
released corpus is the remaining 938. The excluded file names and reasons ship with the dataset as
`excluded_records.txt`.

## Templates

Ten template files are in `templates/`. **Six were used for the released corpus:**

`formal_corporate` · `modern_minimal` · `pos_thermal` · `freelancer_invoice` · `event_ticket` · `utility_bill`

The other four — `delivery_note`, `retail_receipt`, `taxi_receipt`, `hotel_receipt` — are retained
for users extending the corpus but are not reachable from `DOC_TYPE_TO_LAYOUT`, which maps all
fifteen document types onto the six layouts above. A taxi receipt, for example, is sampled as a
document type but rendered through `pos_thermal`, and a hotel receipt through `modern_minimal`.

Template selection is rule-based on the record's layout hint, document type, subcategory and
category, with a weighted random fallback. Each render draws a colour scheme from a ten-palette pool
and a font from weighted sans-serif, serif and monospace pools. Thermal receipts always use
monospace; non-thermal templates use a serif face in about 15% of renders.

## Degradation conditions

| Condition | Stages (in order, with intensity) |
| --- | --- |
| `clean` | **None.** The 150 DPI render is written directly to PNG. This is the reference condition. |
| `faded` | `thermal_fade` (0.60) → `lighting_gradient` (0.30) → `camera_noise` (0.20) → `jpeg_compress` (0.25) |
| `bad_scan` | `scanner_bed` (0.50) → `lighting_gradient` (0.45) → `scanner_artifacts` (0.65) → `camera_noise` (0.25) → `jpeg_compress` (0.40) |

`thermal_fade` applies a warm yellowing tint, reduces contrast by 21% and brightness by 5%, and adds
a patchy edge-biased vignette, print-head streaks and friction scratches. `scanner_bed` rotates the
page by up to ±0.75° and pads the canvas 2–8% to expose the dark scanner-lid border.
`scanner_artifacts` runs the Augraphy pipeline — InkBleed, LowInkRandomLines, ColorPaper,
BrightnessTexturize, NoiseTexturize, DirtyRollers, Folding, Brightness, SubtleNoise, Jpeg — and
reduces its own intensity by 30% when the smaller image dimension is under 600 px, which keeps
narrow POS receipts legible.

**Augraphy is used in the `bad_scan` pipeline only.** The `faded` pipeline uses PIL and OpenCV
operations. Exact parameters are in the `PRESETS` dictionary in `augmentation.py`.

Preview all three conditions for one image:

```bash
python augmentation.py --input sample.png --output-dir ./out --multi
```

## Reproducing the released corpus

The corpus was generated in two runs, of 200 and 800 records.

> **Note the temperature.** The script default is `0.9`; this release used `0.7`, so it must be
> passed explicitly.

```bash
# Stage 1 — structured records
python generate_synthetic_data.py --count 200 --output synthetic_records.json \
    --backend vllm --model openai/gpt-oss-20b --temperature 0.7

python generate_synthetic_data.py --count 800 --output synthetic_records_800.json \
    --backend vllm --model openai/gpt-oss-20b --temperature 0.7

# Stages 2 and 3 — render, then produce all three conditions
python render_receipts.py -i synthetic_records.json     -o ./out_200 --format png --multi-degrade
python render_receipts.py -i synthetic_records_800.json -o ./out_800 --format png --multi-degrade

# Stage 4 — audit; writes the list of records to exclude
python audit.py ground-truth.json --tol 0.02
```

`--multi-degrade` writes `images/clean/`, `images/faded/` and `images/bad_scan/`. **The three
variants of a document share a filename** and are distinguished only by directory, so keep the
directory structure intact.

Regenerating from scratch will not reproduce the released corpus byte-for-byte: the LLM is sampled
at temperature 0.7, and font, palette and degradation choices are randomised. Pass `--seed` to
`generate_synthetic_data.py` to fix the parameter sampling. The released corpus itself is the
canonical artefact and is on Hugging Face.

## Output structure

```
out_800/
├── images/
│   ├── clean/     synth_0479_0b3e610f.png
│   ├── faded/     synth_0479_0b3e610f.png     ← same filename
│   └── bad_scan/  synth_0479_0b3e610f.png     ← same filename
└── ground-truth.json
```

Each label record carries a `file_name` and a `degradation_variants` object mapping each condition to
its image path. The `{index}` in `synth_{index}_{hash}.png` is a per-run counter and is **not
unique** across runs — the 8-character MD5 digest of the source record is what disambiguates, so the
full `synth_{index}_{hash}` string is the document identifier.

## Licence

Code in this repository: **MIT** (see `LICENSE`).
The generated dataset: **CC BY 4.0**, distributed via Hugging Face.

## Citation

```bibtex
@misc{qazi2026invoiceocrsynth,
  author    = {Qazi, Alamgir Munir and Nasir, Jamal Abdul and
               Chambeth, Pratheesh and Qureshi, Waqar Shahid},
  title     = {InvoiceOCR-Synth: An annotation-noise-free synthetic dataset of
               receipt and invoice images for document information extraction},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/alamgirqazi/invoice-ocr-synthetic},
  doi       = {10.57967/hf/9733}
}
```