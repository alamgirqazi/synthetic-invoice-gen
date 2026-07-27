# Step 1: Generate data locally
python generate_synthetic_data.py --count 200 --output synthetic_records.json

python generate_synthetic_data.py --count 800 --output synthetic_records_800.json

# Step 2: Render to images
python render_receipts.py -i synthetic_records.json -o ./rendered --format png --noise --noise-level medium



apt install fonts-roboto fonts-open-sans fonts-lato fonts-liberation fonts-ubuntu fonts-noto-core

uv pip install openai jinja2 weasyprint numpy PyMuPDF augraphy




python render_receipts.py -i synthetic_records_800.json -o ./syn_output_800 --format pdf --multi-degrade


# 4. Run clean render (all 10 templates)
python render_receipts.py -i synthetic_records.json -o ./rendered_clean --format png

# 5. Run multi-degradation (same annotations, 6x the images)
python render_receipts.py -i synthetic_records.json -o ./rendered_multi --format png --multi-degrade

# 6. Or pick specific presets
python render_receipts.py -i synthetic_records.json -o ./rendered_multi --format png \
    --multi-degrade --presets clean light medium heavy phone_photo thermal_aged

    