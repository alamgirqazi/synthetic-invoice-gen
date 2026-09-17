## Run on complete dataset 

python run.py ../dataset/test_receipts --backend olmocr2

## Run on test dataset 

python run.py ../dataset/capisso_receipts --backend olmocr

python run.py ../dataset/capisso_receipts --backend olmocr2

python run.py ../dataset/test_receipts --backend glm

python run.py ../dataset/test_receipts --backend qwen

python run.py ../dataset/test_receipts --backend hunyuan-vllm

python run.py ../dataset/test_receipts --backend qwen3.5-9b

python run.py ../dataset/test_receipts --backend qwen3.5-0.8b

python run.py ../dataset/test_receipts --backend smolvlm

python run.py ../dataset/test_receipts --backend qwen3-vllm


## Quantization

[support for qwen, olmocr, olmocr2 only]

python run.py ../dataset/test_receipts --backend olmocr2 --quantization 8bit
python run.py ../dataset/test_receipts --backend olmocr2 --quantization 4bit



----

QLoRA fine-tuning on Qwen2.5-VL-7B with your 20 annotated docs (since olmOCR is already a Qwen2.5-VL fine-tune, this gives you a direct comparison).

---

-------------------------

Running on 200 receipts

python run.py ../dataset/200_receipts/clean --backend olmocr2

python run.py ../dataset/200_receipts/clean --backend glm


VLLMs


pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly


vllm serve Qwen/Qwen3.5-9B --max-model-len 8192 --gpu-memory-utilization 0.85
vllm serve Qwen/Qwen3.5-0.8B --max-model-len 8192
vllm serve HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-model-len 8192

vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85

vllm serve tencent/HunyuanOCR --no-enable-prefix-caching --mm-processor-cache-gb 0
