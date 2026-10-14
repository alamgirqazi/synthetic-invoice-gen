## Just use splits.json file

## Here is how to generate the split 

### Download the released labels from the dataset

```

curl -L -o ground_truth_released.json https://huggingface.co/datasets/alamgirqazi/invoice-ocr-synthetic/resolve/51cff59/ground_truth_released.json

```

### Create the split

python make_splits.py ground_truth_released.json -o splits-v2.json --seed 42 --ratios 0.7 0.1 0.2