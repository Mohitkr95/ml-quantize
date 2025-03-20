# Model Comparison Script

This script demonstrates how to compare the performance of an original model against its quantized version using the `ml-quantize` package. It provides metrics on model size, inference speed, memory usage, and accuracy.

## What the Script Does

1. Creates a dedicated `model_comparison` directory with subdirectories
2. Downloads an original model from Hugging Face and saves it to disk
3. Quantizes the model using `ml-quantize` and saves it
4. Loads both models from disk
5. Runs inference using both models on a test dataset
6. Measures and compares:
   - Model size on disk
   - Inference time
   - Prediction accuracy
   - Memory usage (when running on GPU)

## Requirements

```bash
pip install torch transformers datasets ml-quantize
```

For 4-bit quantization:
```bash
pip install bitsandbytes
```

## Usage

```bash
python compare_models.py
```

## Customization

You can edit the configuration variables at the top of the script to:

- Change the model (`MODEL_ID`)
- Use a different output directory (`OUTPUT_DIR`)
- Set quantization bit depth (`QUANTIZATION_BITS` - 4 or 8)

## Example Output

```
============================================================
MODEL COMPARISON: ORIGINAL vs. QUANTIZED
============================================================

--- Downloading original model: distilbert-base-uncased-finetuned-sst-2-english ---
Original model saved to model_comparison/original

--- Quantizing model to 8-bit ---
Quantized model saved to model_comparison/quantized

--- Loading models from disk ---

--- Loading test dataset ---

--- Running inference benchmarks ---

--- RESULTS ---
Metric           | Original  | Quantized | Difference
---------------------------------------------------
Model Size (MB)  | 268.42    | 134.21    | -50.00%
Inference Time (s) | 0.0542    | 0.0482    | -11.07%
Accuracy         | 0.9100    | 0.9000    | -0.0100

--- MEMORY USAGE (CUDA) ---
Original: 524.37 MB
Quantized: 262.18 MB
Reduction: 50.00%

Done! Models and comparison results are in: /path/to/model_comparison
```

## Notes

- For classification models, accuracy is calculated as the percentage of correct predictions.
- The script uses the SST-2 sentiment analysis dataset by default, which is appropriate for the default model.
- When using other models, you may need to modify the evaluation metrics.
- Memory measurements are only displayed when running on CUDA-enabled devices. 