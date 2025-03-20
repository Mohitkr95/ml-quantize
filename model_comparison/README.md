# Model Comparison Tools

This directory contains tools to help you compare the performance of original models with their quantized versions using the `ml-quantize` package.

## Available Scripts

1. **compare_models.py** - For classification models with accuracy metrics
2. **compare_large_models.py** - For large language models with text generation quality metrics

## Installation

Install the required dependencies:

```bash
pip install -r requirements_comparison.txt
```

## Script Descriptions

### 1. Compare Models (Classification)

The `compare_models.py` script:
- Downloads a classification model (DistilBERT sentiment analysis by default)
- Quantizes it using ml-quantize
- Measures and compares:
  - Model size on disk
  - Inference speed
  - Prediction accuracy
  - Memory usage (on GPU)

**Run:**
```bash
python compare_models.py
```

**Output Directory Structure:**
```
model_comparison/
├── original/       # Original model files
└── quantized/      # Quantized model files
```

### 2. Compare Large Models (Text Generation)

The `compare_large_models.py` script:
- Downloads a text generation model (GPT-2 by default)
- Quantizes it using ml-quantize
- Generates text with both models
- Measures and compares:
  - Model size on disk
  - Text generation speed
  - Generation quality (BLEU and ROUGE metrics)
  - Memory usage (on GPU)

**Run:**
```bash
python compare_large_models.py
```

**Output Directory Structure:**
```
large_model_comparison/
├── original/       # Original model files
└── quantized/      # Quantized model files
```

## Customization

Both scripts can be customized by editing the configuration variables at the top of each file:

```python
# Configuration
MODEL_ID = "model-id-from-huggingface"  # Change to any Hugging Face model
OUTPUT_DIR = "output_directory"
QUANTIZATION_BITS = 8  # 4 or 8
```

## Notes

- The comparison scripts save both original and quantized models to disk so you can examine them and use them later.
- For larger models, make sure your system has sufficient memory and storage space.
- For very large models (7B+ parameters), consider using 4-bit quantization.
- GPU acceleration is recommended when comparing large models.
- For 4-bit quantization, make sure bitsandbytes library is installed. 