# ml-quantize

A simple Python package to quantize any machine learning model from Hugging Face with just a few lines of code.

## Installation

```bash
pip install ml-quantize
```

For 4-bit quantization support, also install:
```bash
pip install bitsandbytes
```

## Usage

### Basic usage (just 2-3 lines of code)

```python
from ml_quantize import quantize

# One-line quantization (returns the model and tokenizer)
model, tokenizer = quantize("mistralai/Mistral-7B-v0.1", bits=4)

# Use the quantized model
output = model.generate(tokenizer.encode("Hello, I am", return_tensors="pt"))
print(tokenizer.decode(output[0]))
```

### Using configuration presets

```python
from ml_quantize import quantize

# Use the "memory" preset for maximum memory savings (4-bit)
model, tokenizer = quantize("facebook/opt-350m", config="memory")

# Or use the "speed" preset for fastest inference
model, tokenizer = quantize("facebook/opt-350m", config="speed")

# Or use "balanced" for a good balance of speed, memory and quality
model, tokenizer = quantize("facebook/opt-350m", config="balanced")
```

Available presets:
- `"memory"`: 4-bit quantization with double quantization for maximum memory savings
- `"speed"`: 8-bit dynamic quantization for fastest inference
- `"quality"`: 8-bit static quantization for best quality
- `"balanced"`: 4-bit NF4 quantization with balanced memory savings and quality

### Advanced usage with Quantizer class

```python
from ml_quantize import Quantizer

# Initialize the quantizer
quantizer = Quantizer(model_id="facebook/opt-350m")

# Quantize the model with custom parameters
quantized_model = quantizer.quantize(
    bits=8,              # Use 8-bit quantization (4 or 8)
    method="dynamic",    # Use dynamic quantization
    device="cuda",       # Run on GPU
    better_transformer=True  # Apply BetterTransformer optimization
)

# Save the quantized model
quantizer.save("./quantized-opt-350m")
```

### Custom configurations

```python
from ml_quantize import quantize, QuantizationConfig

# Create a custom configuration
custom_config = QuantizationConfig(
    bits=4,                  # 4-bit quantization
    method="dynamic",
    compute_dtype="bfloat16", # Use bfloat16 for compute
    quant_type="nf4",        # Use NF4 quantization
    use_double_quant=True    # Use double quantization
)

# Quantize with custom config
model, tokenizer = quantize(
    "facebook/opt-350m",
    config=custom_config,
    output_dir="./custom-quantized-model"
)
```

## Benchmarking

The package includes a benchmark script to compare performance between original and quantized models:

```bash
python benchmark.py --model_id facebook/opt-125m --bits 4
```

## Supported Features

- 8-bit quantization (dynamic and static)
- 4-bit quantization (using BitsAndBytes)
- BetterTransformer optimization
- Configuration presets for common use cases
- Custom quantization configurations
- Saving and loading quantized models
- Support for most HuggingFace model types

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Optimum 1.10+
- Accelerate 0.20+
- BitsAndBytes (optional, for 4-bit quantization) 