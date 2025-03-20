#!/usr/bin/env python3
"""
Example script demonstrating ml-quantize usage.
"""

from ml_quantize import quantize, Quantizer, RECOMMENDED_CONFIGS, QuantizationConfig

def simple_example():
    """One-line quantization example."""
    print("=== Simple Example ===")
    
    # Quantize with a single line
    model, tokenizer = quantize(
        "facebook/opt-125m",  # Small model for quick demo
        bits=8,
        output_dir="./quantized-opt-125m"
    )
    
    # Generate text with the quantized model
    inputs = tokenizer("Once upon a time", return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\nModel has been saved to ./quantized-opt-125m")

def advanced_example():
    """Example using the Quantizer class with more control."""
    print("\n=== Advanced Example ===")
    
    # Create the quantizer
    quantizer = Quantizer(model_id="facebook/opt-125m")
    
    # Quantize with custom parameters
    model = quantizer.quantize(
        bits=8,
        method="dynamic",
        device="auto",  # Automatically use GPU if available
        better_transformer=True
    )
    
    # Generate text
    inputs = quantizer.tokenizer("Tell me a joke about", return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=100,
        temperature=0.7,
        do_sample=True
    )
    
    print(quantizer.tokenizer.decode(outputs[0], skip_special_tokens=True))

def preset_examples():
    """Examples using configuration presets."""
    print("\n=== Configuration Preset Examples ===")
    
    # Use the "memory" preset for maximum memory savings (4-bit)
    print("Using 'memory' preset (4-bit quantization):")
    model, tokenizer = quantize(
        "facebook/opt-125m",
        config="memory"  # Uses 4-bit quantization with double quantization
    )
    
    # Use the "speed" preset for fastest inference
    print("\nUsing 'speed' preset (8-bit dynamic quantization):")
    model, tokenizer = quantize(
        "facebook/opt-125m",
        config="speed"  # Uses 8-bit dynamic quantization
    )
    
    # Custom configuration
    print("\nUsing custom configuration:")
    custom_config = QuantizationConfig(
        bits=8,
        method="static",
        better_transformer=True,
        device="cpu"
    )
    
    model, tokenizer = quantize(
        "facebook/opt-125m",
        config=custom_config
    )
    
    # Generate with the quantized model
    inputs = tokenizer("The future of AI is", return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    print("ML-Quantize Examples")
    print("--------------------")
    
    try:
        simple_example()
        advanced_example()
        preset_examples()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease make sure you have installed all required dependencies:")
        print("pip install torch transformers optimum accelerate")
    except Exception as e:
        print(f"An error occurred: {e}") 