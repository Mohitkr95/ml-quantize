#!/usr/bin/env python3
"""
Compare original and quantized models with accuracy metrics.
This script:
1. Downloads and saves an original model
2. Quantizes the model
3. Runs both models and compares performance metrics
"""

import os
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from ml_quantize import quantize, QuantizationConfig

# Configuration
MODEL_ID = "facebook/opt-125m"  # Smaller OPT model
OUTPUT_DIR = "model_comparison"
ORIGINAL_DIR = f"{OUTPUT_DIR}/original"
QUANTIZED_DIR = f"{OUTPUT_DIR}/quantized"
QUANTIZATION_BITS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_dirs():
    """Create necessary directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(QUANTIZED_DIR, exist_ok=True)
    print(f"Created directories in {os.path.abspath(OUTPUT_DIR)}")


def download_and_save_original():
    """Download and save the original model."""
    print(f"\n--- Downloading original model: {MODEL_ID} ---")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Save to disk
    model.save_pretrained(ORIGINAL_DIR)
    tokenizer.save_pretrained(ORIGINAL_DIR)
    print(f"Original model saved to {ORIGINAL_DIR}")
    
    return model, tokenizer


def quantize_model():
    """Quantize the model and save it."""
    print(f"\n--- Quantizing model to {QUANTIZATION_BITS}-bit ---")
    
    # Create a balanced configuration
    config = QuantizationConfig(
        bits=4,
        method="gptq",
        device=DEVICE,
        better_transformer=True
    )
    
    # Quantize the model
    q_model, q_tokenizer = quantize(
        MODEL_ID,
        config=config, 
        output_dir=QUANTIZED_DIR
    )
    
    print(f"Quantized model saved to {QUANTIZED_DIR}")
    if hasattr(q_model, 'is_quantized') and q_model.is_quantized:
        print("Model was successfully quantized!")
    else:
        print("WARNING: Model quantization did not apply properly")
    return q_model, q_tokenizer


def load_models():
    """Load both models from disk."""
    print("\n--- Loading models from disk ---")
    
    # Load original model
    original_model = AutoModelForSequenceClassification.from_pretrained(ORIGINAL_DIR)
    original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_DIR)
    
    # Load quantized model
    quantized_model = AutoModelForSequenceClassification.from_pretrained(
        "TheBloke/distilbert-base-uncased-finetuned-sst-2-english-GPTQ"
    )
    quantized_tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_DIR)
    
    # Move to appropriate device
    original_model = original_model.to(DEVICE)
    quantized_model = quantized_model.to(DEVICE)
    
    return (original_model, original_tokenizer), (quantized_model, quantized_tokenizer)


def measure_model_size(model_dir):
    """Measure the size of a saved model in MB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB


def measure_inference_time(model, tokenizer, texts, batch_size=16):
    """Measure inference time for a batch of texts."""
    # Tokenize inputs
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
    
    # Warmup
    with torch.no_grad():
        model(**encodings)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**encodings)
    inference_time = time.time() - start_time
    
    return inference_time, outputs


def calculate_accuracy(outputs, labels):
    """Calculate accuracy from model outputs and true labels."""
    predictions = torch.argmax(outputs.logits, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    return accuracy


def print_table(headers, rows):
    """Print a nicely formatted table."""
    col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    
    # Print header
    header_line = ' | '.join(f'{h:{w}}' for h, w in zip(headers, col_widths))
    print(header_line)
    print('-' * len(header_line))
    
    # Print rows
    for row in rows:
        print(' | '.join(f'{c:{w}}' for c, w in zip(row, col_widths)))


def main():
    """Main function to run the comparison."""
    print("=" * 60)
    print("MODEL COMPARISON: ORIGINAL vs. QUANTIZED")
    print("=" * 60)
    
    # Setup
    setup_dirs()
    
    # Download and save original model
    original_model, original_tokenizer = download_and_save_original()
    
    # Quantize model
    quantized_model, quantized_tokenizer = quantize_model()
    
    # Reload both models from disk
    (original_model, original_tokenizer), (quantized_model, quantized_tokenizer) = load_models()
    
    # Load test dataset (SST-2 for sentiment analysis)
    print("\n--- Loading test dataset ---")
    test_dataset = load_dataset("glue", "sst2", split="validation")
    test_texts = test_dataset["sentence"][:100]  # Use 100 examples for testing
    test_labels = torch.tensor(test_dataset["label"][:100]).to(DEVICE)
    
    # Measure model sizes
    original_size = measure_model_size(ORIGINAL_DIR)
    quantized_size = measure_model_size(QUANTIZED_DIR)
    size_reduction = ((original_size - quantized_size) / original_size) * 100
    
    # Measure inference times
    print("\n--- Running inference benchmarks ---")
    orig_time, orig_outputs = measure_inference_time(original_model, original_tokenizer, test_texts)
    quant_time, quant_outputs = measure_inference_time(quantized_model, quantized_tokenizer, test_texts)
    time_improvement = ((orig_time - quant_time) / orig_time) * 100
    
    # Calculate accuracy
    original_accuracy = calculate_accuracy(orig_outputs, test_labels)
    quantized_accuracy = calculate_accuracy(quant_outputs, test_labels)
    accuracy_diff = quantized_accuracy - original_accuracy
    
    # Print results table
    print("\n--- RESULTS ---")
    headers = ["Metric", "Original", "Quantized", "Difference"]
    rows = [
        ["Model Size (MB)", f"{original_size:.2f}", f"{quantized_size:.2f}", f"-{size_reduction:.2f}%"],
        ["Inference Time (s)", f"{orig_time:.4f}", f"{quant_time:.4f}", f"{'-' if time_improvement > 0 else '+'}{abs(time_improvement):.2f}%"],
        ["Accuracy", f"{original_accuracy:.4f}", f"{quantized_accuracy:.4f}", f"{'+' if accuracy_diff > 0 else ''}{accuracy_diff:.4f}"]
    ]
    
    print_table(headers, rows)
    
    # Memory usage (if on CUDA)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory for original model
        inputs = original_tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            original_model(**inputs)
        original_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory for quantized model
        inputs = quantized_tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            quantized_model(**inputs)
        quantized_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        memory_reduction = ((original_memory - quantized_memory) / original_memory) * 100
        
        print("\n--- MEMORY USAGE (CUDA) ---")
        print(f"Original: {original_memory:.2f} MB")
        print(f"Quantized: {quantized_memory:.2f} MB")
        print(f"Reduction: {memory_reduction:.2f}%")
    
    print("\nDone! Models and comparison results are in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main() 