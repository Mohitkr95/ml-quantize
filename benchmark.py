#!/usr/bin/env python3
"""
Benchmark script to compare original and quantized model performance.
"""

import time
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ml_quantize import quantize

def print_table(headers, rows):
    """Print a formatted table."""
    col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    
    # Print header
    header_line = ' | '.join(f'{h:{w}}' for h, w in zip(headers, col_widths))
    print(header_line)
    print('-' * len(header_line))
    
    # Print rows
    for row in rows:
        print(' | '.join(f'{c:{w}}' for c, w in zip(row, col_widths)))

def measure_memory(model):
    """Measure model memory usage in MB."""
    if next(model.parameters()).is_cuda:
        # GPU memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Run a forward pass to account for any lazy initialization
        dummy_input = torch.ones(1, 10, dtype=torch.long).to(next(model.parameters()).device)
        model(dummy_input)
        torch.cuda.synchronize()
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # CPU memory (approximate using model size)
        memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    return memory_mb

def measure_inference_time(model, tokenizer, input_text, num_runs=5):
    """Measure inference time for the model."""
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    
    # Warmup
    model.generate(**inputs, max_length=50)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.generate(**inputs, max_length=50)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times)

def benchmark(model_id, bits=8, method="dynamic", device="auto"):
    """Run benchmarks comparing original and quantized models."""
    print(f"Benchmarking model: {model_id}")
    print(f"Quantization: {bits}-bit {method}")
    print("-" * 50)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    # Load original model
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(model_id)
    original_tokenizer = AutoTokenizer.from_pretrained(model_id)
    original_model = original_model.to(device)
    
    # Load quantized model
    print("Quantizing model...")
    quantized_model, quantized_tokenizer = quantize(model_id, bits=bits, method=method, device=device)
    
    # Test input
    test_input = "Once upon a time in a land far away,"
    
    # Measure memory usage
    original_memory = measure_memory(original_model)
    quantized_memory = measure_memory(quantized_model)
    memory_reduction = (1 - quantized_memory / original_memory) * 100
    
    # Measure inference time
    print("Measuring inference times...")
    original_time = measure_inference_time(original_model, original_tokenizer, test_input)
    quantized_time = measure_inference_time(quantized_model, quantized_tokenizer, test_input)
    speedup = (original_time / quantized_time - 1) * 100 if quantized_time < original_time else -(1 - quantized_time / original_time) * 100
    
    # Show results
    headers = ["Model", "Memory (MB)", "Inference (s)"]
    rows = [
        ["Original", f"{original_memory:.2f}", f"{original_time:.4f}"],
        ["Quantized", f"{quantized_memory:.2f}", f"{quantized_time:.4f}"],
        ["Difference", f"{memory_reduction:.2f}%", f"{speedup:.2f}%"]
    ]
    
    print("\nResults:")
    print_table(headers, rows)
    
    # Generate sample output for quality check
    print("\nGeneration Sample (Original):")
    original_output = original_model.generate(
        original_tokenizer(test_input, return_tensors="pt").to(device)["input_ids"], 
        max_length=100
    )
    print(original_tokenizer.decode(original_output[0], skip_special_tokens=True))
    
    print("\nGeneration Sample (Quantized):")
    quantized_output = quantized_model.generate(
        quantized_tokenizer(test_input, return_tensors="pt").to(device)["input_ids"], 
        max_length=100
    )
    print(quantized_tokenizer.decode(quantized_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark original vs. quantized models")
    parser.add_argument("--model_id", type=str, default="facebook/opt-350m", 
                        help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8],
                        help="Quantization bits (4 or 8)")
    parser.add_argument("--method", type=str, default="dynamic", choices=["dynamic", "static"],
                        help="Quantization method")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run on")
    
    args = parser.parse_args()
    benchmark(args.model_id, args.bits, args.method, args.device) 