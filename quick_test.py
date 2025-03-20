#!/usr/bin/env python3
"""
Quick test of ml-quantize with a tiny model for fast results.
Downloads, quantizes, and compares a small model in one quick test.
"""

import os
import time
import torch
from transformers import AutoModel, AutoTokenizer
from ml_quantize import quantize

# Use a tiny test model
MODEL_ID = "hf-internal-testing/tiny-random-bert"
OUTPUT_DIR = "quick_test_results"
QUANTIZED_DIR = f"{OUTPUT_DIR}/quantized"

print("=" * 50)
print("ML-QUANTIZE QUICK TEST")
print("=" * 50)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Download and load the original model
print(f"\nDownloading tiny test model: {MODEL_ID}")
start_time = time.time()
original_model = AutoModel.from_pretrained(MODEL_ID)
original_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
load_time = time.time() - start_time
print(f"Original model loaded in {load_time:.2f} seconds")

# Step 2: Quantize the model
print("\nQuantizing model to 8-bit...")
start_time = time.time()
quantized_model, quantized_tokenizer = quantize(
    MODEL_ID,
    bits=8,
    method="dynamic",
    output_dir=QUANTIZED_DIR
)
quantize_time = time.time() - start_time
print(f"Model quantized in {quantize_time:.2f} seconds")

# Step 3: Compare model sizes
original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 * 1024)
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
size_reduction = ((original_size - quantized_size) / original_size) * 100

# Step 4: Compare inference speed
test_input = "This is a test sentence for inference speed comparison."
encoded_input = original_tokenizer(test_input, return_tensors="pt")

# Original model inference
start_time = time.time()
with torch.no_grad():
    original_output = original_model(**encoded_input)
original_inference = time.time() - start_time

# Quantized model inference
encoded_input = quantized_tokenizer(test_input, return_tensors="pt")
start_time = time.time()
with torch.no_grad():
    quantized_output = quantized_model(**encoded_input)
quantized_inference = time.time() - start_time

# Print results
print("\n" + "=" * 50)
print("QUICK TEST RESULTS")
print("=" * 50)
print(f"Model: {MODEL_ID}")
print(f"Quantization: 8-bit dynamic")
print("\nMEMORY:")
print(f"Original model size:  {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Size reduction:       {size_reduction:.2f}%")
print("\nSPEED:")
print(f"Original inference:   {original_inference:.4f} seconds")
print(f"Quantized inference:  {quantized_inference:.4f} seconds")
print(f"Speed improvement:    {((original_inference - quantized_inference) / original_inference) * 100:.2f}%")

print(f"\nQuantized model saved to: {os.path.abspath(QUANTIZED_DIR)}")
print("\nQUICK TEST COMPLETED SUCCESSFULLY!") 