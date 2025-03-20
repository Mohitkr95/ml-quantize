#!/usr/bin/env python3
"""
Compare original and quantized large language models.
This script:
1. Downloads and saves an original LLM
2. Quantizes the model
3. Runs both models and compares generation quality and performance
"""

import os
import time
import torch
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
from ml_quantize import quantize, QuantizationConfig

# Configuration
MODEL_ID = "gpt2"  # Small GPT-2 model for demonstration
OUTPUT_DIR = "large_model_comparison"
ORIGINAL_DIR = f"{OUTPUT_DIR}/original"
QUANTIZED_DIR = f"{OUTPUT_DIR}/quantized"
QUANTIZATION_BITS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text generation prompts for quality evaluation
TEST_PROMPTS = [
    "The quick brown fox",
    "Once upon a time in",
    "The meaning of life is",
    "Artificial intelligence will",
    "In the future, humans will"
]


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
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
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
        bits=QUANTIZATION_BITS,
        method="dynamic",
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
    return q_model, q_tokenizer


def load_models():
    """Load both models from disk."""
    print("\n--- Loading models from disk ---")
    
    # Load original model
    original_model = AutoModelForCausalLM.from_pretrained(ORIGINAL_DIR)
    original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_DIR)
    
    # Load quantized model
    quantized_model = AutoModelForCausalLM.from_pretrained(QUANTIZED_DIR)
    quantized_tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_DIR)
    
    # Move to appropriate device
    original_model = original_model.to(DEVICE)
    quantized_model = quantized_model.to(DEVICE)
    
    # Set padding token if needed
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token
    if quantized_tokenizer.pad_token is None:
        quantized_tokenizer.pad_token = quantized_tokenizer.eos_token
    
    return (original_model, original_tokenizer), (quantized_model, quantized_tokenizer)


def measure_model_size(model_dir):
    """Measure the size of a saved model in MB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB


def generate_text(model, tokenizer, prompt, max_length=50):
    """Generate text from a prompt."""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate text
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7
        )
    generation_time = time.time() - start_time
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text, generation_time


def measure_generation_metrics(original_generations, quantized_generations):
    """Measure text generation quality metrics (BLEU and ROUGE)."""
    # Initialize ROUGE scorer
    rouge = Rouge()
    
    bleu_scores = []
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    
    for orig, quant in zip(original_generations, quantized_generations):
        # Calculate BLEU score
        orig_tokens = orig.split()
        quant_tokens = quant.split()
        smoothie = SmoothingFunction().method1
        try:
            bleu = sentence_bleu([orig_tokens], quant_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
        
        # Calculate ROUGE scores
        try:
            scores = rouge.get_scores(quant, orig)[0]
            rouge_scores['rouge-1'].append(scores['rouge-1']['f'])
            rouge_scores['rouge-2'].append(scores['rouge-2']['f'])
            rouge_scores['rouge-l'].append(scores['rouge-l']['f'])
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
    
    # Calculate averages
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge1 = np.mean(rouge_scores['rouge-1']) if rouge_scores['rouge-1'] else 0
    avg_rouge2 = np.mean(rouge_scores['rouge-2']) if rouge_scores['rouge-2'] else 0
    avg_rougeL = np.mean(rouge_scores['rouge-l']) if rouge_scores['rouge-l'] else 0
    
    return {
        'bleu': avg_bleu,
        'rouge-1': avg_rouge1,
        'rouge-2': avg_rouge2,
        'rouge-l': avg_rougeL
    }


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
    print("=" * 70)
    print("LARGE LANGUAGE MODEL COMPARISON: ORIGINAL vs. QUANTIZED")
    print("=" * 70)
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError:
        print("NLTK not installed. Install with: pip install nltk")
    
    # Setup
    setup_dirs()
    
    # Download and save original model
    original_model, original_tokenizer = download_and_save_original()
    
    # Quantize model
    quantized_model, quantized_tokenizer = quantize_model()
    
    # Reload both models from disk
    (original_model, original_tokenizer), (quantized_model, quantized_tokenizer) = load_models()
    
    # Measure model sizes
    original_size = measure_model_size(ORIGINAL_DIR)
    quantized_size = measure_model_size(QUANTIZED_DIR)
    size_reduction = ((original_size - quantized_size) / original_size) * 100
    
    # Text generation evaluation
    print("\n--- Running text generation benchmarks ---")
    original_generations = []
    quantized_generations = []
    original_times = []
    quantized_times = []
    
    print("\nGenerating text samples...")
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\nPrompt {i+1}: '{prompt}'")
        
        # Generate with original model
        orig_text, orig_time = generate_text(original_model, original_tokenizer, prompt)
        original_generations.append(orig_text)
        original_times.append(orig_time)
        print(f"Original: '{orig_text[:100]}...' ({orig_time:.4f}s)")
        
        # Generate with quantized model
        quant_text, quant_time = generate_text(quantized_model, quantized_tokenizer, prompt)
        quantized_generations.append(quant_text)
        quantized_times.append(quant_time)
        print(f"Quantized: '{quant_text[:100]}...' ({quant_time:.4f}s)")
    
    # Calculate average generation time
    avg_orig_time = np.mean(original_times)
    avg_quant_time = np.mean(quantized_times)
    time_improvement = ((avg_orig_time - avg_quant_time) / avg_orig_time) * 100
    
    # Calculate text quality metrics
    print("\n--- Calculating generation quality metrics ---")
    quality_metrics = measure_generation_metrics(original_generations, quantized_generations)
    
    # Print performance results
    print("\n--- PERFORMANCE RESULTS ---")
    headers = ["Metric", "Original", "Quantized", "Difference"]
    rows = [
        ["Model Size (MB)", f"{original_size:.2f}", f"{quantized_size:.2f}", f"-{size_reduction:.2f}%"],
        ["Generation Time (s)", f"{avg_orig_time:.4f}", f"{avg_quant_time:.4f}", 
         f"{'-' if time_improvement > 0 else '+'}{abs(time_improvement):.2f}%"]
    ]
    print_table(headers, rows)
    
    # Print quality results
    print("\n--- GENERATION QUALITY METRICS ---")
    # BLEU and ROUGE scores measure similarity between quantized and original output
    # Higher is better (closer to original model output)
    quality_headers = ["Metric", "Score", "Interpretation"]
    quality_rows = [
        ["BLEU", f"{quality_metrics['bleu']:.4f}", "Similarity to original (0-1)"],
        ["ROUGE-1", f"{quality_metrics['rouge-1']:.4f}", "Unigram overlap"],
        ["ROUGE-2", f"{quality_metrics['rouge-2']:.4f}", "Bigram overlap"],
        ["ROUGE-L", f"{quality_metrics['rouge-l']:.4f}", "Longest sequence overlap"]
    ]
    print_table(quality_headers, quality_rows)
    
    # Memory usage (if on CUDA)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory for original model
        input_ids = original_tokenizer(TEST_PROMPTS[0], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            original_model.generate(input_ids["input_ids"], max_length=50)
        original_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory for quantized model
        input_ids = quantized_tokenizer(TEST_PROMPTS[0], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            quantized_model.generate(input_ids["input_ids"], max_length=50)
        quantized_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        memory_reduction = ((original_memory - quantized_memory) / original_memory) * 100
        
        print("\n--- MEMORY USAGE (CUDA) ---")
        print(f"Original: {original_memory:.2f} MB")
        print(f"Quantized: {quantized_memory:.2f} MB")
        print(f"Reduction: {memory_reduction:.2f}%")
    
    print("\nDone! Models and comparison results are in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main() 