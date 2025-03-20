#!/bin/bash
# Script to run model comparisons sequentially

# Display header
echo "=========================================================="
echo "          ML-QUANTIZE MODEL COMPARISON SUITE              "
echo "=========================================================="

# Create directories if they don't exist
mkdir -p model_comparison
mkdir -p large_model_comparison

# Check if requirements are installed
if ! pip show torch transformers datasets ml-quantize > /dev/null 2>&1; then
  echo "Installing requirements..."
  pip install -r requirements_comparison.txt
fi

# Run classification model comparison
echo -e "\n\n=========================================================="
echo "PART 1: CLASSIFICATION MODEL COMPARISON"
echo "=========================================================="
python compare_models.py

# Run text generation model comparison
echo -e "\n\n=========================================================="
echo "PART 2: TEXT GENERATION MODEL COMPARISON"
echo "=========================================================="
python compare_large_models.py

# Summarize results
echo -e "\n\n=========================================================="
echo "SUMMARY OF RESULTS"
echo "=========================================================="
echo "Classification model comparison results: $(pwd)/model_comparison"
echo "Text generation model comparison results: $(pwd)/large_model_comparison"

echo -e "\nComparison complete! View the directories above to see the results." 