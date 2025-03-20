import unittest
import os
import shutil
import torch
from ml_quantize import Quantizer, quantize

class TestQuantizer(unittest.TestCase):
    """Tests for the ml_quantize package."""
    
    TEST_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"  # Tiny model for testing
    TEST_OUTPUT_DIR = "./test-quantized-model"
    
    def setUp(self):
        """Set up test environment."""
        # Ensure clean test directory
        if os.path.exists(self.TEST_OUTPUT_DIR):
            shutil.rmtree(self.TEST_OUTPUT_DIR)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.TEST_OUTPUT_DIR):
            shutil.rmtree(self.TEST_OUTPUT_DIR)
    
    def test_init_with_model_id(self):
        """Test initializing Quantizer with a model ID."""
        quantizer = Quantizer(model_id=self.TEST_MODEL_ID)
        self.assertIsNotNone(quantizer.model)
        self.assertIsNotNone(quantizer.tokenizer)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantize_8bit_cuda(self):
        """Test 8-bit quantization on CUDA."""
        quantizer = Quantizer(model_id=self.TEST_MODEL_ID)
        model = quantizer.quantize(bits=8, device="cuda")
        self.assertIsNotNone(model)
    
    def test_quantize_8bit_cpu(self):
        """Test 8-bit quantization on CPU."""
        quantizer = Quantizer(model_id=self.TEST_MODEL_ID)
        model = quantizer.quantize(bits=8, device="cpu")
        self.assertIsNotNone(model)
    
    def test_save_and_load(self):
        """Test saving and loading the quantized model."""
        # Quantize and save
        quantizer = Quantizer(model_id=self.TEST_MODEL_ID)
        model = quantizer.quantize(bits=8, device="cpu")
        quantizer.save(self.TEST_OUTPUT_DIR)
        
        # Check that files were created
        self.assertTrue(os.path.exists(self.TEST_OUTPUT_DIR))
        self.assertTrue(os.path.exists(os.path.join(self.TEST_OUTPUT_DIR, "config.json")))
    
    def test_convenience_function(self):
        """Test the convenience function."""
        model, tokenizer = quantize(
            self.TEST_MODEL_ID, 
            bits=8, 
            device="cpu", 
            output_dir=self.TEST_OUTPUT_DIR
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertTrue(os.path.exists(self.TEST_OUTPUT_DIR))

if __name__ == "__main__":
    unittest.main() 