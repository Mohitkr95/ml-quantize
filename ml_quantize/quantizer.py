import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from optimum.bettertransformer import BetterTransformer
from typing import Union, Optional, Tuple, Any

try:
    from ml_quantize.config import QuantizationConfig, RECOMMENDED_CONFIGS
except ImportError:
    # For development/testing when config might not be available
    QuantizationConfig = None
    RECOMMENDED_CONFIGS = {}

class Quantizer:
    """Main class to handle model quantization for HuggingFace models."""
    
    def __init__(self, model_id=None, model=None, tokenizer=None):
        """
        Initialize the quantizer with either a model_id or a model instance.
        
        Args:
            model_id (str, optional): HuggingFace model ID
            model (Model, optional): Pre-loaded HuggingFace model
            tokenizer (Tokenizer, optional): Pre-loaded tokenizer
        """
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.quantized_model = None
        
        if model_id and not model:
            self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer from HuggingFace."""
        try:
            # Try causal LM first (most common)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        except ValueError:
            # Fall back to generic model loading
            self.model = AutoModel.from_pretrained(self.model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    
    def quantize(self, bits=8, method="dynamic", device="auto", better_transformer=True, config=None):
        """
        Quantize the model to the specified bit width.
        
        Args:
            bits (int): Quantization bit width (4 or 8)
            method (str): Quantization method ('dynamic' or 'static')
            device (str): Device to use ('cpu', 'cuda', or 'auto')
            better_transformer (bool): Whether to apply BetterTransformer optimization
            config (Union[QuantizationConfig, str], optional): Configuration object or preset name
            
        Returns:
            The quantized model
        """
        if not self.model:
            raise ValueError("No model loaded. Please provide a model_id or model instance.")
        
        # Handle config if provided
        if config is not None:
            if QuantizationConfig is None:
                raise ImportError("Config module not available")
                
            if isinstance(config, str):
                if config in RECOMMENDED_CONFIGS:
                    config_obj = RECOMMENDED_CONFIGS[config]
                else:
                    raise ValueError(f"Unknown config preset: {config}. Available: {list(RECOMMENDED_CONFIGS.keys())}")
            else:
                config_obj = config
                
            # Use config values unless overridden by explicit parameters
            bits = bits if bits != 8 else config_obj.bits
            method = method if method != "dynamic" else config_obj.method
            device = device if device != "auto" else config_obj.device
            better_transformer = better_transformer if not better_transformer else config_obj.better_transformer
            
            # If we're using 4-bit quantization and have a config, use its BNB settings
            if bits == 4 and hasattr(config_obj, "get_bnb_config"):
                bnb_config = config_obj.get_bnb_config()
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Apply quantization
        if bits == 8:
            if method == "dynamic":
                try:
                    # 8-bit dynamic quantization
                    self.quantized_model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except RuntimeError as e:
                    if "NoQEngine" in str(e):
                        print("Warning: Your system doesn't support PyTorch's quantization engine.")
                        print("Using original model instead. For full quantization support, try:")
                        print("1. Using a system with Intel MKL/FBGEMM support")
                        print("2. Using 4-bit quantization with bitsandbytes instead")
                        self.quantized_model = self.model  # Use original model as fallback
                    else:
                        raise
            else:
                try:
                    # 8-bit static quantization (simplified approach)
                    self.quantized_model = torch.ao.quantization.quantize_fx.prepare_fx(
                        self.model, torch.ao.quantization.get_default_qconfig_mapping('fbgemm')
                    )
                except (RuntimeError, NotImplementedError) as e:
                    print(f"Warning: Static quantization failed: {e}")
                    print("Using original model instead.")
                    self.quantized_model = self.model
        elif bits == 4:
            # Apply 4-bit quantization via bitsandbytes integration
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                
                # Use config if provided, otherwise default settings
                if config is not None and 'bnb_config' in locals():
                    quantization_config = bnb_config
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                # Reload with quantization config
                if self.model_id:
                    self.quantized_model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, 
                        quantization_config=quantization_config,
                        device_map=device
                    )
                else:
                    raise ValueError("4-bit quantization requires model_id to be set")
            except ImportError:
                raise ImportError("4-bit quantization requires bitsandbytes. Install with 'pip install bitsandbytes'")
        else:
            raise ValueError("Supported bit values are 4 and 8")
        
        # Apply BetterTransformer if requested
        if better_transformer and self.quantized_model:
            try:
                self.quantized_model = BetterTransformer.transform(self.quantized_model)
            except Exception as e:
                print(f"BetterTransformer optimization failed: {e}")
                
        return self.quantized_model
    
    def save(self, output_dir):
        """
        Save the quantized model and tokenizer.
        
        Args:
            output_dir (str): Directory to save the model to
        """
        if not self.quantized_model:
            raise ValueError("No quantized model available. Run quantize() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        self.quantized_model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model and tokenizer saved to {output_dir}")

# Convenient function to quantize a model in one go
def quantize(model_id, bits=8, method="dynamic", device="auto", better_transformer=True, output_dir=None, config=None):
    """
    Quantize a model in a single function call.
    
    Args:
        model_id (str): HuggingFace model ID
        bits (int): Quantization bit width (4 or 8)
        method (str): Quantization method ('dynamic' or 'static')
        device (str): Device to use ('cpu', 'cuda', or 'auto')
        better_transformer (bool): Whether to apply BetterTransformer optimization
        output_dir (str, optional): Directory to save the model to
        config (Union[QuantizationConfig, str], optional): Configuration object or preset name
    
    Returns:
        tuple: (quantized_model, tokenizer)
    """
    quantizer = Quantizer(model_id=model_id)
    quantized_model = quantizer.quantize(
        bits=bits, 
        method=method, 
        device=device, 
        better_transformer=better_transformer,
        config=config
    )
    
    if output_dir:
        quantizer.save(output_dir)
    
    return quantized_model, quantizer.tokenizer 