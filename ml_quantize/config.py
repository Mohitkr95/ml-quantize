"""
Advanced configuration options for ml-quantize.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Basic options
    bits: int = 8  # 4 or 8 bits
    method: str = "dynamic"  # "dynamic" or "static"
    device: str = "auto"  # "cpu", "cuda", or "auto"
    better_transformer: bool = True
    
    # Advanced options for 4-bit quantization
    compute_dtype: str = "float16"  # "float16" or "bfloat16"
    use_double_quant: bool = True
    quant_type: str = "nf4"  # "nf4" or "fp4"
    
    # Advanced options for 8-bit static quantization
    qconfig_mapping: Optional[Dict[str, Any]] = None
    
    def get_bnb_config(self):
        """Get BitsAndBytes configuration for 4-bit quantization."""
        if self.bits != 4:
            return None
            
        # Import here to avoid dependency if not used
        try:
            import torch
            from transformers import BitsAndBytesConfig
            
            compute_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype_map.get(self.compute_dtype, torch.float16),
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_quant_type=self.quant_type,
            )
        except ImportError:
            raise ImportError("4-bit quantization requires bitsandbytes and transformers libraries")
    
    def get_torch_qconfig(self):
        """Get PyTorch quantization configuration for 8-bit quantization."""
        if self.bits != 8:
            return None
            
        import torch
        
        if self.method == "dynamic":
            return torch.quantization.default_dynamic_qconfig
        else:
            return torch.quantization.get_default_qconfig('fbgemm')


# Common configuration presets
RECOMMENDED_CONFIGS = {
    "quality": QuantizationConfig(
        bits=8,
        method="static",
        better_transformer=True
    ),
    "speed": QuantizationConfig(
        bits=8,
        method="dynamic",
        better_transformer=True
    ),
    "memory": QuantizationConfig(
        bits=4,
        method="dynamic",
        better_transformer=True,
        use_double_quant=True
    ),
    "balanced": QuantizationConfig(
        bits=4,
        method="dynamic",
        better_transformer=True,
        quant_type="nf4"
    ),
} 