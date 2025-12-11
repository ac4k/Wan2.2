"""
T5 NVFP4 quantized model loading utilities
"""
import os
import glob
import logging
import torch
import torch.nn as nn
from safetensors import safe_open
from .t5 import T5Encoder, T5SelfAttention
from .t5_quant_linear_nvfp4 import T5QuantizedLinearNVFP4


def replace_t5_linear_with_quantized_nvfp4(model, weight_dict):
    """
    Replace Linear layers in T5 model with NVFP4 QuantizedLinear layers.
    
    Args:
        model: T5Encoder instance
        weight_dict: Dictionary containing quantized weights
    """
    # Replace Linear layers in blocks (attn and ffn)
    for i, block in enumerate(model.blocks):
        # Self attention: q, k, v, o
        for name in ['q', 'k', 'v', 'o']:
            linear = getattr(block.attn, name)
            weight_key = f"blocks.{i}.attn.{name}.weight"
            quant_linear = _create_t5_quantized_linear_nvfp4(linear, weight_key, weight_dict)
            if quant_linear is not None:
                setattr(block.attn, name, quant_linear)
        
        # FFN: fc1, fc2, gate.0
        for name in ['fc1', 'fc2']:
            linear = getattr(block.ffn, name)
            weight_key = f"blocks.{i}.ffn.{name}.weight"
            quant_linear = _create_t5_quantized_linear_nvfp4(linear, weight_key, weight_dict)
            if quant_linear is not None:
                setattr(block.ffn, name, quant_linear)
        
        # Gate layer (first layer in Sequential)
        if hasattr(block.ffn, 'gate') and isinstance(block.ffn.gate, nn.Sequential):
            gate_linear = block.ffn.gate[0]
            if isinstance(gate_linear, nn.Linear):
                weight_key = f"blocks.{i}.ffn.gate.0.weight"
                quant_linear = _create_t5_quantized_linear_nvfp4(gate_linear, weight_key, weight_dict)
                if quant_linear is not None:
                    block.ffn.gate[0] = quant_linear


def _create_t5_quantized_linear_nvfp4(original_linear, weight_key, weight_dict):
    """
    Create a NVFP4 quantized linear layer from original linear layer.
    
    Args:
        original_linear: Original nn.Linear layer
        weight_key: Key for the weight in weight_dict
        weight_dict: Dictionary containing quantized weights
    
    Returns:
        T5QuantizedLinearNVFP4 instance or None if weights not found
    """
    in_features = original_linear.in_features
    out_features = original_linear.out_features
    bias = original_linear.bias is not None
    
    # Check if quantized weights exist
    weight_scale_key = f"{weight_key}_scale"
    weight_global_scale_key = f"{weight_key}_global_scale"
    
    if weight_key not in weight_dict or weight_scale_key not in weight_dict or weight_global_scale_key not in weight_dict:
        # Quantized weights don't exist, return None to skip quantization
        return None
    
    quant_linear = T5QuantizedLinearNVFP4(in_features, out_features, bias=bias)
    
    # Load quantized weights
    weight_fp4 = weight_dict[weight_key]  # uint8, packed fp4
    weight_scale = weight_dict[weight_scale_key]  # float8_e4m3fn, swizzled
    weight_global_scale = weight_dict[weight_global_scale_key]  # float32
    
    # Get bias if exists
    bias_tensor = None
    if bias:
        bias_key = weight_key.replace(".weight", ".bias")
        if bias_key in weight_dict:
            bias_tensor = weight_dict[bias_key]
    
    quant_linear.load_quantized_weights(weight_fp4, weight_scale, weight_global_scale, bias_tensor)
    
    return quant_linear


def load_t5_quantized_weights_nvfp4(checkpoint_dir):
    """
    Load NVFP4 quantized T5 weights from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing quantized T5 weights (.safetensors files)
    
    Returns:
        Dictionary containing quantized weights
    """
    if os.path.isdir(checkpoint_dir):
        # Look for safetensors files
        safetensors_files = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
        # Also check for index file
        index_file = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file, 'r') as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            safetensors_files = [os.path.join(checkpoint_dir, f) for f in set(weight_map.values())]
    else:
        safetensors_files = [checkpoint_dir]
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {checkpoint_dir}")
    
    weight_dict = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                weight_dict[key] = f.get_tensor(key)
    
    return weight_dict


def create_quantized_t5_encoder_nvfp4(text_len, dtype, device, checkpoint_path, quantized_checkpoint_dir, tokenizer_path, shard_fn=None):
    """
    Create a NVFP4 quantized T5 encoder model.
    
    Args:
        text_len: Text length
        dtype: Data type
        device: Device
        checkpoint_path: Path to original checkpoint (for model structure)
        quantized_checkpoint_dir: Directory containing quantized weights
        tokenizer_path: Path to tokenizer
        shard_fn: Optional sharding function
    
    Returns:
        T5EncoderModel instance with quantized weights
    """
    from .t5 import T5EncoderModel, umt5_xxl
    from .tokenizers import HuggingfaceTokenizer
    
    # Create model structure
    model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=dtype,
        device=device).eval().requires_grad_(False)
    
    # Load original state dict to get non-quantized weights
    logging.info(f'Loading model structure from {checkpoint_path}')
    original_state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Load quantized weights
    logging.info(f'Loading NVFP4 quantized weights from {quantized_checkpoint_dir}')
    quantized_weights = load_t5_quantized_weights_nvfp4(quantized_checkpoint_dir)
    
    # Replace Linear layers with quantized versions BEFORE loading state dict
    replace_t5_linear_with_quantized_nvfp4(model, quantized_weights)
    
    # Load non-quantized weights (embeddings, norms, etc.) from original checkpoint
    non_quantized_state_dict = {}
    for key in original_state_dict.keys():
        # Skip quantized layers (attn and ffn weights)
        if not any(x in key for x in ['attn.q.weight', 'attn.k.weight', 'attn.v.weight', 'attn.o.weight', 
                                      'ffn.fc1.weight', 'ffn.fc2.weight', 'ffn.gate.0.weight']):
            non_quantized_state_dict[key] = original_state_dict[key]
    
    model.load_state_dict(non_quantized_state_dict, strict=False)
    
    # Create T5EncoderModel instance manually (avoiding __init__ which loads weights)
    t5_model = object.__new__(T5EncoderModel)
    t5_model.text_len = text_len
    t5_model.dtype = dtype
    t5_model.device = device
    t5_model.checkpoint_path = checkpoint_path
    t5_model.tokenizer_path = tokenizer_path
    t5_model.model = model
    
    if shard_fn is not None:
        t5_model.model = shard_fn(t5_model.model, sync_module_states=False)
    else:
        t5_model.model.to(device)
    
    # Init tokenizer
    t5_model.tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path, seq_len=text_len, clean='whitespace')
    
    return t5_model


