"""
Modified WanModel with W4A16 quantization support
"""
import os
import glob
import torch
import torch.nn as nn
from safetensors import safe_open
from .model import WanModel
from .quant_linear import QuantizedLinear


def replace_linear_with_quantized(model, weight_dict):
    """
    Replace Linear layers in model with QuantizedLinear layers.
    
    Args:
        model: WanModel instance
        weight_dict: Dictionary containing quantized weights
    """
    # modules to quantize: self_attn, cross_attn, ffn
    target_modules = ['self_attn', 'cross_attn', 'ffn']
    
    # Replace Linear layers in blocks
    for i, block in enumerate(model.blocks):
        # Self attention: q, k, v, o
        for name in ['q', 'k', 'v', 'o']:
            linear = getattr(block.self_attn, name)
            weight_key = f"blocks.{i}.self_attn.{name}.weight"
            quant_linear = _create_quantized_linear(linear, weight_key, weight_dict)
            if quant_linear is not None:
                setattr(block.self_attn, name, quant_linear)
        
        # Cross attention: q, k, v, o
        for name in ['q', 'k', 'v', 'o']:
            linear = getattr(block.cross_attn, name)
            weight_key = f"blocks.{i}.cross_attn.{name}.weight"
            quant_linear = _create_quantized_linear(linear, weight_key, weight_dict)
            if quant_linear is not None:
                setattr(block.cross_attn, name, quant_linear)
        
        # FFN: two Linear layers
        ffn_linear0 = block.ffn[0]
        ffn_linear2 = block.ffn[2]
        
        weight_key0 = f"blocks.{i}.ffn.0.weight"
        quant_linear0 = _create_quantized_linear(ffn_linear0, weight_key0, weight_dict)
        if quant_linear0 is not None:
            block.ffn[0] = quant_linear0
        
        weight_key2 = f"blocks.{i}.ffn.2.weight"
        quant_linear2 = _create_quantized_linear(ffn_linear2, weight_key2, weight_dict)
        if quant_linear2 is not None:
            block.ffn[2] = quant_linear2
    
    # Replace text embedding Linear layers
    for i, layer in enumerate(model.text_embedding):
        if isinstance(layer, nn.Linear):
            weight_key = f"text_embedding.{i}.weight"
            quant_linear = _create_quantized_linear(layer, weight_key, weight_dict)
            if quant_linear is not None:
                model.text_embedding[i] = quant_linear
    
    # Replace time embedding Linear layers
    for i, layer in enumerate(model.time_embedding):
        if isinstance(layer, nn.Linear):
            weight_key = f"time_embedding.{i}.weight"
            quant_linear = _create_quantized_linear(layer, weight_key, weight_dict)
            if quant_linear is not None:
                model.time_embedding[i] = quant_linear

    # Replace time projection Linear layer
    time_proj_linear = model.time_projection[1]
    weight_key = "time_projection.1.weight"
    quant_linear = _create_quantized_linear(time_proj_linear, weight_key, weight_dict)
    if quant_linear is not None:
        model.time_projection[1] = quant_linear
    
    # Replace head Linear layer
    head_linear = model.head.head
    weight_key = "head.head.weight"
    quant_linear = _create_quantized_linear(head_linear, weight_key, weight_dict)
    if quant_linear is not None:
        model.head.head = quant_linear


def _create_quantized_linear(original_linear, weight_key, weight_dict):
    in_features = original_linear.in_features
    out_features = original_linear.out_features
    bias = original_linear.bias is not None
    
    # Check if quantized weights exist
    weight_scale_key = f"{weight_key}_scale"
    weight_global_scale_key = f"{weight_key}_global_scale"
    
    if weight_key not in weight_dict or weight_scale_key not in weight_dict or weight_global_scale_key not in weight_dict:
        # Quantized weights don't exist, return None to skip quantization
        return None
    
    quant_linear = QuantizedLinear(in_features, out_features, bias=bias)
    
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


def load_quantized_weights(checkpoint_dir, subfolder=None):
    if subfolder:
        checkpoint_path = os.path.join(checkpoint_dir, subfolder)
    else:
        checkpoint_path = checkpoint_dir

    if os.path.isdir(checkpoint_path):
        safetensors_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    else:
        safetensors_files = [checkpoint_path]
    
    weight_dict = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                weight_dict[key] = f.get_tensor(key)

    calib_path = os.path.join(checkpoint_path, "calib.pt")
    
    return weight_dict


def create_quantized_wan_model(quantized_ckpt_dir, original_ckpt_dir=None, subfolder=None):
    # First try from quantized directory, if not found, try from original directory
    model = None
    config_path = os.path.join(quantized_ckpt_dir, subfolder) if subfolder else quantized_ckpt_dir
    config_file = os.path.join(config_path, "config.json")

    if os.path.exists(config_file):
        # Config exists in quantized directory
        model = WanModel.from_pretrained(quantized_ckpt_dir, subfolder=subfolder)
    elif original_ckpt_dir is not None:
        # Try to load config from original directory
        original_config_path = os.path.join(original_ckpt_dir, subfolder) if subfolder else original_ckpt_dir
        original_config_file = os.path.join(original_config_path, "config.json")
        if os.path.exists(original_config_file):
            model = WanModel.from_pretrained(original_ckpt_dir, subfolder=subfolder)
        else:
            # Fallback: try parent directory
            model = WanModel.from_pretrained(original_ckpt_dir)
    else:
        model = WanModel.from_pretrained(quantized_ckpt_dir)

    if model is None:
        raise ValueError(f"Could not load model config. Please ensure config.json exists in {quantized_ckpt_dir} or provide original_ckpt_dir.")

    weight_dict = load_quantized_weights(quantized_ckpt_dir, subfolder=subfolder)
    replace_linear_with_quantized(model, weight_dict)

    return model

