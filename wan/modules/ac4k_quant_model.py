"""
Modified WanModel with W4A16 quantization support
"""
import os
import glob
import json
import torch
import torch.nn as nn
from safetensors import safe_open
from .model import WanModel
from .ac4k_quant_linear import QuantizedLinear


def replace_linear_with_quantized(model, weight_dict):
    """
    Replace Linear layers in model with QuantizedLinear layers.

    Args:
        model: WanModel instance
        weight_dict: Dictionary containing quantized weights
    """
    # modules to quantize: self_attn, cross_attn, ffn
    target_modules = ['self_attn', 'cross_attn', 'ffn']

    for i, block in enumerate(model.blocks):
        # Self attention: q, k, v, o
        for name in ['q', 'k', 'v', 'o']:
            linear = getattr(block.self_attn, name)
            weight_key = f"blocks.{i}.self_attn.{name}.weight"
            quant_linear = _create_quantized_linear(
                linear, weight_key, weight_dict)
            if quant_linear is not None:
                setattr(block.self_attn, name, quant_linear)

        # Cross attention: q, k, v, o
        for name in ['q', 'k', 'v', 'o']:
            linear = getattr(block.cross_attn, name)
            weight_key = f"blocks.{i}.cross_attn.{name}.weight"
            quant_linear = _create_quantized_linear(
                linear, weight_key, weight_dict)
            if quant_linear is not None:
                setattr(block.cross_attn, name, quant_linear)

        # FFN: two Linear layers
        ffn_linear0 = block.ffn[0]
        ffn_linear2 = block.ffn[2]

        weight_key0 = f"blocks.{i}.ffn.0.weight"
        quant_linear0 = _create_quantized_linear(
            ffn_linear0, weight_key0, weight_dict)
        if quant_linear0 is not None:
            block.ffn[0] = quant_linear0

        weight_key2 = f"blocks.{i}.ffn.2.weight"
        quant_linear2 = _create_quantized_linear(
            ffn_linear2, weight_key2, weight_dict)
        if quant_linear2 is not None:
            block.ffn[2] = quant_linear2

    # maybe replace text embedding Linear layers
    for i, layer in enumerate(model.text_embedding):
        if isinstance(layer, nn.Linear):
            weight_key = f"text_embedding.{i}.weight"
            quant_linear = _create_quantized_linear(
                layer, weight_key, weight_dict)
            if quant_linear is not None:
                model.text_embedding[i] = quant_linear

    # maybe time embedding Linear layers
    for i, layer in enumerate(model.time_embedding):
        if isinstance(layer, nn.Linear):
            weight_key = f"time_embedding.{i}.weight"
            quant_linear = _create_quantized_linear(
                layer, weight_key, weight_dict)
            if quant_linear is not None:
                model.time_embedding[i] = quant_linear

    # maybe time projection Linear layer
    time_proj_linear = model.time_projection[1]
    weight_key = "time_projection.1.weight"
    quant_linear = _create_quantized_linear(
        time_proj_linear, weight_key, weight_dict)
    if quant_linear is not None:
        model.time_projection[1] = quant_linear

    # maybe Replace head Linear layer
    head_linear = model.head.head
    weight_key = "head.head.weight"
    quant_linear = _create_quantized_linear(
        head_linear, weight_key, weight_dict)
    if quant_linear is not None:
        model.head.head = quant_linear


def convert_scale_into_swizzle(scales_linear: torch.Tensor, m: int, k: int, block_size: int = 16, num_cols: int = None):
    rounded_m = ((m + 128 - 1) // 128) * 128
    rounded_k = ((k + 4 - 1) // 4) * 4

    if num_cols is None:
        num_cols = k * block_size

    scales_padded = torch.zeros(
        rounded_m, rounded_k, dtype=scales_linear.dtype, device=scales_linear.device)
    scales_padded[:m, :k] = scales_linear

    # Calculate tile dimensions
    # Swizzled layout uses numCols (original column count) to calculate numKTiles
    # NOT k (scales count) to calculate k_tiles
    m_tiles = rounded_m // 128
    f = block_size * 4  # 64
    numKTiles = (num_cols + f - 1) // f

    # Swizzled layout structure: [numMTiles=40, numKTiles=80, 32, 4, 4]
    # We need to organize all k=320 scales into this structure

    # Process all k scales: organize them into swizzled layout
    # Reshape linear scales to [m_tiles, 128, k]
    scales_reshaped = scales_padded.reshape(m_tiles, 4, 32, numKTiles, 4)

    # Permute to [m_tiles, numKTiles, 32, 4, 4] (swizzled layout structure)
    scales_swizzled = scales_reshaped.permute(0, 3, 2, 1, 4)

    # Reshape to [rounded_m, rounded_k] (unpacked)
    scales_swizzled = scales_swizzled.reshape(rounded_m, rounded_k)

    # Pack 4 scales per int32: [rounded_m, rounded_k] -> [rounded_m, rounded_k//4]
    scales_swizzled_uint8 = scales_swizzled.view(torch.uint8)
    scales_swizzled_packed = scales_swizzled_uint8.reshape(
        rounded_m * rounded_k // 4, 4).contiguous().view(torch.int32).reshape(rounded_m, rounded_k // 4)

    # Return as float8_e4m3fn view
    # scales_swizzled_packed is int32 with shape [rounded_m, rounded_k//4] = [5120, 80]
    # Each int32 contains 4 float8_e4m3fn values (packed as uint8)
    # When view as float8_e4m3fn, we need to unpack first
    scales_swizzled_packed_uint8 = scales_swizzled_packed.view(
        torch.uint8)  # [5120, 320] uint8
    result = scales_swizzled_packed_uint8.view(
        torch.float8_e4m3fn)  # [5120, 320] float8_e4m3fn
    # Verify shape is correct: should be [rounded_m, rounded_k] = [5120, 320]
    expected_shape = (rounded_m, rounded_k)
    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"
    return result


def _create_quantized_linear(original_linear, weight_key, weight_dict):
    out_features = original_linear.out_features
    in_features = original_linear.in_features
    bias = original_linear.bias is not None

    # Check if quantized weights exist
    weight_scale_key = f"{weight_key}_scale"
    weight_global_scale_key = f"{weight_key}_global_scale"

    if weight_key not in weight_dict or weight_scale_key not in weight_dict or weight_global_scale_key not in weight_dict:
        return None

    quant_linear = QuantizedLinear(out_features, bias=bias)

    weight_fp4 = weight_dict[weight_key]  # uint8, packed fp4

    weight_scale = weight_dict[weight_scale_key]
    weight_global_scale = weight_dict[weight_global_scale_key]  # float32

    # Convert scales from linear layout to swizzled layout
    # weight_scale shape: [out_features, in_features//16] (linear layout)
    # Need to convert to swizzled layout expected by QuantizedLinear
    m, k = weight_scale.shape  # m = out_features, k = in_features // 16

    # TODO: change to swizzled layout after using linear layout in weight scale
    # weight_scale_swizzled = weight_scale
    weight_scale_swizzled = convert_scale_into_swizzle(
        weight_scale, m, k, num_cols=in_features)

    bias_tensor = None
    if bias:
        bias_key = weight_key.replace(".weight", ".bias")
        if bias_key in weight_dict:
            bias_tensor = weight_dict[bias_key]

    quant_linear.load_quantized_weights(
        weight_fp4, weight_scale_swizzled, weight_global_scale, bias_tensor)

    return quant_linear


def load_quantized_weights(checkpoint_dir, subfolder=None):
    if subfolder:
        checkpoint_path = os.path.join(checkpoint_dir, subfolder)
    else:
        checkpoint_path = checkpoint_dir

    if os.path.isdir(checkpoint_path):
        safetensors_files = glob.glob(
            os.path.join(checkpoint_path, "*.safetensors"))
    else:
        safetensors_files = [checkpoint_path]

    weight_dict = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                weight_dict[key] = f.get_tensor(key)

    return weight_dict


def create_quantized_wan_model(quantized_ckpt_dir, subfolder=None):
    config_path = os.path.join(
        quantized_ckpt_dir, subfolder) if subfolder else quantized_ckpt_dir
    config_file = os.path.join(config_path, "config.json")

    if not os.path.exists(config_file):
        raise ValueError(
            f"Config file not found: {config_file}. "
            f"Please ensure config.json exists in {quantized_ckpt_dir}.")

    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    model_config = {}
    for config_key in config_dict.keys():
        model_config[config_key] = config_dict[config_key]

    model = WanModel(**model_config)

    weight_dict = load_quantized_weights(
        quantized_ckpt_dir, subfolder=subfolder)

    # Replace quantized linear layers
    replace_linear_with_quantized(model, weight_dict)

    # Load non-quantized weights (norm layers, embeddings, etc.)
    quantized_keys = set()
    for key in weight_dict.keys():
        if key.endswith('_scale') or key.endswith('_global_scale'):
            base_key = key.rsplit('_', 1)[0] if key.endswith(
                '_scale') else key.rsplit('_global_scale', 1)[0]
            quantized_keys.add(base_key)

    for key, value in weight_dict.items():
        if key.endswith('_scale') or key.endswith('_global_scale'):
            continue
        if key in quantized_keys:
            continue  # This is a quantized weight, already handled by replace_linear_with_quantized

        # Try to load non-quantized weights
        parts = key.split('.')
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)

        param_name = parts[-1]
        if hasattr(obj, param_name):
            param = getattr(obj, param_name)
            if isinstance(param, torch.nn.Parameter) or isinstance(param, torch.Tensor):
                if value.shape == param.shape:
                    if isinstance(param, torch.nn.Parameter):
                        param.data.copy_(value)
                    else:
                        setattr(obj, param_name, value)
                else:
                    raise ValueError(f"Shape mismatch for {key}: expected {param.shape}, got {value.shape}")

    return model
