"""
Wan2.2 DiT Model NVFP4 Offline Quantization Script

This script quantizes Wan2.2 DiT model weights to NVFP4 format for efficient storage.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from wan_quant import NVFP4Quantizer
except ImportError:
    print("Error: wan_quant package not found. Please install it first:")
    print("  cd offline_quant && pip install -e .")
    sys.exit(1)

def quantize_linear_layer(quantizer, weight_tensor, layer_name):
    """
    Quantize a single linear layer weight.

    Args:
        quantizer: NVFP4Quantizer instance
        weight_tensor: Weight tensor [out_features, in_features]
        layer_name: Name of the layer for logging

    Returns:
        Dictionary with quantized weights and scales
    """
    print(f"Quantizing {layer_name}: shape {weight_tensor.shape}")

    if weight_tensor.dim() != 2:
        raise ValueError(
            f"Expected 2D weight tensor, got {weight_tensor.dim()}D for {layer_name}")

    M, N = weight_tensor.shape
    if N % 16 != 0:
        raise ValueError(
            f"N dimension ({N}) must be multiple of 16 for {layer_name}")

    quantized, scales, global_scale = quantizer.quantize(weight_tensor)

    return {
        f"{layer_name}": quantized,
        f"{layer_name}".replace("weight", "weight_scale"): scales,
        f"{layer_name}".replace("weight", "weight_global_scale"): global_scale,
    }


def quantize_model_checkpoint(
    input_path: str,
    output_path: str,
    subfolder: str = None
):
    """
    Quantize a model checkpoint to NVFP4 format.

    Args:
        input_path: Path to input checkpoint directory or file
        output_path: Path to output directory
        subfolder: Optional subfolder name (e.g., 'low_noise_model', 'high_noise_model')
    """
    quantizer = NVFP4Quantizer()

    if subfolder:
        input_dir = os.path.join(input_path, subfolder)
    else:
        input_dir = input_path

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    if subfolder:
        output_dir = os.path.join(output_path, subfolder)
    else:
        output_dir = output_path

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_dir):
        safetensors_files = list(Path(input_dir).glob("*.safetensors"))
    else:
        safetensors_files = [Path(input_dir)] if input_dir.endswith(
            '.safetensors') else []

    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")

    print(f"Found {len(safetensors_files)} safetensors file(s)")

    all_quantized_weights = {}
    valid_keys = ["ffn", "self_attn", "cross_attn"]
    invalid_keys = ["norm"]

    for safetensors_file in safetensors_files:
        print(f"\nProcessing {safetensors_file.name}...")

        with safe_open(safetensors_file, framework="pt") as f:
            keys = list(f.keys())
            print(f"  Found {len(keys)} tensors")
            for key in tqdm(keys, desc=f"  Quantizing {safetensors_file.name}"):
                has_valid_key = any(
                    valid_key in key for valid_key in valid_keys)
                has_invalid_key = any(
                    invalid_key in key for invalid_key in invalid_keys)
                is_valid = has_valid_key and (not has_invalid_key)
                tensor = f.get_tensor(key)
                # Only quantize linear layer weights (2D tensors with 'weight' in name)
                if 'weight' in key and tensor.dim() == 2 and is_valid:
                    try:
                        quantized_dict = quantize_linear_layer(
                            quantizer, tensor, key)
                        all_quantized_weights.update(quantized_dict)
                    except Exception as e:
                        raise ValueError(f"Failed to quantize {key}: {e}")
                else:
                    all_quantized_weights[key] = tensor.to(torch.bfloat16)

    print("\nEnsuring all tensors are on CPU...")
    for key, tensor in all_quantized_weights.items():
        if tensor.is_cuda:
            all_quantized_weights[key] = tensor.cpu()

    output_file = os.path.join(output_dir, "quantized_weights.safetensors")
    print(f"\nSaving quantized weights to {output_file}...")
    save_file(all_quantized_weights, output_file)
    print(f"Saved {len(all_quantized_weights)} tensors")

def main():
    parser = argparse.ArgumentParser(
        description="Quantize Wan2.2 DiT model to NVFP4 format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input checkpoint directory or file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for quantized weights"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Optional subfolder name (e.g., 'low_noise_model', 'high_noise_model')"
    )

    args = parser.parse_args()

    try:
        quantize_model_checkpoint(
            args.input,
            args.output,
            args.subfolder
        )
        print("\nQuantization completed successfully!")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
