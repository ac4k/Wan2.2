"""NVFP4 Quantizer for Wan2.2 DiT Models"""

import ctypes
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional


class NVFP4Quantizer:
    """
    NVFP4 (NVIDIA FP4) Quantizer for offline weight quantization.
    """

    def __init__(self):
        try:
            import wan_quant_kernel
            self.kernel_available = True
        except ImportError:
            raise ImportError(
                "CUDA kernel not available, please install wan_quant_kernel")

    def quantize(
        self,
        weight: torch.Tensor,
        global_scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a weight tensor to NVFP4 format.

        Args:
            weight: Input weight tensor of shape [M, N], dtype bfloat16 or float32
            global_scale: Optional global scaling factor. If None, computed automatically.

        Returns:
            Tuple of:
            - quantized: Quantized weights [M, N/2] uint8 (2 e2m1 values per uint8) - LINEAR LAYOUT
            - scales: Scale factors in LINEAR LAYOUT [M, N/16] float8_e4m3fn
            - global_scale: Global scale factor used (float32)
        """
        # Ensure 2D tensor
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {weight.dim()}D")

        M, N = weight.shape

        # Check N is multiple of 16 (required for block-wise quantization)
        if N % 16 != 0:
            raise ValueError(f"N dimension ({N}) must be multiple of 16")

        # Convert to bfloat16 if needed
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        # Move to CUDA if needed - MUST be on CUDA
        if not weight.is_cuda:
            weight = weight.cuda()

        # Get device AFTER moving to CUDA to ensure it's CUDA
        device = weight.device
        assert weight.is_cuda, f"Weight must be on CUDA device, but got {device}"

        # Compute global scale if not provided
        if global_scale is None:
            global_scale = (
                2688.0 / torch.max(torch.abs(weight))).to(torch.float32)

        # Prepare output tensors - MUST be on CUDA
        # Each uint32_t contains 8 e2m1 values (4 bits each)
        # Weight: LINEAR LAYOUT [M, N/8]
        quantized_shape = (M, N // 8)

        # Scales: LINEAR LAYOUT [M, N/16]
        # Each scale is a float8_e4m3fn value stored as uint8
        block_size = 16
        scale_n = N // block_size  # Number of scales per row
        scales_shape = (M, scale_n)

        output_quantized = torch.empty(
            quantized_shape,
            dtype=torch.uint32,
            device=device
        )
        assert output_quantized.is_cuda, "output_quantized must be on CUDA device"

        # Create linear scales tensor as uint8 (float8_e4m3fn values)
        output_scales = torch.zeros(
            scales_shape,
            dtype=torch.uint8,
            device=device
        )
        assert output_scales.is_cuda, "output_scales must be on CUDA device"

        try:
            # Use torch.ops to call the registered operator
            # This matches the TORCH_LIBRARY_FRAGMENT registration
            torch.ops.wan_quant_kernel.nvfp4_quantize_cuda(
                weight,
                output_quantized,
                output_scales,
                global_scale
            )
        except Exception as e:
            raise Exception(f"CUDA kernel failed: {e}")

        # Convert uint32 to uint8 format
        # Each uint32 contains 8 e2m1 values (4 bits each) = 4 bytes
        # Each uint8 will contain 2 e2m1 values (4 bits each) = 1 byte
        # So [M, N/8] uint32 -> [M, N/2] uint8
        # We need to unpack the uint32 values into bytes (little-endian)
        # view(torch.uint8) will reinterpret the memory as uint8, giving us 4 bytes per uint32
        quantized_uint8 = output_quantized.view(torch.uint8)
        # Reshape from [M, N/8, 4] to [M, N/2]
        # Total elements: M * (N/8) * 4 = M * N / 2
        quantized_uint8 = quantized_uint8.reshape(M, N // 2)

        # Convert uint8 scale bytes to float8_e4m3fn format
        # The uint8 values are already the raw bytes of float8_e4m3fn
        # We need to reinterpret them as float8_e4m3fn
        # Check if float8_e4m3fn is available
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                "torch.float8_e4m3fn is not available in this PyTorch version. "
                "Please use PyTorch 2.1 or later."
            )

        # Ensure contiguous memory layout before view operation
        # view() requires the tensor to be contiguous
        # Scales are in LINEAR LAYOUT [M, N/16]
        output_scales_contiguous = output_scales.contiguous()
        scales_fp8 = output_scales_contiguous.view(torch.float8_e4m3fn)

        # Move to CPU for saving to safetensors (safetensors requires CPU tensors)
        quantized_uint8 = quantized_uint8.cpu()
        scales_fp8 = scales_fp8.cpu()
        global_scale_tensor = torch.tensor(
            global_scale, dtype=torch.float32, device='cpu')

        return quantized_uint8, scales_fp8, global_scale_tensor
