"""NVFP4 Quantizer for Wan2.2 DiT Models"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ac4k_kernel.ops import quantize as quantize_fn


class NVFP4Quantizer:
    """
    NVFP4 (NVIDIA FP4) Quantizer for offline weight quantization.
    """

    def __init__(self):
        pass

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

        if weight.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {weight.dim()}D")

        M, N = weight.shape

        if N % 16 != 0:
            raise ValueError(f"N dimension ({N}) must be multiple of 16")

        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        if not weight.is_cuda:
            weight = weight.cuda()

        device = weight.device
        assert weight.is_cuda, f"Weight must be on CUDA device, but got {device}"

        # Use ac4k_kernel.ops.quantize
        # cross_dim=0 (K dimension), reduce_dim=1 (N dimension)
        quantized_uint8, scales_swizzled, global_scale_tensor = quantize_fn(
            weight,
            cross_dim=0,
            reduce_dim=1
        )


        # Move to CPU
        quantized_uint8 = quantized_uint8.cpu()
        scales_swizzled = scales_swizzled.cpu()
        if isinstance(global_scale_tensor, torch.Tensor):
            global_scale_tensor = global_scale_tensor.cpu()
        else:
            global_scale_tensor = torch.tensor(
                global_scale_tensor, dtype=torch.float32, device='cpu'
            )

        return quantized_uint8, scales_swizzled, global_scale_tensor


