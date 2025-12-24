"""
Quantized Linear layer for W4A16 (NVFP4 weights, BF16 activations)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer for W4A16.
    Weights are stored in NVFP4 format and dequantized to BF16 for computation.
    Activations remain in BF16 (matching Wan2.2 T2V 14B model).
    """

    def __init__(self, out_features, bias=True):
        super().__init__()

        # Register buffers for quantized weights
        self.register_buffer('weight_fp4', None)  # uint8, packed fp4
        self.register_buffer('weight_scale', None)  # float8_e4m3fn, swizzled
        self.register_buffer('weight_global_scale', None)  # float32
        self.register_buffer(
            'bias', None if not bias else torch.zeros(out_features))
        # Backup for weight_scale to restore dtype after conversion
        self._weight_scale_backup = None

    def load_quantized_weights(self, weight_fp4, weight_scale, weight_global_scale, bias=None):
        self.weight_fp4 = weight_fp4
        self.weight_scale = weight_scale
        # Save a persistent backup of weight_scale for dtype restoration
        if weight_scale is not None:
            self._weight_scale_backup = weight_scale.clone().detach()
        self.weight_global_scale = weight_global_scale
        if bias is not None:
            self.bias = bias

    def _apply(self, fn):
        """
        Override _apply to preserve weight_scale dtype as float8_e4m3fn.
        This is called by to(), cuda(), cpu(), etc. to apply transformations.
        """

        weight_scale_backup = None
        if self._weight_scale_backup is not None:
            weight_scale_backup = self._weight_scale_backup
        elif self.weight_scale is not None:
            weight_scale_backup = self.weight_scale.clone()

        # Apply base function to all buffers and parameters
        result = super()._apply(fn)

        # Restore weight_scale to float8_e4m3fn if it was changed
        if weight_scale_backup is not None and self.weight_scale is not None:
            # restore dtype
            if self.weight_scale.dtype != torch.float8_e4m3fn:
                current_device = self.weight_scale.device
                self.weight_scale = weight_scale_backup.to(
                    device=current_device, dtype=torch.float8_e4m3fn)
            # restore device
            elif self.weight_scale.device != weight_scale_backup.device:
                current_device = self.weight_scale.device
                self.weight_scale = weight_scale_backup.to(
                    device=current_device, dtype=torch.float8_e4m3fn)

        return result

    def forward(self, input):
        """
        Forward pass with BF16 activations and dequantized BF16 weights.
        """
        # Ensure weight_scale is in correct dtype (float8_e4m3fn)
        # This is a safety check - the _apply method should prevent dtype conversion
        if self.weight_scale is not None and self.weight_scale.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"weight_scale dtype mismatch: got {self.weight_scale.dtype}, "
                "expected float8_e4m3fn. This indicates that _apply method did not "
                "correctly preserve the dtype. Please check the model loading process."
            )

        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        # Quantize input to nvfp4
        input_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                              torch.amax(torch.abs(input.flatten()), dim=-1)).to(torch.float32)
        alpha = 1.0 / (input_global_scale * self.weight_global_scale)
        assert (input.dim() == 3 or input.dim()
                == 2), "input should be 3D or 2D"
        if input.dim() == 3:
            assert input.shape[0] == 1, "batch size should be 1"
            input = input.squeeze(0)
        input_fp4, input_scale_interleaved = nvfp4_quant(
            input, input_global_scale)

        if self.bias is not None and self.bias.dim() == 1:
            self.bias = self.bias.unsqueeze(0)
        output = nvfp4_matmul(input_fp4, self.weight_fp4,
                              input_scale_interleaved, self.weight_scale, alpha, self.bias)

        return output
