"""
Quantized Linear layer for T5 NVFP4 (NVFP4 weights, BF16 activations)
"""
import torch
import torch.nn as nn
from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


class T5QuantizedLinearNVFP4(nn.Module):
    """
    Quantized Linear layer for T5 NVFP4.
    Weights are stored in NVFP4 format and require ac4k_kernel for computation.
    Activations remain in BF16 (matching Wan2.2 T2V 14B model).
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Register buffers for quantized weights
        self.register_buffer('weight_fp4', None)  # uint8, packed fp4
        self.register_buffer('weight_scale', None)  # float8_e4m3fn, swizzled
        self.register_buffer('weight_global_scale', None)  # float32
        self.register_buffer('bias', None if not bias else torch.zeros(out_features))
        
        # Backup for weight_scale to restore dtype after conversion
        self._weight_scale_backup = None
    
    def load_quantized_weights(self, weight_fp4, weight_scale, weight_global_scale, bias=None):
        """
        Load quantized weights.
        
        Args:
            weight_fp4: Quantized weight tensor (uint8, packed fp4)
            weight_scale: Scale tensor (float8_e4m3fn, swizzled)
            weight_global_scale: Global scale (float32)
            bias: Optional bias tensor
        """
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
        # Use the persistent backup if available, otherwise create a temporary one
        weight_scale_backup = None
        if self._weight_scale_backup is not None:
            weight_scale_backup = self._weight_scale_backup
        elif self.weight_scale is not None:
            weight_scale_backup = self.weight_scale.clone()

        # Apply function to all buffers and parameters
        result = super()._apply(fn)

        # Restore weight_scale to float8_e4m3fn if it was changed
        if weight_scale_backup is not None and self.weight_scale is not None:
            # If dtype was changed, restore it while keeping the new device
            if self.weight_scale.dtype != torch.float8_e4m3fn:
                current_device = self.weight_scale.device
                self.weight_scale = weight_scale_backup.to(device=current_device, dtype=torch.float8_e4m3fn)
            # If only device changed, update device while keeping dtype
            elif self.weight_scale.device != weight_scale_backup.device:
                current_device = self.weight_scale.device
                self.weight_scale = weight_scale_backup.to(device=current_device, dtype=torch.float8_e4m3fn)

        return result
    
    def forward(self, input):
        """
        Forward pass with BF16 activations using NVFP4 quantized weights.
        
        Args:
            input: Input tensor in BF16, shape (..., in_features)
        
        Returns:
            Output tensor in BF16, shape (..., out_features)
        """
        # Ensure weight_scale is in correct dtype (float8_e4m3fn)
        if self.weight_scale is not None and self.weight_scale.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"weight_scale dtype mismatch: got {self.weight_scale.dtype}, "
                "expected float8_e4m3fn. This indicates that _apply method did not "
                "correctly preserve the dtype. Please check the model loading process."
            )

        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        # Handle multi-dimensional input (e.g., [batch, seq_len, dim])
        original_shape = input.shape
        if input.dim() > 2:
            # Reshape to 2D: [batch*seq_len, dim]
            input_2d = input.reshape(-1, input.shape[-1])
        else:
            input_2d = input

        # Quantize input to NVFP4
        input_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                    torch.amax(torch.abs(input_2d.flatten()), dim=-1)).to(torch.float32)
        alpha = 1.0 / (input_global_scale * self.weight_global_scale)
        input_fp4, input_scale_interleaved = nvfp4_quant(input_2d, input_global_scale)

        if self.bias is not None and self.bias.dim() == 1:
            bias = self.bias.unsqueeze(0)
        else:
            bias = self.bias

        # Use optimized NVFP4 kernel
        output_2d = nvfp4_matmul(
            input_fp4, 
            self.weight_fp4, 
            input_scale_interleaved, 
            self.weight_scale, 
            alpha, 
            bias
        )
        
        # Reshape output back to original shape (except last dimension)
        if len(original_shape) > 2:
            output = output_2d.reshape(*original_shape[:-1], output_2d.shape[-1])
        else:
            output = output_2d
        
        return output


