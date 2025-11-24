"""
Quantized Linear layer for W4A16 (NVFP4 weights, BF16 activations)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer for W4A16.
    Weights are stored in NVFP4 format and dequantized to BF16 for computation.
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
        
        # Cache for dequantized weights (optional optimization)
        self._dequantized_weight = None
        self._weight_cached = False
    
    def load_quantized_weights(self, weight_fp4, weight_scale, weight_global_scale, bias=None):
        self.weight_fp4 = weight_fp4
        self.weight_scale = weight_scale
        self.weight_global_scale = weight_global_scale
        if bias is not None:
            self.bias = bias
        self._weight_cached = False
        self._dequantized_weight = None
    
    def forward(self, input):
        """
        Forward pass with BF16 activations and dequantized BF16 weights.
        TODO: Skip actual computation for memory/flow verification.
        
        Args:
            input: Input tensor in BF16, shape (..., in_features)
        
        Returns:
            Output tensor in BF16, shape (..., out_features)
        """
        # TODO: now we skip actual computation, just return random tensor with correct shape
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)
        
        # Get output shape without actual computation
        if input.dim() == 2:
            batch_size, _ = input.shape
            output_shape = (batch_size, self.out_features)
        elif input.dim() == 3:
            batch_size, seq_len, _ = input.shape
            output_shape = (batch_size, seq_len, self.out_features)
        else:
            *batch_dims, _ = input.shape
            output_shape = (*batch_dims, self.out_features)

        output = torch.randn(
            output_shape,
            dtype=torch.bfloat16,
            device=input.device,
            requires_grad=False
        )
        
        return output

