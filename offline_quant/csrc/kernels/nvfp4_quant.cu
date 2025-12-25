/*
 * NVFP4 Quantization CUDA Kernel Implementation
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdio>

// NVFP4 uses e2m1 format (2 exponent bits, 1 mantissa bit)
// Each 4-bit value represents a floating point number
// Two 4-bit values are packed into one uint8, four pairs into one uint32

// Block size for quantization: 16 elements share one scale factor
#define QUANT_BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 8

// Tile-based interleaved layout constants for scale factor storage
#define SCALE_VECTOR_SIZE 16  // Number of weight elements per scale factor
#define THREADS_PER_SCALE 1   // Each thread processes one quantization block (one scale)

// Convert 8 float32 values to e2m1 format using PTX instruction
// Uses cvt.rn.satfinite.e2m1x2.f32 to convert 2 f32 values at once
// Returns uint32_t containing 8 e2m1 values (4 bits each)
__device__ inline uint32_t float8_to_e2m1_packed(float vals[8]) {
    uint32_t result;
    
    // Use PTX inline assembly to convert 2 f32 values to 2 e2m1 values at once
    // cvt.rn.satfinite.e2m1x2.f32 converts 2 f32 to 2 e2m1 (packed in 1 byte)
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(result)
        : "f"(vals[0]), "f"(vals[1]),
          "f"(vals[2]), "f"(vals[3]),
          "f"(vals[4]), "f"(vals[5]),
          "f"(vals[6]), "f"(vals[7])
    );
    
    return result;
}

// Fast reciprocal approximation using PTX
__device__ inline float fast_rcp(float a) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(a));
    return result;
}

// Compute scale factor for a block of 16 values
// Returns the scale value that will be quantized to fp8
// NOTE: block_data contains ORIGINAL values (NOT scaled by global_scale)
__device__ inline float compute_block_scale_fp8(const float* block_data, float global_scale) {
    // Find maximum absolute value in the block (original values)
    float block_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < QUANT_BLOCK_SIZE; i++) {
        block_max = fmaxf(block_max, fabsf(block_data[i]));
    }

    // 0.16666666666666666f = 1.0 / 6.0
    float scale_fp8_value = global_scale * (block_max * 0.16666666666666666f);
    
    return scale_fp8_value;
}

// Compute the output scale used for quantizing input values
// output_scale = global_scale / quantized_scale_fp32
__device__ inline float compute_output_scale(float global_scale, float quantized_scale_fp32) {
    if (quantized_scale_fp32 > 1e-8f) {
        return global_scale * fast_rcp(quantized_scale_fp32);
    }
    return 0.0f;
}

// Quantize a single block of 16 float32 values to NVFP4
// Output: 2 uint32_t values (8 e2m1 values per uint32, 4 bytes each)
// Scale: 1 float8_e4m3fn value
__device__ void quantize_block(
    const float* input_block,
    uint32_t* output_quantized,
    uint8_t* output_scale,
    float global_scale) {
    // Step 1: Compute scale value that will be quantized to fp8
    float scale_fp8_value = compute_block_scale_fp8(input_block, global_scale);
    
    // Step 2: Quantize scale to float8_e4m3fn
    uint8_t fp8_scale_in_uint8;
    __nv_fp8_e4m3 scale_fp8 = __nv_fp8_e4m3(scale_fp8_value);
    fp8_scale_in_uint8 = scale_fp8.__x;

    if (output_scale != nullptr) {
        *output_scale = fp8_scale_in_uint8;
    }
    
    // Step 3: Convert quantized scale back to fp32 for computing output scale
    float scale_fp8_fp32 = static_cast<float>(scale_fp8);
    
    // Step 4: Compute output scale for quantizing input values
    float output_scale_value = compute_output_scale(global_scale, scale_fp8_fp32);
    
    // Step 5: Scale the input values using output_scale
    float scaled_block[QUANT_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < QUANT_BLOCK_SIZE; i++) {
        scaled_block[i] = input_block[i] * output_scale_value;
    }
    
    // Step 6: Convert first 8 values to e2m1 (packed in uint32_t)
    output_quantized[0] = float8_to_e2m1_packed(scaled_block);
    
    // Step 7: Convert next 8 values to e2m1 (packed in uint32_t)
    output_quantized[1] = float8_to_e2m1_packed(scaled_block + 8);
}

// Main quantization kernel
// Input: [M, N] tensor in bfloat16
// Output quantized: [M, N/8] tensor in uint32_t (8 e2m1 values per uint32) - LINEAR LAYOUT
// Output scales: [M, N/16] tensor in uint8 (float8_e4m3fn) - LINEAR LAYOUT
__global__ void nvfp4_quantize_kernel(
    const __nv_bfloat16* input,
    uint32_t* output_quantized,
    uint8_t* output_scales,
    int M,
    int N,
    float global_scale) {
    int row = blockIdx.x;
    int num_blocks_per_row = N / QUANT_BLOCK_SIZE;
    if (row >= M) {
        return;
    }
    
    // Each thread processes one or more blocks using a loop
    // This handles cases where num_blocks_per_row > blockDim.x
    for (int block_idx = threadIdx.x; block_idx < num_blocks_per_row; block_idx += blockDim.x) {
        // Each thread processes one block of 16 elements
        int col_start = block_idx * QUANT_BLOCK_SIZE;
        float block_data[QUANT_BLOCK_SIZE];
        
        // Load and convert to float32 (keep ORIGINAL values, do NOT multiply by global_scale)
        #pragma unroll
        for (int i = 0; i < QUANT_BLOCK_SIZE; i++) {
            int col = col_start + i;
            if (col < N) {
                block_data[i] = __bfloat162float(input[row * N + col]);
            } else {
                block_data[i] = 0.0f;
            }
        }
        
        // Quantize block
        // Each block produces 2 uint32_t values (8 e2m1 values each)
        // Weight output: LINEAR LAYOUT
        int quant_idx = row * (N / 8) + block_idx * 2;  // 2 uint32_t per block
        
        // Scale output: LINEAR LAYOUT
        // Each block produces 1 scale value (float8_e4m3fn)
        int scale_idx = row * num_blocks_per_row + block_idx;
        uint8_t* scale_mem_ptr = output_scales + scale_idx;

        quantize_block(
            block_data,
            output_quantized + quant_idx,
            scale_mem_ptr,
            global_scale);
    }
}

// Wrapper function called from Python
void nvfp4_quantize_cuda(
    torch::Tensor input,
    torch::Tensor output_quantized,
    torch::Tensor output_scales,
    double global_scale) {
    
    // Convert double to float for CUDA kernel
    float global_scale_float = static_cast<float>(global_scale);

    // Check inputs - ALL tensors must be on CUDA
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device, but got CPU");
    TORCH_CHECK(output_quantized.is_cuda(), "output_quantized tensor must be on CUDA device, but got CPU");
    TORCH_CHECK(output_scales.is_cuda(), "output_scales tensor must be on CUDA device, but got CPU");

    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(output_quantized.dim() == 2, "output_quantized must be 2D tensor");
    
    int M = input.size(0);
    int N = input.size(1);
    
    TORCH_CHECK(N % QUANT_BLOCK_SIZE == 0, 
                "N dimension must be multiple of ", QUANT_BLOCK_SIZE);
    
    // Check output shapes
    TORCH_CHECK(output_quantized.dtype() == torch::kUInt32,
                "Output quantized must be uint32");
    TORCH_CHECK(output_quantized.size(0) == M && output_quantized.size(1) == N / 8,
                "Output quantized shape mismatch: expected [", M, ", ", N / 8, "], got [", 
                output_quantized.size(0), ", ", output_quantized.size(1), "]");
    
    // Check output_scales shape (LINEAR LAYOUT)
    // Scales are stored as uint8 (float8_e4m3fn) in linear layout
    TORCH_CHECK(output_scales.dtype() == torch::kUInt8,
                "Output scales must be uint8 (for linear layout)");
    
    // Calculate expected linear layout shape
    int scale_n = N / QUANT_BLOCK_SIZE;  // Number of scales per row
    int expected_scales_rows = M;
    int expected_scales_cols = scale_n;
    
    TORCH_CHECK(output_scales.size(0) == expected_scales_rows && output_scales.size(1) == expected_scales_cols,
                "Output scales shape mismatch: expected linear layout [", expected_scales_rows, ", ", expected_scales_cols, "], got [",
                output_scales.size(0), ", ", output_scales.size(1), "]");
    // Ensure tensors are contiguous
    TORCH_CHECK(output_quantized.is_contiguous(), "Output quantized must be contiguous");
    TORCH_CHECK(output_scales.is_contiguous(), "Output scales must be contiguous");
    
    // Setup CUDA
    at::cuda::CUDAGuard device_guard(input.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
    
    // Launch kernel
    // Each thread processes one block of 16 elements
    int num_blocks_per_row = N / QUANT_BLOCK_SIZE;
    
    // Use a loop in the kernel to handle cases where num_blocks_per_row > block_size
    // Set block_size to a reasonable value (256 threads per block)
    int block_size_val = 256;
    dim3 block_size(block_size_val);
    dim3 grid_size(M);
    
    // Use reinterpret_cast instead of data_ptr<T>() to avoid linking issues
    // output_scales is uint8 tensor (linear layout)
    nvfp4_quantize_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<uint32_t*>(output_quantized.data_ptr()),
        reinterpret_cast<uint8_t*>(output_scales.data_ptr()),
        M,
        N,
        global_scale_float);
    
    // Check for errors immediately after launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    // Synchronize to catch any kernel execution errors
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
    }
}

