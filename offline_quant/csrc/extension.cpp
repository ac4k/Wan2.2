/*
 * PyTorch Extension Binding for NVFP4 Quantization
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <Python.h>

void nvfp4_quantize_cuda(
    torch::Tensor input,
    torch::Tensor output_quantized,
    torch::Tensor output_scales,
    double global_scale);

// Register the operator
TORCH_LIBRARY_FRAGMENT(wan_quant_kernel, m) {
    m.def("nvfp4_quantize_cuda(Tensor input, Tensor! output_quantized, Tensor! output_scales, float global_scale) -> ()");
}

TORCH_LIBRARY_IMPL(wan_quant_kernel, CUDA, m) {
    m.impl("nvfp4_quantize_cuda", nvfp4_quantize_cuda);
}

PyMODINIT_FUNC PyInit_wan_quant_kernel(void) {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "wan_quant_kernel",
        nullptr,
        0,
        nullptr
    };
    return PyModule_Create(&module);
}

