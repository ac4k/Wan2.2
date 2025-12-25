"""Setup script for Wan2.2 Offline Quantization Package"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension
import os

# Get CUDA paths
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

# Get PyTorch library directory for runtime linking
torch_lib_dir = os.path.dirname(torch.__file__) + '/lib'

# PyTorch CUDAExtension should automatically link against PyTorch libraries
# But we need to ensure libtorch.so and libtorch_cuda.so are linked
# BuildExtension will add most libraries, but we explicitly add the missing ones
ext_modules = [
    cpp_extension.CUDAExtension(
        name='wan_quant_kernel',
        sources=[
            'csrc/extension.cpp',
            'csrc/kernels/nvfp4_quant.cu',
        ],
        include_dirs=[
            'csrc',
        ],
        library_dirs=[torch_lib_dir] if os.path.exists(torch_lib_dir) else [],
        libraries=['torch', 'torch_cuda'] if os.path.exists(torch_lib_dir) else [],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math',
                '-gencode', 'arch=compute_120a,code=sm_120a',
            ]
        },
        # BuildExtension automatically adds PyTorch libraries (c10, c10_cuda, etc.)
        # We explicitly add libtorch.so and libtorch_cuda.so for data_ptr symbols
        extra_link_args=[
            f'-Wl,-rpath,{torch_lib_dir}' if os.path.exists(torch_lib_dir) else '',
        ] if os.path.exists(torch_lib_dir) else []
    )
]

setup(
    name='wan-offline-quant',
    version='0.1.0',
    description='Offline NVFP4 Quantization for Wan2.2 DiT Models',
    author='AC4K',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['wan_quant'],
    package_dir={'': 'python'},
    python_requires='>=3.10',
    zip_safe=False,
)

