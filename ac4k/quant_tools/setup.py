"""Setup script for Wan2.2 Offline Quantization Package"""

from setuptools import setup

setup(
    name='wan-offline-quant',
    version='0.1.0',
    description='Offline NVFP4 Quantization for Wan2.2 DiT Models',
    author='AC4K',
    packages=['wan_quant'],
    package_dir={'': 'python'},
    python_requires='>=3.10',
    install_requires=[],
    zip_safe=False,
)

