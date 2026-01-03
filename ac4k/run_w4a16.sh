#!/bin/bash
export USE_AC4K_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 generate_w4a16.py  --task t2v-A14B \
  --size 1280*720 \
  --quantized_ckpt_dir /data/models/wan/ac4k_wan_t2v_nvfp4 \
  --offload_model True \
  --convert_model_dtype \
  --enable_hooks \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
