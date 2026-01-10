#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 generate_nvfp4.py  --task i2v-A14B \
  --size 1280*720 \
  --quantized_ckpt_dir /data/models/wan/ac4k_new_i2v_nvfp4 \
  --offload_model True \
  --convert_model_dtype \
  --image ../examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
