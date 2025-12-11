#!/bin/bash
# Download model files required by VBench

CACHE_DIR="${HOME}/.cache/vbench"
MODEL_URL="https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
MODEL_DIR="${CACHE_DIR}/ViCLIP"
MODEL_PATH="${MODEL_DIR}/ViClip-InternVid-10M-FLT.pth"

echo "Downloading VBench model files..."
echo "Target path: ${MODEL_PATH}"

# Create directory
mkdir -p "${MODEL_DIR}"

# Check if already exists
if [ -f "${MODEL_PATH}" ]; then
    echo "Model file already exists: ${MODEL_PATH}"
    exit 0
fi

# Try using curl to download (if wget is not available)
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget "${MODEL_URL}" -O "${MODEL_PATH}"
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L "${MODEL_URL}" -o "${MODEL_PATH}"
else
    echo "Error: wget or curl not found, please install one of them"
    echo "Install wget: sudo apt-get install wget"
    echo "Or install curl: sudo apt-get install curl"
    exit 1
fi

if [ -f "${MODEL_PATH}" ]; then
    echo "Download successful: ${MODEL_PATH}"
else
    echo "Download failed, please check network connection"
    exit 1
fi

