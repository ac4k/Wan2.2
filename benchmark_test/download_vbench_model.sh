#!/bin/bash
# 下载VBench所需的模型文件

CACHE_DIR="${HOME}/.cache/vbench"
MODEL_URL="https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
MODEL_DIR="${CACHE_DIR}/ViCLIP"
MODEL_PATH="${MODEL_DIR}/ViClip-InternVid-10M-FLT.pth"

echo "下载VBench模型文件..."
echo "目标路径: ${MODEL_PATH}"

# 创建目录
mkdir -p "${MODEL_DIR}"

# 检查是否已存在
if [ -f "${MODEL_PATH}" ]; then
    echo "模型文件已存在: ${MODEL_PATH}"
    exit 0
fi

# 尝试使用curl下载（如果没有wget）
if command -v wget &> /dev/null; then
    echo "使用wget下载..."
    wget "${MODEL_URL}" -O "${MODEL_PATH}"
elif command -v curl &> /dev/null; then
    echo "使用curl下载..."
    curl -L "${MODEL_URL}" -o "${MODEL_PATH}"
else
    echo "错误: 未找到wget或curl，请先安装其中一个工具"
    echo "安装wget: sudo apt-get install wget"
    echo "或安装curl: sudo apt-get install curl"
    exit 1
fi

if [ -f "${MODEL_PATH}" ]; then
    echo "下载成功: ${MODEL_PATH}"
else
    echo "下载失败，请检查网络连接"
    exit 1
fi

