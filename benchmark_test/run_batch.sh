#!/bin/bash
# 批量运行脚本：将50条prompts分成4份，在4个GPU上并行运行

PROMPTS_FILE="./selected_prompts_50.txt"
NUM_GPUS=4
SCRIPT="./generate_and_evaluate.sh"

# 计算总prompts数
TOTAL=$(wc -l < "$PROMPTS_FILE")
PROMPTS_PER_GPU=$((TOTAL / NUM_GPUS))
REMAINDER=$((TOTAL % NUM_GPUS))

echo "总prompts数: $TOTAL"
echo "每个GPU约处理: $PROMPTS_PER_GPU 条"
echo ""

# 为每个GPU启动后台任务
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # 计算该GPU的起始和结束索引
    if [ $gpu_id -lt $REMAINDER ]; then
        START=$((gpu_id * (PROMPTS_PER_GPU + 1)))
        END=$((START + PROMPTS_PER_GPU))
    else
        START=$((gpu_id * PROMPTS_PER_GPU + REMAINDER))
        END=$((START + PROMPTS_PER_GPU - 1))
    fi
    
    echo "GPU $gpu_id: 处理索引 $START 到 $END"
    
    # 在后台运行，设置GPU和prompt索引
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export PROMPTS_FILE="$PROMPTS_FILE"
        
        for idx in $(seq $START $END); do
            if [ $idx -lt $TOTAL ]; then
                export PROMPT_INDEX=$idx
                echo "[GPU $gpu_id] 处理索引 $idx"
                bash "$SCRIPT"
            fi
        done
    ) > "gpu${gpu_id}.log" 2>&1 &
    
    echo "GPU $gpu_id 进程ID: $!"
done

echo ""
echo "所有GPU任务已启动，日志文件: gpu0.log, gpu1.log, gpu2.log, gpu3.log"
echo "等待所有任务完成..."

wait

echo "所有任务完成！"

