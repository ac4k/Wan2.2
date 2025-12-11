#!/bin/bash
# Batch run script: Split 50 prompts into 4 parts and run in parallel on 4 GPUs

PROMPTS_FILE="./selected_prompts_50.txt"
NUM_GPUS=4
SCRIPT="./generate_and_evaluate.sh"

# Calculate total number of prompts
TOTAL=$(wc -l < "$PROMPTS_FILE")
PROMPTS_PER_GPU=$((TOTAL / NUM_GPUS))
REMAINDER=$((TOTAL % NUM_GPUS))

echo "Total prompts: $TOTAL"
echo "Each GPU will process approximately: $PROMPTS_PER_GPU prompts"
echo ""

# Start background tasks for each GPU
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate start and end indices for this GPU
    if [ $gpu_id -lt $REMAINDER ]; then
        START=$((gpu_id * (PROMPTS_PER_GPU + 1)))
        END=$((START + PROMPTS_PER_GPU))
    else
        START=$((gpu_id * PROMPTS_PER_GPU + REMAINDER))
        END=$((START + PROMPTS_PER_GPU - 1))
    fi

    echo "GPU $gpu_id: Processing indices $START to $END"

    # Run in background, set GPU and prompt index
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export PROMPTS_FILE="$PROMPTS_FILE"

        for idx in $(seq $START $END); do
            if [ $idx -lt $TOTAL ]; then
                export PROMPT_INDEX=$idx
                echo "[GPU $gpu_id] Processing index $idx"
                bash "$SCRIPT"
            fi
        done
    ) > "gpu${gpu_id}.log" 2>&1 &

    echo "GPU $gpu_id process ID: $!"
done

echo ""
echo "All GPU tasks started, log files: gpu0.log, gpu1.log, gpu2.log, gpu3.log"
echo "Waiting for all tasks to complete..."

wait

echo "All tasks completed!"

