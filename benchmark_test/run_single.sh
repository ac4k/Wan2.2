#!/bin/bash
# Single GPU run script: Process all prompts sequentially

PROMPTS_FILE="./selected_prompts_50.txt"
SCRIPT="./generate_and_evaluate.sh"
LOG_FILE="./run_single.log"

# Signal handler function: cleanup processes
cleanup() {
    echo ""
    echo "Received interrupt signal, cleaning up..." | tee -a "$LOG_FILE"
    # Terminate python processes
    pkill -TERM -f "generate_w4a16.py" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "generate_w4a16.py" 2>/dev/null || true
    echo "Cleanup completed" | tee -a "$LOG_FILE"
    exit 130
}

# Register signal handler
trap cleanup SIGINT SIGTERM

# Clear old log, start fresh
> "$LOG_FILE"

# Calculate total number of prompts
TOTAL=$(wc -l < "$PROMPTS_FILE")

{
    echo "Total prompts: $TOTAL"
    echo "Starting sequential processing..."
    echo "Press Ctrl+C to interrupt"
    echo ""
} | tee "$LOG_FILE"

# Set environment variables
export PROMPTS_FILE="$PROMPTS_FILE"

# Process each prompt sequentially, redirect output to log
for idx in $(seq 0 $((TOTAL - 1))); do
    export PROMPT_INDEX=$idx
    {
        echo "========================================"
        echo "Processing index $idx / $((TOTAL - 1))"
        echo "========================================"
    } | tee -a "$LOG_FILE"

    bash "$SCRIPT" >> "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "Error: Failed to process index $idx" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "" | tee -a "$LOG_FILE"
done

echo "All tasks completed!" | tee -a "$LOG_FILE"

