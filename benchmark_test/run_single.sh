#!/bin/bash
# 单GPU运行脚本：顺序处理所有prompts

PROMPTS_FILE="./selected_prompts_50.txt"
SCRIPT="./generate_and_evaluate.sh"
LOG_FILE="./run_single.log"

# 信号处理函数：清理进程
cleanup() {
    echo ""
    echo "收到中断信号，正在清理..." | tee -a "$LOG_FILE"
    # 终止python进程
    pkill -TERM -f "generate_w4a16.py" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "generate_w4a16.py" 2>/dev/null || true
    echo "清理完成" | tee -a "$LOG_FILE"
    exit 130
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 清理旧日志，重新开始
> "$LOG_FILE"

# 计算总prompts数
TOTAL=$(wc -l < "$PROMPTS_FILE")

{
    echo "总prompts数: $TOTAL"
    echo "开始顺序处理..."
    echo "按 Ctrl+C 可以中断"
    echo ""
} | tee "$LOG_FILE"

# 设置环境变量
export PROMPTS_FILE="$PROMPTS_FILE"

# 顺序处理每个prompt，输出重定向到日志
for idx in $(seq 0 $((TOTAL - 1))); do
    export PROMPT_INDEX=$idx
    {
        echo "========================================"
        echo "处理索引 $idx / $((TOTAL - 1))"
        echo "========================================"
    } | tee -a "$LOG_FILE"
    
    bash "$SCRIPT" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "错误: 索引 $idx 处理失败" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

echo "所有任务完成！" | tee -a "$LOG_FILE"

