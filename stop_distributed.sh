#!/bin/bash
# ============================================================================
# 停止所有分布式训练进程
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${PROJECT_ROOT}/outputs_distributed"
PID_FILE="${OUTPUT_BASE}/pids.txt"

echo "正在停止所有分布式训练进程..."

if [ -f "$PID_FILE" ]; then
    while read pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "  停止 PID: $pid"
            kill "$pid" 2>/dev/null
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# 确保清理干净
pkill -f "train_pretrain_france" 2>/dev/null || true

sleep 2
echo "✅ 所有进程已停止"
