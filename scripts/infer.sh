#!/bin/bash

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=4,5

# 设置输入文件为 level1.jsonl
export TEST_FILE="data/dpo_by_level/level1.jsonl"

# 运行推理
echo "开始推理 level1.jsonl..."
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "输入文件: $TEST_FILE"
echo "日志文件: logs/stage_infer_level1.log"
echo "========================================"

nohup python stage_infer.py > logs/stage_infer_level1.log 2>&1 &

# 保存进程 ID
PID=$!
echo "推理进程已启动，PID: $PID"
echo "查看实时日志: tail -f logs/stage_infer_level1.log"
echo "查看进程状态: ps -p $PID"

