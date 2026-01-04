#!/bin/bash
# ==================== 并行运行所有数据集 ====================
# 功能：同时处理多个数据集，使用不同的GPU避免冲突
# GPU分配：
#   - BBH:   GPU 0,1
#   - MMLU:  GPU 2,3
#   - GSM8K: GPU 4,5
#   - MATH:  GPU 6,7
#   - SVAMP: 单独运行（或使用剩余GPU）
# =================================================================

echo "=========================================="
echo "开始并行处理所有数据集"
echo "=========================================="
echo ""

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "检测到 ${GPU_COUNT} 个GPU"
echo ""

if [ ${GPU_COUNT} -lt 8 ]; then
    echo "⚠️  警告: 需要至少8个GPU才能并行运行所有数据集"
    echo "当前GPU数量: ${GPU_COUNT}"
    echo ""
    echo "建议方案："
    echo "1. 顺序运行: bash run_all_datasets.sh"
    echo "2. 选择性并行运行（见下方）"
    echo ""
fi

echo "GPU分配方案："
echo "  BBH   -> GPU 0,1"
echo "  GSM8K -> GPU 2,3"
echo "  MATH  -> GPU 4,5"
echo "  MMLU  -> GPU 6,7"
echo ""

# 询问用户确认
read -p "是否继续并行运行? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "启动并行任务..."
echo ""

# 后台启动各个数据集处理
nohup bash run_bbh.sh > run_bbh_parallel.out 2>&1 &
BBH_PID=$!
echo "✓ BBH 已启动 (PID: ${BBH_PID}, GPU: 0,1)"

nohup bash run_gsm8k.sh > run_gsm8k_parallel.out 2>&1 &
GSM8K_PID=$!
echo "✓ GSM8K 已启动 (PID: ${GSM8K_PID}, GPU: 2,3)"

nohup bash run_math.sh > run_math_parallel.out 2>&1 &
MATH_PID=$!
echo "✓ MATH 已启动 (PID: ${MATH_PID}, GPU: 4,5)"

nohup bash run_mmlu.sh > run_mmlu_parallel.out 2>&1 &
MMLU_PID=$!
echo "✓ MMLU 已启动 (PID: ${MMLU_PID}, GPU: 6,7)"

echo ""
echo "=========================================="
echo "所有任务已在后台启动"
echo "=========================================="
echo ""
echo "进程ID:"
echo "  BBH:   ${BBH_PID}"
echo "  MMLU:  ${MMLU_PID}"
echo "  GSM8K: ${GSM8K_PID}"
echo "  MATH:  ${MATH_PID}"
echo ""
echo "监控命令:"
echo "  查看所有进程: ps aux | grep 'run_.*\.sh'"
echo "  查看GPU使用: nvidia-smi"
echo "  查看BBH日志: tail -f run_bbh_parallel.out"
echo "  查看MMLU日志: tail -f run_mmlu_parallel.out"
echo "  查看GSM8K日志: tail -f run_gsm8k_parallel.out"
echo "  查看MATH日志: tail -f run_math_parallel.out"
echo ""
echo "停止所有任务:"
echo "  kill ${BBH_PID} ${MMLU_PID} ${GSM8K_PID} ${MATH_PID}"
echo ""
echo "🎉 并行处理已开始！"
