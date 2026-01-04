#!/bin/bash
# ==================== 全部数据集批量处理 ====================
# 功能：按顺序运行所有5个数据集的处理脚本
# 使用：bash run_all_datasets.sh
# =============================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "批量处理所有数据集"
echo "=========================================="
echo "数据集: BBH, GSM8K, MATH, MMLU, SVAMP"
echo "GPU: 3,4 (2张A800)"
echo "=========================================="
echo ""

start_time=$(date +%s)

# 1. BBH 数据集
echo "==================== [1/5] BBH 数据集 ===================="
bash run_bbh.sh
echo ""

# 2. GSM8K 数据集
echo "==================== [2/5] GSM8K 数据集 ===================="
bash run_gsm8k.sh
echo ""

# 3. MATH 数据集
echo "==================== [3/5] MATH 数据集 ===================="
bash run_math.sh
echo ""

# 4. MMLU 数据集
echo "==================== [4/5] MMLU 数据集 ===================="
bash run_mmlu.sh
echo ""

# 5. SVAMP 数据集
echo "==================== [5/5] SVAMP 数据集 ===================="
bash run_svamp.sh
echo ""

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

# 汇总统计
echo ""
echo "=========================================="
echo "全部处理完成！"
echo "=========================================="
echo ""
echo "输出文件："
echo "  - output/dpo_bbh_all.jsonl"
echo "  - output/dpo_gsm8k.jsonl"
echo "  - output/dpo_math.jsonl"
echo "  - output/dpo_mmlu_all.jsonl"
echo "  - output/dpo_svamp.jsonl"
echo ""

# 统计总数据量
if command -v wc &> /dev/null; then
    total_lines=$(cat output/dpo_*.jsonl 2>/dev/null | wc -l || echo "0")
    echo "总数据量: ${total_lines} 条"
fi

echo ""
echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
echo ""
echo "日志文件保存在: logs/"
echo "=========================================="
echo "🎉 全部完成！"
