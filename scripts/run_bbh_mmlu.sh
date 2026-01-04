#!/bin/bash
# ==================== BBH & MMLU DPO数据生成流程 ====================
# 功能：批量处理 BBH 和 MMLU 数据集，生成 DPO 训练数据
# 作者：AI Assistant
# 日期：2025-12-23
# =================================================================

set -e  # 遇到错误立即退出

# ==================== 配置区 ====================
# GPU配置（根据实际情况修改）
export CUDA_VISIBLE_DEVICES="3,4"

# 批处理配置
export BATCH_SIZE=64
export MAX_WORKERS=20

# 输出目录
OUTPUT_DIR="output"
BBH_OUTPUT_DIR="${OUTPUT_DIR}/bbh"
MMLU_OUTPUT_DIR="${OUTPUT_DIR}/mmlu"

# 创建输出目录
mkdir -p "${BBH_OUTPUT_DIR}"
mkdir -p "${MMLU_OUTPUT_DIR}"

# 日志目录
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "BBH & MMLU DPO 数据生成流程"
echo "=========================================="
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "批次大小: ${BATCH_SIZE}"
echo "并发数: ${MAX_WORKERS}"
echo "=========================================="

# ==================== BBH 数据集处理 ====================
echo ""
echo "==================== 第1部分：处理 BBH 数据集 ===================="
echo ""

# BBH 任务列表（27个任务）
BBH_TASKS=(
    "boolean_expressions"
    "causal_judgement"
    "date_understanding"
    "disambiguation_qa"
    "dyck_languages"
    "formal_fallacies"
    "geometric_shapes"
    "hyperbaton"
    "logical_deduction_five_objects"
    "logical_deduction_seven_objects"
    "logical_deduction_three_objects"
    "movie_recommendation"
    "multistep_arithmetic_two"
    "navigate"
    "object_counting"
    "penguins_in_a_table"
    "reasoning_about_colored_objects"
    "ruin_names"
    "salient_translation_error_detection"
    "snarks"
    "sports_understanding"
    "temporal_sequences"
    "tracking_shuffled_objects_five_objects"
    "tracking_shuffled_objects_seven_objects"
    "tracking_shuffled_objects_three_objects"
    "web_of_lies"
    "word_sorting"
)

echo "BBH 共有 ${#BBH_TASKS[@]} 个任务"
echo ""

# 处理每个 BBH 任务
for task in "${BBH_TASKS[@]}"; do
    echo "----------------------------------------"
    echo "处理 BBH 任务: ${task}"
    echo "----------------------------------------"
    
    # 设置输入输出路径
    export DATASET_NAME="bbh"
    export DATASET_PATH="dataset/bbh/${task}.json"
    
    # 检查文件是否存在
    if [ ! -f "${DATASET_PATH}" ]; then
        echo "⚠️  警告: 文件不存在 - ${DATASET_PATH}，跳过"
        continue
    fi
    
    # 设置输出文件
    export DPO_OUTPUT_FILE="${BBH_OUTPUT_DIR}/dpo_${task}.jsonl"
    
    # 运行数据生成
    echo "开始生成 DPO 数据..."
    python stage_first.py 2>&1 | tee "${LOG_DIR}/bbh_${task}_$(date +%Y%m%d_%H%M%S).log"
    
    # 移动生成的文件
    if [ -f "output/dpo_final.jsonl" ]; then
        mv "output/dpo_final.jsonl" "${DPO_OUTPUT_FILE}"
        echo "✅ 完成: ${task} -> ${DPO_OUTPUT_FILE}"
        
        # 统计数据量
        line_count=$(wc -l < "${DPO_OUTPUT_FILE}")
        echo "   数据量: ${line_count} 条"
    else
        echo "❌ 错误: 未生成输出文件"
    fi
    
    echo ""
done

echo ""
echo "✅ BBH 数据集处理完成！"
echo "输出目录: ${BBH_OUTPUT_DIR}"
ls -lh "${BBH_OUTPUT_DIR}"
echo ""

# ==================== MMLU 数据集处理 ====================
echo ""
echo "==================== 第2部分：处理 MMLU 数据集 ===================="
echo ""

# MMLU 数据文件列表
MMLU_FILES=(
    "auxiliary_train"
    "dev"
    "test"
    "validation"
)

echo "MMLU 共有 ${#MMLU_FILES[@]} 个数据文件"
echo ""

# 处理每个 MMLU 文件
for file in "${MMLU_FILES[@]}"; do
    echo "----------------------------------------"
    echo "处理 MMLU 文件: ${file}"
    echo "----------------------------------------"
    
    # 设置输入输出路径
    export DATASET_NAME="mmlu"
    export DATASET_PATH="dataset/mmlu/${file}.json"
    
    # 检查文件是否存在
    if [ ! -f "${DATASET_PATH}" ]; then
        echo "⚠️  警告: 文件不存在 - ${DATASET_PATH}，跳过"
        continue
    fi
    
    # 设置输出文件
    export DPO_OUTPUT_FILE="${MMLU_OUTPUT_DIR}/dpo_${file}.jsonl"
    
    # 运行数据生成
    echo "开始生成 DPO 数据..."
    python stage_first.py 2>&1 | tee "${LOG_DIR}/mmlu_${file}_$(date +%Y%m%d_%H%M%S).log"
    
    # 移动生成的文件
    if [ -f "output/dpo_final.jsonl" ]; then
        mv "output/dpo_final.jsonl" "${DPO_OUTPUT_FILE}"
        echo "✅ 完成: ${file} -> ${DPO_OUTPUT_FILE}"
        
        # 统计数据量
        line_count=$(wc -l < "${DPO_OUTPUT_FILE}")
        echo "   数据量: ${line_count} 条"
    else
        echo "❌ 错误: 未生成输出文件"
    fi
    
    echo ""
done

echo ""
echo "✅ MMLU 数据集处理完成！"
echo "输出目录: ${MMLU_OUTPUT_DIR}"
ls -lh "${MMLU_OUTPUT_DIR}"
echo ""

# ==================== 汇总统计 ====================
echo ""
echo "=========================================="
echo "数据生成汇总"
echo "=========================================="

echo ""
echo "BBH 数据集:"
if [ -d "${BBH_OUTPUT_DIR}" ]; then
    bbh_files=$(ls -1 "${BBH_OUTPUT_DIR}"/*.jsonl 2>/dev/null | wc -l)
    bbh_total=$(cat "${BBH_OUTPUT_DIR}"/*.jsonl 2>/dev/null | wc -l)
    echo "  文件数: ${bbh_files}"
    echo "  总数据量: ${bbh_total} 条"
else
    echo "  输出目录不存在"
fi

echo ""
echo "MMLU 数据集:"
if [ -d "${MMLU_OUTPUT_DIR}" ]; then
    mmlu_files=$(ls -1 "${MMLU_OUTPUT_DIR}"/*.jsonl 2>/dev/null | wc -l)
    mmlu_total=$(cat "${MMLU_OUTPUT_DIR}"/*.jsonl 2>/dev/null | wc -l)
    echo "  文件数: ${mmlu_files}"
    echo "  总数据量: ${mmlu_total} 条"
else
    echo "  输出目录不存在"
fi

echo ""
echo "=========================================="
echo "🎉 全部处理完成！"
echo "=========================================="
