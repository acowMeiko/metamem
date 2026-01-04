#!/bin/bash
# ==================== BBH 数据集 DPO 数据生成 ====================
# 功能：处理所有 BBH 任务（27个），生成统一的 DPO 数据
# 使用：bash run_bbh.sh
# =================================================================

# 不使用 set -e，允许单个任务失败后继续

# ==================== GPU 配置 ====================
export CUDA_VISIBLE_DEVICES="2,3"  # 使用 GPU 0,1 (BBH专用)
export BATCH_SIZE=128              # vLLM 批处理大小（2张A800优化）
export MAX_WORKERS=30              # API 并发数（2张A800优化）
export VLLM_WORKER_MULTIPROC_METHOD="spawn"  # 多进程方法

# ==================== 数据集配置 ====================
export DATASET_NAME="bbh"
DATASET_DIR="dataset/bbh"
OUTPUT_DIR="output/bbh"  # BBH数据保存在output/bbh文件夹中
LOG_DIR="logs"

# 创建输出和日志目录
[ -f "${OUTPUT_DIR}" ] && rm -f "${OUTPUT_DIR}"
[ -f "${LOG_DIR}" ] && rm -f "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true
mkdir -p "${LOG_DIR}" 2>/dev/null || true

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/bbh_${TIMESTAMP}.log"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "BBH 数据集 DPO 数据生成 (每个任务独立保存)" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "批次大小: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "并发数: ${MAX_WORKERS}" | tee -a "${LOG_FILE}"
echo "输出目录: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# BBH 任务列表（27个）
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

echo "BBH 共有 ${#BBH_TASKS[@]} 个任务 (每个任务独立保存)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 处理每个 BBH 任务
success_count=0
fail_count=0
total_lines=0

for task in "${BBH_TASKS[@]}"; do
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "处理任务 [$((success_count + fail_count + 1))/${#BBH_TASKS[@]}]: ${task}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    
    # 为每个任务设置独立的输出文件
    export DPO_OUTPUT_FILE="${OUTPUT_DIR}/dpo_${task}.jsonl"
    
    # 设置数据集路径
    export DATASET_PATH="${DATASET_DIR}/${task}.json"
    
    # 检查文件是否存在
    if [ ! -f "${DATASET_PATH}" ]; then
        echo "⚠️  警告: 文件不存在 - ${DATASET_PATH}，跳过" | tee -a "${LOG_FILE}"
        ((fail_count++))
        continue
    fi
    
    # 运行数据生成（每个任务独立输出文件）
    echo "开始生成 DPO 数据..." | tee -a "${LOG_FILE}"
    echo "输出文件: ${DPO_OUTPUT_FILE}" | tee -a "${LOG_FILE}"
    
    # 使用 || true 确保即使失败也继续
    if python stage_first.py 2>&1 | tee -a "${LOG_FILE}" || true; then
        # 检查输出文件是否存在且有内容
        if [ -f "${DPO_OUTPUT_FILE}" ] && [ -s "${DPO_OUTPUT_FILE}" ]; then
            task_lines=$(wc -l < "${DPO_OUTPUT_FILE}")
            task_size=$(du -h "${DPO_OUTPUT_FILE}" | cut -f1)
            echo "✅ 完成: ${task} (${task_lines} 条, ${task_size})" | tee -a "${LOG_FILE}"
            ((success_count++))
            total_lines=$((total_lines + task_lines))
        else
            echo "⚠️  可能失败: ${task} (输出文件为空或不存在)" | tee -a "${LOG_FILE}"
            ((fail_count++))
        fi
    else
        echo "❌ 失败: ${task}" | tee -a "${LOG_FILE}"
        ((fail_count++))
    fi
    
    echo "" | tee -a "${LOG_FILE}"
done

# 统计结果
echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "BBH 数据集处理完成" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "成功: ${success_count}/${#BBH_TASKS[@]}" | tee -a "${LOG_FILE}"
echo "失败: ${fail_count}/${#BBH_TASKS[@]}" | tee -a "${LOG_FILE}"
echo "总数据量: ${total_lines} 条" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "生成的文件列表:" | tee -a "${LOG_FILE}"
ls -lh "${OUTPUT_DIR}"/dpo_*.jsonl 2>/dev/null | awk '{print $9, $5}' | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "🎉 完成！" | tee -a "${LOG_FILE}"
