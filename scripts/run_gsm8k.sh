#!/bin/bash
# ==================== GSM8K 数据集 DPO 数据生成 ====================
# 功能：处理 GSM8K 测试集，生成 DPO 数据
# 使用：bash run_gsm8k.sh
# =================================================================

set -e  # 遇到错误立即退出

# ==================== GPU 配置 ====================
export CUDA_VISIBLE_DEVICES="2,3"  # 使用 GPU 2,3 (GSM8K专用，与其他数据集不冲突)
export BATCH_SIZE=128              # vLLM 批处理大小（2张A800优化）
export MAX_WORKERS=30              # API 并发数（2张A800优化）
export VLLM_WORKER_MULTIPROC_METHOD="spawn"  # 多进程方法

# ==================== 数据集配置 ====================
export DATASET_NAME="gsm8k"
export DATASET_PATH="dataset/gsm8k/test.jsonl"  # 使用测试集
OUTPUT_DIR="output/gsm8k"  # GSM8K数据保存在output/gsm8k文件夹中
LOG_DIR="logs"

# 创建输出和日志目录
[ -f "${OUTPUT_DIR}" ] && rm -f "${OUTPUT_DIR}"
[ -f "${LOG_DIR}" ] && rm -f "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true
mkdir -p "${LOG_DIR}" 2>/dev/null || true

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/gsm8k_${TIMESTAMP}.log"

# 输出文件
export DPO_OUTPUT_FILE="${OUTPUT_DIR}/dpo_gsm8k.jsonl"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "GSM8K 数据集 DPO 数据生成" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "数据集: ${DATASET_NAME}" | tee -a "${LOG_FILE}"
echo "数据路径: ${DATASET_PATH}" | tee -a "${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "批次大小: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "并发数: ${MAX_WORKERS}" | tee -a "${LOG_FILE}"
echo "输出文件: ${DPO_OUTPUT_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 检查文件是否存在
if [ ! -f "${DATASET_PATH}" ]; then
    echo "❌ 错误: 数据集文件不存在 - ${DATASET_PATH}" | tee -a "${LOG_FILE}"
    exit 1
fi

# 运行数据生成
echo "开始生成 DPO 数据..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

if python stage_first.py 2>&1 | tee -a "${LOG_FILE}"; then
    echo "" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    echo "✅ GSM8K 数据集处理完成" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    
    if [ -f "${DPO_OUTPUT_FILE}" ]; then
        line_count=$(wc -l < "${DPO_OUTPUT_FILE}")
        file_size=$(du -h "${DPO_OUTPUT_FILE}" | cut -f1)
        echo "输出文件: ${DPO_OUTPUT_FILE}" | tee -a "${LOG_FILE}"
        echo "数据量: ${line_count} 条" | tee -a "${LOG_FILE}"
        echo "文件大小: ${file_size}" | tee -a "${LOG_FILE}"
    else
        echo "⚠️  警告: 未生成输出文件" | tee -a "${LOG_FILE}"
    fi
    
    echo "" | tee -a "${LOG_FILE}"
    echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "🎉 完成！" | tee -a "${LOG_FILE}"
else
    echo "" | tee -a "${LOG_FILE}"
    echo "❌ 处理失败" | tee -a "${LOG_FILE}"
    echo "请查看日志: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi
