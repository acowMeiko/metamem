#!/bin/bash
# 检查可用的数据集文件

echo "==================== 检查数据集文件 ===================="

echo ""
echo "1. 检查 GSM8K:"
if [ -f "dataset/gsm8k/test.jsonl" ]; then
    echo "  ✅ dataset/gsm8k/test.jsonl 存在"
    wc -l dataset/gsm8k/test.jsonl
else
    echo "  ❌ dataset/gsm8k/test.jsonl 不存在"
fi

echo ""
echo "2. 检查 MATH:"
if [ -f "dataset/math/test.jsonl" ]; then
    echo "  ✅ dataset/math/test.jsonl 存在"
    wc -l dataset/math/test.jsonl
else
    echo "  ❌ dataset/math/test.jsonl 不存在"
fi

echo ""
echo "3. 检查 SVAMP:"
if [ -f "dataset/svamp/test.json" ]; then
    echo "  ✅ dataset/svamp/test.json 存在"
else
    echo "  ❌ dataset/svamp/test.json 不存在"
fi

echo ""
echo "4. 检查 BBH 目录:"
if [ -d "dataset/bbh" ]; then
    echo "  ✅ dataset/bbh 目录存在"
    echo "  文件列表:"
    ls -1 dataset/bbh/*.json 2>/dev/null | head -10 || echo "  ⚠️  没有找到 .json 文件"
else
    echo "  ❌ dataset/bbh 目录不存在"
    echo "  尝试查找BBH相关目录:"
    find dataset -type d -name "*bbh*" 2>/dev/null || echo "  未找到"
fi

echo ""
echo "5. 检查 MMLU 目录:"
if [ -d "dataset/mmlu" ]; then
    echo "  ✅ dataset/mmlu 目录存在"
    echo "  文件列表:"
    ls -1 dataset/mmlu/*.json 2>/dev/null | head -10 || echo "  ⚠️  没有找到 .json 文件"
else
    echo "  ❌ dataset/mmlu 目录不存在"
    echo "  尝试查找MMLU相关目录:"
    find dataset -type d -name "*mmlu*" 2>/dev/null || echo "  未找到"
fi

echo ""
echo "==================== 所有 dataset 下的目录 ===================="
ls -la dataset/ 2>/dev/null || echo "dataset 目录不存在"

echo ""
echo "==================== 检查完成 ===================="
