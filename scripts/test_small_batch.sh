#!/bin/bash
# ========================================
# 小批次测试脚本 (Bash)
# ========================================
# 从每个数据集提取前128条数据并运行测试

echo "========================================"
echo "🧪 小批次测试脚本"
echo "========================================"
echo ""
echo "说明:"
echo "  - 测试批次大小: 128 条"
echo "  - 快速验证端到端功能"
echo ""

# 测试批次大小
BATCH_SIZE=8

# 定义数据集
declare -A DATASETS=(
    ["bbh"]="dataset/bbh/boolean_expressions.json"
    ["svamp"]="dataset/svamp/test.json"
    ["gsm8k"]="dataset/gsm8k/test.jsonl"
    ["math"]="dataset/math/test.jsonl"
)

# 创建测试数据目录
TEST_DATA_DIR="data/test_small_batch"
mkdir -p "$TEST_DATA_DIR"

# 函数: 创建小测试数据集
create_test_dataset() {
    local dataset_name=$1
    local dataset_path=$2
    local output_path="$TEST_DATA_DIR/${dataset_name}_test_${BATCH_SIZE}.json"
    
    echo "----------------------------------------"
    echo "📦 处理: $dataset_name"
    echo "   原始路径: $dataset_path"
    
    if [ ! -f "$dataset_path" ]; then
        echo "   ⚠️  文件不存在，跳过"
        return 1
    fi
    
    # 使用 Python 加载、预处理并截取前128条
    python3 << EOF
import json
import sys
sys.path.insert(0, '.')

try:
    from stage_first import load_and_preprocess_dataset
    
    # 加载并预处理
    data = load_and_preprocess_dataset('$dataset_name', '$dataset_path')
    print(f'   原始数据: {len(data)} 条')
    
    # 截取前$BATCH_SIZE条
    small_data = data[:$BATCH_SIZE]
    print(f'   测试数据: {len(small_data)} 条')
    
    # 保存
    with open('$output_path', 'w', encoding='utf-8') as f:
        json.dump(small_data, f, ensure_ascii=False, indent=2)
    
    print(f'   ✅ 已保存到: $output_path')
    
except Exception as e:
    print(f'   ❌ 失败: {e}')
    sys.exit(1)
EOF
    
    return $?
}

# 函数: 运行测试
run_test() {
    local dataset_name=$1
    local test_data_path="$TEST_DATA_DIR/${dataset_name}_test_${BATCH_SIZE}.json"
    
    if [ ! -f "$test_data_path" ]; then
        echo "❌ 测试数据不存在: $test_data_path"
        return 1
    fi
    
    echo ""
    echo "========================================"
    echo "🚀 测试: $dataset_name"
    echo "========================================"
    
    # 使用 deepscaler 模式加载（因为已经是标准格式）
    export DATASET_NAME="deepscaler"
    export DATASET_PATH="$test_data_path"
    export BATCH_SIZE=8  # 降低批次大小以节省显存
    
    python3 stage_first.py
    
    local result=$?
    
    if [ $result -eq 0 ]; then
        echo "✅ $dataset_name 测试通过"
        return 0
    else
        echo "❌ $dataset_name 测试失败"
        return 1
    fi
}

# 主菜单
echo "========================================"
echo "选择测试模式:"
echo "========================================"
echo "  1. 快速测试 - 仅测试 BBH (最小)"
echo "  2. 创建所有测试数据集"
echo "  3. 测试所有数据集"
echo "  4. 测试指定数据集"
echo "  q. 退出"
echo "========================================"
read -p "请选择: " choice

case $choice in
    1)
        echo ""
        echo "🚀 快速测试模式 - BBH"
        create_test_dataset "bbh" "${DATASETS[bbh]}"
        if [ $? -eq 0 ]; then
            run_test "bbh"
        fi
        ;;
    
    2)
        echo ""
        echo "📦 创建所有测试数据集"
        for name in "${!DATASETS[@]}"; do
            create_test_dataset "$name" "${DATASETS[$name]}"
        done
        echo ""
        echo "✅ 测试数据集创建完成"
        echo "   位置: $TEST_DATA_DIR/"
        ;;
    
    3)
        echo ""
        echo "🚀 测试所有数据集"
        
        # 先创建所有测试数据集
        echo "步骤1: 创建测试数据集"
        for name in "${!DATASETS[@]}"; do
            create_test_dataset "$name" "${DATASETS[$name]}"
        done
        
        # 运行测试
        echo ""
        echo "步骤2: 运行测试"
        declare -A results
        for name in "${!DATASETS[@]}"; do
            run_test "$name"
            results[$name]=$?
        done
        
        # 打印总结
        echo ""
        echo "========================================"
        echo "📊 测试总结"
        echo "========================================"
        passed=0
        total=0
        for name in "${!results[@]}"; do
            total=$((total + 1))
            if [ ${results[$name]} -eq 0 ]; then
                echo "  $name: ✅ 通过"
                passed=$((passed + 1))
            else
                echo "  $name: ❌ 失败"
            fi
        done
        echo "----------------------------------------"
        echo "  总计: $passed/$total 通过"
        echo "========================================"
        ;;
    
    4)
        echo ""
        echo "可用的数据集:"
        i=1
        declare -a names
        for name in "${!DATASETS[@]}"; do
            echo "  $i. $name"
            names[$i]=$name
            i=$((i + 1))
        done
        
        read -p "请选择数字: " num
        
        if [ -n "${names[$num]}" ]; then
            selected="${names[$num]}"
            create_test_dataset "$selected" "${DATASETS[$selected]}"
            if [ $? -eq 0 ]; then
                run_test "$selected"
            fi
        else
            echo "❌ 无效的选择"
        fi
        ;;
    
    q)
        echo "👋 已退出"
        exit 0
        ;;
    
    *)
        echo "❌ 无效的选择: $choice"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成!"
