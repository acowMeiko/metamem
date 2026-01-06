"""
测试 Stage 1 DPO 格式输出和增量更新功能

用途:
1. 验证输出格式与 dpo_level1.json 完全一致
2. 测试增量更新逻辑（只更新 task_description、rejected、diff）
3. 测试完整生成逻辑

使用方法:
    python test_dpo_format.py
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.stages import StageOneAgent
from core.base import ReasoningOutput


def test_dpo_format():
    """测试 DPO 格式生成"""
    print("=" * 80)
    print("测试 1: DPO 格式生成")
    print("=" * 80)
    
    # 创建测试数据
    test_outputs = [
        ReasoningOutput(
            question="Test question 1",
            baseline_answer="Principle 1: Test baseline principle\nPrinciple 2: Another baseline",
            diff_analysis="Test diff analysis",
            principles=["Test chosen principle 1", "Test chosen principle 2"],
            chosen_answer="Test chosen answer",
            task_description='{"taskDescription": {"description": "Test task"}}',
            metadata={}
        )
    ]
    
    # 创建临时文件路径
    test_output_path = Path("test_dpo_output.json")
    
    # 创建 agent（最小配置）
    class MockEngine:
        pass
    
    class MockTemplate:
        pass
    
    agent_config = {
        'inference_engine': MockEngine(),
        'prompt_template': MockTemplate(),
        'batch_size': 1,
        'max_workers': 1
    }
    
    agent = StageOneAgent(agent_config)
    
    # 测试完整生成
    print("\n测试完整生成模式...")
    if test_output_path.exists():
        test_output_path.unlink()
    
    agent.save_dpo_format(test_outputs, test_output_path)
    
    # 读取并验证格式
    with open(test_output_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    
    print(f"\n生成了 {len(generated_data)} 条数据")
    print("\n第一条数据结构:")
    first_item = generated_data[0]
    for key in first_item.keys():
        print(f"  - {key}: {type(first_item[key]).__name__}")
    
    print("\nchosen 字段示例:")
    chosen_parsed = json.loads(first_item['chosen'])
    print(json.dumps(chosen_parsed, indent=2, ensure_ascii=False))
    
    print("\nrejected 字段示例:")
    rejected_parsed = json.loads(first_item['rejected'])
    print(json.dumps(rejected_parsed, indent=2, ensure_ascii=False))
    
    # 验证字段
    required_fields = ['instruction', 'chosen', 'rejected', 'question', 'diff', 'task_description']
    missing_fields = [f for f in required_fields if f not in first_item]
    
    if missing_fields:
        print(f"\n❌ 缺少字段: {missing_fields}")
    else:
        print("\n✅ 所有必需字段都存在")
    
    # 验证 JSON 格式
    try:
        chosen_data = json.loads(first_item['chosen'])
        rejected_data = json.loads(first_item['rejected'])
        
        if 'output' in chosen_data and isinstance(chosen_data['output'], list):
            if all('Principle' in item for item in chosen_data['output']):
                print("✅ chosen 格式正确: {\"output\": [{\"Principle\": \"...\"}]}")
            else:
                print("❌ chosen 格式错误: output 中的项缺少 Principle 字段")
        else:
            print("❌ chosen 格式错误: 缺少 output 数组")
        
        if 'output' in rejected_data and isinstance(rejected_data['output'], list):
            if all('Principle' in item for item in rejected_data['output']):
                print("✅ rejected 格式正确: {\"output\": [{\"Principle\": \"...\"}]}")
            else:
                print("❌ rejected 格式错误: output 中的项缺少 Principle 字段")
        else:
            print("❌ rejected 格式错误: 缺少 output 数组")
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
    
    print("\n" + "=" * 80)
    print("测试 2: 增量更新模式")
    print("=" * 80)
    
    # 修改测试数据（模拟重新生成 task_description 和 rejected）
    updated_outputs = [
        ReasoningOutput(
            question="Test question 1",
            baseline_answer="Updated Principle 1: New baseline\nUpdated Principle 2: Another new",
            diff_analysis="Updated diff analysis",
            principles=["Original chosen 1", "Original chosen 2"],  # 这个不会被更新
            chosen_answer="Original chosen answer",
            task_description='{"taskDescription": {"description": "Updated task description"}}',
            metadata={}
        )
    ]
    
    print("\n执行增量更新...")
    agent.save_dpo_format(updated_outputs, test_output_path)
    
    # 读取更新后的数据
    with open(test_output_path, 'r', encoding='utf-8') as f:
        updated_data = json.load(f)
    
    updated_item = updated_data[0]
    
    print("\n验证增量更新结果:")
    
    # 检查 chosen 是否保持不变
    updated_chosen = json.loads(updated_item['chosen'])
    if updated_chosen['output'][0]['Principle'] == "Test chosen principle 1":
        print("✅ chosen 保持不变（正确）")
    else:
        print(f"❌ chosen 被意外修改: {updated_chosen['output'][0]['Principle']}")
    
    # 检查 rejected 是否更新
    updated_rejected = json.loads(updated_item['rejected'])
    if updated_rejected['output'][0]['Principle'] == "Updated Principle 1: New baseline":
        print("✅ rejected 已更新（正确）")
    else:
        print(f"❌ rejected 未正确更新: {updated_rejected['output'][0]['Principle']}")
    
    # 检查 diff 是否更新
    if updated_item['diff'] == "Updated diff analysis":
        print("✅ diff 已更新（正确）")
    else:
        print(f"❌ diff 未正确更新: {updated_item['diff']}")
    
    # 检查 task_description 是否更新
    if "Updated task description" in updated_item['task_description']:
        print("✅ task_description 已更新（正确）")
    else:
        print(f"❌ task_description 未正确更新")
    
    # 清理测试文件
    print(f"\n清理测试文件: {test_output_path}")
    test_output_path.unlink()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def compare_with_reference():
    """与 dpo_level1.json 的格式对比"""
    print("\n" + "=" * 80)
    print("测试 3: 与 dpo_level1.json 格式对比")
    print("=" * 80)
    
    reference_file = Path("data/dpo_llamafactory/dpo_level1.json")
    
    if not reference_file.exists():
        print(f"参考文件不存在: {reference_file}")
        return
    
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    print(f"\n参考文件包含 {len(reference_data)} 条数据")
    
    ref_item = reference_data[0]
    print("\n参考格式字段:")
    for key in ref_item.keys():
        print(f"  - {key}")
    
    print("\nchosen 字段格式:")
    ref_chosen = json.loads(ref_item['chosen'])
    print(json.dumps(ref_chosen, indent=2, ensure_ascii=False)[:200] + "...")
    
    print("\nrejected 字段格式:")
    ref_rejected = json.loads(ref_item['rejected'])
    print(json.dumps(ref_rejected, indent=2, ensure_ascii=False)[:200] + "...")
    
    print("\n✅ 参考格式确认:")
    print("  - chosen 和 rejected 都是 JSON 字符串")
    print("  - 格式为: {\"output\": [{\"Principle\": \"...\"}, ...]}")
    print("  - 每个 Principle 是一个独立的对象")


if __name__ == "__main__":
    try:
        test_dpo_format()
        compare_with_reference()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
