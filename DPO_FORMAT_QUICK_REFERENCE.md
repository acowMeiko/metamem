# DPO 格式快速参考

## 标准格式

```json
{
  "instruction": "Based on the comparison of high-quality and low-quality answers, generate reusable problem-solving principles.",
  "chosen": "{\"output\": [{\"Principle\": \"Principle 1\"}, {\"Principle\": \"Principle 2\"}]}",
  "rejected": "{\"output\": [{\"Principle\": \"Principle A\"}]}",
  "question": "Question text...",
  "diff": "Diff analysis...",
  "task_description": "{\"taskDescription\": {...}}"
}
```

## 关键点

✅ **chosen/rejected 是 JSON 字符串（不是对象）**
✅ **格式：`{"output": [{"Principle": "..."}, ...]}`**
✅ **每个 Principle 只包含一个字段：`Principle`**
❌ **不包含 `Application` 或 `Explanation` 字段**

## 增量更新行为

| 字段 | 首次生成 | 增量更新 |
|------|---------|---------|
| instruction | ✅ 生成 | 🔒 保留 |
| chosen | ✅ 生成 | 🔒 保留 |
| question | ✅ 生成 | 🔒 保留 |
| rejected | ✅ 生成 | 🔄 更新 |
| diff | ✅ 生成 | 🔄 更新 |
| task_description | ✅ 生成 | 🔄 更新 |

## 使用场景

### 场景 1：首次生成
```bash
# DPO 文件不存在
python run_experiments.py stage1 --dataset gsm8k --output output/dpo_gsm8k.json
# → 完整生成所有字段
```

### 场景 2：更新 task_description
```bash
# DPO 文件已存在，想更新 task_description
python run_experiments.py stage1 --dataset gsm8k --output output/dpo_gsm8k.json
# → 自动进入增量更新模式
# → 只更新 task_description、rejected、diff
# → 保留原有的 chosen（节省成本）
```

### 场景 3：完全重新生成
```bash
# 需要完全重新生成所有内容
rm output/dpo_gsm8k.json  # 先删除
python run_experiments.py stage1 --dataset gsm8k --output output/dpo_gsm8k.json
# → 完整生成所有字段
```

## 测试命令

```bash
# 运行测试
python test_dpo_format.py

# 检查现有文件格式
python -c "import json; data=json.load(open('data/dpo_llamafactory/dpo_level1.json', 'r', encoding='utf-8')); chosen=json.loads(data[0]['chosen']); print(json.dumps(chosen, indent=2, ensure_ascii=False))"
```

## 常见问题

**Q: 为什么 chosen 不更新？**
A: chosen 来自强模型（成本高），通常质量已经很好，不需要重新生成。

**Q: 如何强制完整重新生成？**
A: 删除或移动现有的 DPO 文件即可。

**Q: 增量更新时数据长度不匹配怎么办？**
A: 系统会按最小长度更新，并保留额外的现有数据。
