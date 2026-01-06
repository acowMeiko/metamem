# Stage 1 DPO 格式修复和增量更新功能

## 修改日期
2026-01-05

## 修改内容

### 1. 修复 DPO 输出格式（与 dpo_level1.json 完全一致）

**问题：**
之前的 `save_dpo_format` 方法输出的格式不符合 `dpo_level1.json` 的标准格式。

**解决方案：**
修改 `core/stages.py` 中的 `StageOneAgent.save_dpo_format()` 方法，确保输出格式为：

```json
{
  "instruction": "Based on the comparison of high-quality and low-quality answers, generate reusable problem-solving principles.",
  "chosen": "{\"output\": [{\"Principle\": \"...\"}, {\"Principle\": \"...\"}]}",
  "rejected": "{\"output\": [{\"Principle\": \"...\"}, {\"Principle\": \"...\"}]}",
  "question": "...",
  "diff": "...",
  "task_description": "..."
}
```

**关键点：**
- `chosen` 和 `rejected` 字段是 **JSON 字符串**（不是对象）
- 内容格式为：`{"output": [{"Principle": "..."}, ...]}`
- 每个 Principle 都包含在独立的对象中，只有 `Principle` 字段

### 2. 新增增量更新功能

**功能描述：**
添加了智能的增量更新逻辑，可以在不重新生成所有内容的情况下，只更新特定字段。

**使用场景：**
1. **DPO 文件为空或不存在** → 完整生成所有字段
2. **DPO 文件已存在且非空** → 增量更新模式

**增量更新规则：**

| 字段 | 操作 | 说明 |
|------|------|------|
| `instruction` | 保留 | 固定文本，不需要更新 |
| `chosen` | **保留** | 高质量答案的 principles，来自强模型，通常不需要重新生成 |
| `question` | 保留 | 原始问题，不会改变 |
| `rejected` | **更新** | 从新生成的 baseline_answer 中提取 principles |
| `diff` | **更新** | 重新生成的差异分析 |
| `task_description` | **更新** | 重新生成的任务描述 |

**为什么只更新这三个字段？**
- `task_description`：可能需要优化任务描述的质量
- `rejected`：基线模型可能会更新，需要重新提取 principles
- `diff`：差异分析可能随着模型改进而变化

**为什么保留 `chosen`？**
- `chosen` 来自强模型（如 GPT-4），质量高且生成成本高
- 在大多数情况下不需要重新生成
- 保留原有数据可以节省大量的 API 调用成本

## 新增的内部方法

### `_extract_principles_to_json(principles_list: List[str]) -> str`
将 principles 列表转换为符合格式的 JSON 字符串。

**输入：** `["Principle 1", "Principle 2"]`

**输出：** `'{"output": [{"Principle": "Principle 1"}, {"Principle": "Principle 2"}]}'`

### `_incremental_update(existing_data: List[Dict], outputs: List[ReasoningOutput]) -> List[Dict]`
执行增量更新逻辑。

**处理逻辑：**
1. 对比现有数据长度和新数据长度
2. 遍历每条数据，保留特定字段，更新其他字段
3. 如果现有数据更多，保留额外的数据

### `_generate_full_dpo_data(outputs: List[ReasoningOutput]) -> List[Dict]`
执行完整生成逻辑，生成所有字段。

## 使用示例

### 场景 1：首次生成 DPO 数据
```python
from core.stages import StageOneAgent
from pathlib import Path

# ... 初始化 agent ...

# 首次生成（文件不存在）
outputs = agent.process_batch(inputs)
agent.save_dpo_format(outputs, Path("output/dpo_new.json"))

# 日志输出：
# "完整生成完成: 创建了 100 条新数据"
```

### 场景 2：更新现有 DPO 数据
```python
# 文件已存在且包含数据
outputs = agent.process_batch(inputs)
agent.save_dpo_format(outputs, Path("output/dpo_existing.json"))

# 日志输出：
# "检测到现有 DPO 文件包含 100 条数据，启用增量更新模式"
# "将更新: task_description, rejected 中的 principle, diff"
# "增量更新完成: 更新了 100 条数据"
```

### 场景 3：处理长度不匹配
```python
# 现有数据 150 条，新生成 100 条
outputs = agent.process_batch(inputs)  # 100 条
agent.save_dpo_format(outputs, Path("output/dpo_existing.json"))

# 日志输出：
# "数据长度不匹配: 现有 150 条，新生成 100 条。将按照最小长度进行更新。"
# "保留额外的 50 条现有数据"
# "增量更新完成: 更新了 150 条数据"
```

## 测试验证

运行测试脚本验证功能：
```bash
python test_dpo_format.py
```

**测试内容：**
1. ✅ DPO 格式生成（完整模式）
2. ✅ 增量更新模式
3. ✅ 与 dpo_level1.json 格式对比

**测试结果：**
- 所有必需字段都存在
- chosen 和 rejected 格式正确
- 增量更新正确保留 chosen
- 增量更新正确更新 rejected、diff、task_description

## 兼容性说明

### 向后兼容
- 旧的代码调用 `save_dpo_format()` 时会自动判断是完整生成还是增量更新
- 不需要修改调用代码

### 文件格式兼容
- 与 `data/dpo_llamafactory/dpo_level*.json` 格式完全一致
- 可以直接用于 LlamaFactory DPO 训练

## 注意事项

1. **数据长度匹配**：
   - 增量更新时，建议新数据和现有数据长度一致
   - 如果长度不同，会按最小长度更新，并保留额外的现有数据

2. **chosen 字段保留**：
   - 如果需要完全重新生成所有字段，请先删除或移动现有文件
   - 或者手动修改代码强制使用完整生成模式

3. **Principle 解析**：
   - 当前使用简单的换行分割来解析 principles
   - 如果 baseline_answer 格式复杂，可能需要改进 `_parse_principles()` 方法

## 相关文件

- `core/stages.py` - 主要修改文件
- `test_dpo_format.py` - 测试脚本
- `data/dpo_llamafactory/dpo_level*.json` - 参考格式文件

## 后续优化建议

1. **更智能的 Principle 解析**：
   - 支持 JSON 格式的 principle 输出
   - 支持 Markdown 列表格式
   - 支持编号列表格式

2. **选择性更新**：
   - 添加参数控制具体更新哪些字段
   - 例如：`save_dpo_format(outputs, path, update_fields=['rejected', 'diff'])`

3. **版本控制**：
   - 保存生成时间戳
   - 记录生成版本信息
   - 便于追踪数据变更历史
