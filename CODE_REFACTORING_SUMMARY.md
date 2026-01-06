# stages.py 代码重构总结

## 重构日期
2026-01-05

## 重构目标
在**不改变任何命名**的前提下，提高代码可读性和可维护性。

---

## 主要改进

### 1. ✨ 添加详细的架构说明

**文件头部增强：**
```python
"""
Architecture:
    StageOneAgent: 生成 DPO 训练数据
        ├─ 生成任务描述 (Task Description)
        ├─ 生成基线答案 (Baseline/Rejected)
        ├─ 分析差异 (Diff Analysis)
        ├─ 提取原则 (Principles)
        └─ 生成优质答案 (Chosen)
    
    StageTwoAgent: 更新记忆系统
        ├─ 语义匹配任务
        ├─ 合并或添加原则
        └─ 定期保存记忆
    
    InferenceAgent: 基于记忆的推理
        ├─ 生成任务描述
        ├─ 检索相关原则
        └─ 执行引导推理
"""
```

### 2. 📦 提取常量定义

**新增常量区块：**
```python
# ============================================================================
# 常量定义
# ============================================================================

# DPO 格式常量
DPO_INSTRUCTION = "Based on the comparison..."
DPO_OUTPUT_FORMAT = {"output": []}

# 默认配置参数
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_WORKERS = 20
DEFAULT_SAVE_FREQUENCY = 50
DEFAULT_MAX_TOKENS_TASK_DESC = 2560
DEFAULT_MAX_TOKENS_BASELINE = 2048
DEFAULT_MAX_TOKENS_DIFF = 1024
DEFAULT_MAX_TOKENS_PRINCIPLES = 2560
DEFAULT_MAX_TOKENS_CHOSEN = 4096
DEFAULT_MAX_TOKENS_INFERENCE = 2048
```

**好处：**
- ✅ 消除魔法数字
- ✅ 便于统一调整参数
- ✅ 提高代码可维护性

### 3. 📝 优化类和方法的文档字符串

#### 类级别文档

**Before:**
```python
class StageOneAgent(MetaAgentBase):
    """
    Stage 1: DPO Training Data Generation Agent.
    
    Flow: Baseline → Diff Analysis → ...
    """
```

**After:**
```python
class StageOneAgent(MetaAgentBase):
    """
    Stage 1: DPO Training Data Generation Agent.
    
    职责：生成用于 DPO 训练的数据对
    
    完整流程：
        1. Task Description  - 生成任务描述
        2. Baseline Answer   - 生成基线答案
        3. Diff Analysis     - 分析差异
        4. Principles        - 提取原则
        5. Chosen Answer     - 生成优质答案
    
    输出格式：
        符合 LlamaFactory DPO 训练格式
    
    增量更新支持：
        - 文件不存在：完整生成
        - 文件存在：只更新部分字段
    """
```

#### 方法级别文档

**Before:**
```python
def _generate_baseline_answers(self, questions: List[str]) -> List[str]:
    """Generate baseline (rejected) answers using the weak model."""
```

**After:**
```python
def _generate_baseline_answers(self, questions: List[str]) -> List[str]:
    """
    生成基线答案（将作为 DPO 的 rejected 答案）。
    
    使用弱模型直接回答问题，不使用任何指导原则。
    这些答案通常质量较低，用于与高质量答案形成对比。
    
    Args:
        questions: 问题列表
        
    Returns:
        基线答案列表
    """
```

### 4. 🗂️ 添加清晰的代码分区

**使用视觉分隔符：**
```python
# ============================================================================
# Stage 1: DPO 训练数据生成代理
# ============================================================================

class StageOneAgent(MetaAgentBase):
    
    # ------------------------------------------------------------------------
    # 公共接口方法
    # ------------------------------------------------------------------------
    
    def process(self, ...):
        ...
    
    # ------------------------------------------------------------------------
    # Stage 1 子流程：模型调用方法
    # ------------------------------------------------------------------------
    
    def _generate_task_descriptions(self, ...):
        ...
    
    # ------------------------------------------------------------------------
    # DPO 数据保存：支持完整生成和增量更新
    # ------------------------------------------------------------------------
    
    def save_dpo_format(self, ...):
        ...
```

### 5. 💡 改进代码注释

**Before:**
```python
# Stage 1.1: Generate task descriptions (for future memory lookup)
task_descs = self._generate_task_descriptions(questions)
```

**After:**
```python
# ===== 阶段 1.1: 生成任务描述 =====
self._log_processing("TASK_DESC", "Generating task descriptions...")
task_descs = self._generate_task_descriptions(questions)
```

**关键改进：**
- ✅ 使用醒目的分隔符（`=====`）
- ✅ 清晰的阶段标识
- ✅ 中文注释增强理解

### 6. 🔄 优化逻辑流程注释

**增量更新方法：**
```python
def _incremental_update(self, existing_data, outputs):
    """
    增量更新模式：只更新特定字段。
    
    保留字段（不更新）：
        - instruction: 固定文本
        - chosen: 来自强模型，成本高
        - question: 原始问题，不会改变
    
    更新字段：
        - rejected: 基线模型可能改进
        - diff: 差异分析可能优化
        - task_description: 任务描述可能改进
    """
```

### 7. 🎯 增强方法目的说明

**推理方法：**
```python
# 决定推理模式
if principles:
    # 模式 A: 引导推理（有原则）
    principles_text = "\n".join(f"- {p}" for p in principles)
    prompt = self.prompts.get_guided_answer_prompt(question, principles_text)
    inference_type = "guided"
else:
    # 模式 B: 直接推理（无原则）
    prompt = self.prompts.get_direct_answer_prompt(question)
    inference_type = "direct"
```

---

## 代码结构对比

### Before（原始结构）
```
stages.py
├─ Imports
├─ StageOneAgent
│  ├─ __init__
│  ├─ process methods
│  ├─ pipeline methods (混杂)
│  └─ save methods
├─ StageTwoAgent
└─ InferenceAgent
```

### After（重构后结构）
```
stages.py
├─ Module docstring (with architecture)
├─ Imports
├─ Constants (新增)
│  ├─ DPO format constants
│  └─ Default parameters
├─ StageOneAgent (清晰分区)
│  ├─ Class docstring (详细说明)
│  ├─ __init__ (with comments)
│  ├─ Public interface methods
│  ├─ Stage 1 sub-processes
│  └─ DPO data saving
├─ StageTwoAgent (清晰分区)
│  ├─ Class docstring (详细说明)
│  └─ Well-commented methods
└─ InferenceAgent (清晰分区)
   ├─ Class docstring (详细说明)
   └─ Well-commented methods
```

---

## 可读性提升统计

| 指标 | Before | After | 提升 |
|------|--------|-------|------|
| 类文档字符串 | 简短 | 详细（职责+流程+特性） | ⭐⭐⭐ |
| 方法文档字符串 | 一行 | 多行（说明+参数+返回） | ⭐⭐⭐ |
| 代码分区 | 无 | 清晰分隔符 | ⭐⭐⭐ |
| 常量管理 | 魔法数字 | 统一常量 | ⭐⭐⭐ |
| 注释质量 | 英文简短 | 中文详细 | ⭐⭐⭐ |
| 逻辑说明 | 基本 | 详细（含原因） | ⭐⭐⭐ |

---

## 保持不变的内容

✅ **所有命名**（类名、方法名、变量名）
✅ **所有逻辑**（业务逻辑完全一致）
✅ **所有接口**（API 签名不变）
✅ **功能行为**（输入输出保持一致）

---

## 测试验证

### 语法检查
```bash
python -c "from core.stages import StageOneAgent, StageTwoAgent, InferenceAgent"
# ✅ 通过
```

### 功能测试
```bash
python test_dpo_format.py
# ✅ 所有测试通过
```

---

## 维护建议

### 1. 持续改进文档
- 当添加新功能时，更新类文档字符串
- 保持方法文档字符串的详细程度

### 2. 使用常量
- 新增参数优先定义为常量
- 避免硬编码魔法数字

### 3. 保持分区习惯
- 使用统一的分隔符格式
- 相关方法归类在同一分区

### 4. 注释原则
- 说明"为什么"而不只是"做什么"
- 关键决策点添加详细注释
- 复杂逻辑添加示例

---

## 代码示例对比

### 示例 1: 常量使用

**Before:**
```python
return self.engine.batch_inference(
    prompts=prompts,
    model_type='weak',
    batch_size=self.batch_size,
    max_tokens=2560,  # 魔法数字
    temperature=0.1
)
```

**After:**
```python
return self.engine.batch_inference(
    prompts=prompts,
    model_type='weak',
    batch_size=self.batch_size,
    max_tokens=DEFAULT_MAX_TOKENS_TASK_DESC,  # 语义化常量
    temperature=0.1
)
```

### 示例 2: 文档字符串

**Before:**
```python
def _analyze_differences(self, questions, predictions, labels):
    """Analyze differences between baseline and ground truth."""
```

**After:**
```python
def _analyze_differences(self, questions, predictions, labels):
    """
    分析基线答案与标准答案之间的差异。
    
    对比弱模型生成的基线答案与正确答案，识别关键差异点，
    为后续提取改进原则提供依据。
    
    Args:
        questions: 问题列表
        predictions: 基线答案（预测）列表
        labels: 标准答案（标签）列表
        
    Returns:
        差异分析结果列表
    """
```

---

## 后续优化建议

### 短期
1. ✅ 添加类型提示（已完成）
2. ⏳ 添加单元测试覆盖率
3. ⏳ 添加性能监控日志

### 中期
1. ⏳ 提取配置类（Config dataclass）
2. ⏳ 添加错误处理装饰器
3. ⏳ 实现重试机制

### 长期
1. ⏳ 重构为异步架构
2. ⏳ 添加插件系统
3. ⏳ 实现流式处理

---

## 总结

✅ **可读性显著提升**  
✅ **维护成本降低**  
✅ **功能完全保留**  
✅ **零破坏性修改**

重构遵循"Boy Scout Rule"：让代码比你发现时更好！
