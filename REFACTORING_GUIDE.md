# MetaEvo Framework 重构完成报告

## 📋 重构概述

本次重构将 `metanew3` 项目从扁平化、紧耦合的结构重构为模块化、可扩展的 `memr3` 风格架构。

### 核心改进

1. **模块化设计**: 清晰的职责分离，每个模块有明确的功能边界
2. **数据与逻辑解耦**: Agent 不再直接处理文件路径，只接收标准化数据
3. **策略模式**: 不同推理阶段作为独立策略类实现
4. **模板注册**: 所有 Prompt 集中管理，无硬编码
5. **配置化**: 统一的配置管理系统

---

## 🏗️ 新架构目录结构

```
metanew3/
├── core/                          # 核心模块
│   ├── __init__.py               # 模块导出
│   ├── base.py                   # 抽象基类 (MetaAgentBase)
│   ├── stages.py                 # 具体策略类 (Stage1/2/3 Agent)
│   └── config.py                 # 配置管理类
│
├── data/                          # 数据处理模块
│   ├── __init__.py
│   └── processor.py              # 数据集处理器 (注册表模式)
│
├── inference/                     # 推理引擎
│   ├── __init__.py
│   ├── engine.py                 # 统一推理引擎 (适配器模式)
│   ├── local_inference.py        # vLLM 本地推理 (原有)
│   └── api_inference.py          # API 推理 (原有)
│
├── templates/                     # Prompt 模板
│   ├── __init__.py
│   └── prompts.py                # 集中式 Prompt 管理
│
├── module/                        # 辅助模块 (保留兼容)
│   ├── memory_module.py          # 记忆管理
│   ├── execute_module.py         # 执行模块 (可逐步废弃)
│   └── plan_module.py            # 规划模块 (可逐步废弃)
│
├── run_experiments.py            # 新主入口 (推荐)
├── main.py                       # 旧主入口 (保留兼容)
├── config.py                     # 旧配置 (保留兼容)
│
└── [其他原有文件保持不变]
```

---

## 🎯 核心设计模式

### 1. 抽象基类模式 (Abstract Base Class)

**文件**: `core/base.py`

```python
class MetaAgentBase(ABC):
    """所有 Agent 的抽象基类"""
    
    @abstractmethod
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """单条处理"""
        pass
    
    @abstractmethod
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """批量处理"""
        pass
```

**设计优势**:
- 统一接口，保证所有 Agent 行为一致
- 标准化输入输出格式 (`ReasoningInput` / `ReasoningOutput`)
- 易于扩展新的推理策略

### 2. 策略模式 (Strategy Pattern)

**文件**: `core/stages.py`

三个具体策略类：
- `StageOneAgent`: DPO 数据生成
- `StageTwoAgent`: Memory 更新
- `InferenceAgent`: 带 Memory 引导的推理

**示例**:
```python
# Stage 1: DPO 生成
agent = StageOneAgent(config)
outputs = agent.process_batch(inputs)
agent.save_dpo_format(outputs, output_path)

# Stage 2: Memory 更新
agent = StageTwoAgent(config)
outputs = agent.process_batch(inputs)

# Stage 3: 推理
agent = InferenceAgent(config)
outputs = agent.process_batch(inputs)
```

### 3. 注册表模式 (Registry Pattern)

**文件**: `data/processor.py`

```python
class DatasetProcessor:
    def __init__(self):
        self._preprocessors = {
            'gsm8k': self._preprocess_gsm8k,
            'math': self._preprocess_math,
            'bbh': self._preprocess_bbh,
            'mmlu': self._preprocess_mmlu,
            'svamp': self._preprocess_svamp,
        }
    
    def register(self, dataset_name: str, preprocessor: Callable):
        """注册新数据集处理器"""
        self._preprocessors[dataset_name] = preprocessor
```

**扩展新数据集**:
```python
processor = DatasetProcessor()
processor.register('new_dataset', preprocess_new_dataset)
data = processor.load_dataset('new_dataset', 'path/to/data.json')
```

### 4. 适配器模式 (Adapter Pattern)

**文件**: `inference/engine.py`

统一 vLLM (本地) 和 API (远程) 推理接口：

```python
class InferenceEngine:
    def batch_inference(self, prompts, model_type='weak', ...):
        """统一批量推理接口"""
        if model_config['type'] == 'local':
            return vllm_batch_inference(...)
        else:
            return self.concurrent_api_inference(...)
```

### 5. 模板注册模式 (Template Registry)

**文件**: `templates/prompts.py`

```python
class PromptTemplate:
    TASK_DESC_TEMPLATE = Template('...')
    DIRECT_ANSWER_TEMPLATE = Template('...')
    GUIDED_ANSWER_TEMPLATE = Template('...')
    
    def get_task_description_prompt(self, question: str) -> str:
        return self.TASK_DESC_TEMPLATE.substitute(question=question)
```

**优势**:
- 所有 Prompt 集中管理
- 无硬编码
- 易于版本控制和 A/B 测试

---

## 🔄 原有逻辑迁移对照表

### Stage 1: `stage_first.py` → `core/stages.py::StageOneAgent`

| 原有函数 | 新方法 | 说明 |
|---------|--------|------|
| `prepare_stage1()` | `StageOneAgent.process_batch()` | 主处理流程 |
| `batch_generate_task_descriptions()` | `_generate_task_descriptions()` | 任务描述生成 |
| `batch_answer_questions_directly()` | `_generate_baseline_answers()` | Baseline 生成 |
| `batch_generate_difference_list()` | `_analyze_differences()` | 差异分析 |
| `batch_generate_principles()` | `_extract_principles()` | 原则提取 |
| `concurrent_generate_chosen()` | `_generate_chosen_answers()` | Chosen 生成 |

### Stage 2: `stage_second.py` → `core/stages.py::StageTwoAgent`

| 原有函数 | 新方法 | 说明 |
|---------|--------|------|
| `prepare_step2_update_memory_from_dpo()` | `StageTwoAgent.process_batch()` | Memory 更新 |
| 内联逻辑 | `memory.retrieve()` | 语义匹配 |
| 内联逻辑 | `memory.merge_principles()` | 原则合并 |

### 数据处理: `stage_first.py::数据集适配层` → `data/processor.py`

| 原有函数 | 新方法 | 说明 |
|---------|--------|------|
| `preprocess_gsm8k()` | `DatasetProcessor._preprocess_gsm8k()` | GSM8K 预处理 |
| `preprocess_math()` | `DatasetProcessor._preprocess_math()` | MATH 预处理 |
| `preprocess_bbh()` | `DatasetProcessor._preprocess_bbh()` | BBH 预处理 |
| `preprocess_mmlu()` | `DatasetProcessor._preprocess_mmlu()` | MMLU 预处理 |
| `preprocess_svamp()` | `DatasetProcessor._preprocess_svamp()` | SVAMP 预处理 |
| `load_and_preprocess_dataset()` | `DatasetProcessor.load_dataset()` | 统一加载接口 |

---

## 🚀 使用方式

### 新架构使用 (推荐)

```bash
# Stage 1: 生成 DPO 数据
python run_experiments.py --stage 1 \
    --dataset gsm8k \
    --dataset-path dataset/gsm8k/test.jsonl \
    --output output/dpo_gsm8k.json

# Stage 2: 更新 Memory
python run_experiments.py --stage 2 \
    --dpo-file data/dpo_llamafactory/dpo_all_levels_llamafactory.json

# Stage 3: 推理 (带 Memory 引导)
python run_experiments.py --stage 3 \
    --dataset gsm8k \
    --dataset-path dataset/gsm8k/test.jsonl \
    --output output/inference_gsm8k.json

# 调试模式
python run_experiments.py --stage 1 --dataset gsm8k --dataset-path ... --debug
```

### 环境变量配置

```bash
# 模型配置
export BASE_MODEL_NAME="/home/share/hcz/qwen2.5-14b-awq"
export STRONG_MODEL_NAME="DeepSeek-R1"
export STRONG_MODEL_API_URL="https://llmapi.paratera.com/v1/"
export STRONG_MODEL_KEY="sk-xxx"

# 推理参数
export BATCH_SIZE=256
export MAX_WORKERS=20
export DEFAULT_TEMPERATURE=0.0

# 日志
export LOG_LEVEL=INFO
export DEBUG_MODE=false
```

### 编程式使用

```python
from core.config import MetaConfig, initialize_config
from core.stages import StageOneAgent
from data.processor import DatasetProcessor
from inference.engine import InferenceEngineBuilder
from templates.prompts import PromptTemplate

# 1. 初始化配置
config = MetaConfig.from_env()
initialize_config(config)

# 2. 加载数据
processor = DatasetProcessor()
data = processor.load_dataset('gsm8k', 'dataset/gsm8k/test.jsonl')

# 3. 构建推理引擎
engine = (InferenceEngineBuilder()
          .set_weak_model('local', '/path/to/model')
          .set_strong_model('api', 'DeepSeek-R1', url='...', api_key='...')
          .build())

# 4. 创建 Agent
agent = StageOneAgent({
    'inference_engine': engine,
    'prompt_template': PromptTemplate(),
    'batch_size': 256
})

# 5. 处理数据
from core.base import ReasoningInput
inputs = [ReasoningInput(question=d['question'], answer=d['answer']) for d in data]
outputs = agent.process_batch(inputs)

# 6. 保存结果
agent.save_dpo_format(outputs, 'output/dpo_data.json')
```

---

## 📊 架构对比

### 旧架构问题

❌ **紧耦合**: 数据处理、Prompt 构建、模型推理混在一起  
❌ **硬编码**: Prompt 散落在各处，难以维护  
❌ **职责不清**: 单个文件混杂多种功能  
❌ **难扩展**: 添加新数据集或推理策略需要大量修改  
❌ **配置混乱**: 配置参数分散，缺乏验证

### 新架构优势

✅ **模块化**: 清晰的模块边界，单一职责  
✅ **解耦**: 数据 → Agent → 推理引擎分离  
✅ **可扩展**: 注册表模式，易于添加新功能  
✅ **可维护**: 集中式配置和 Prompt 管理  
✅ **可测试**: 标准接口，易于单元测试  
✅ **类型安全**: 使用 dataclass 定义标准格式

---

## 🔧 扩展指南

### 添加新数据集

```python
# 在 data/processor.py 中添加
@staticmethod
def _preprocess_new_dataset(raw_data: List[Dict]) -> List[Dict[str, str]]:
    return [
        {"question": item['q'], "answer": item['a']}
        for item in raw_data
    ]

# 注册
processor = DatasetProcessor()
processor.register('new_dataset', processor._preprocess_new_dataset)
```

### 添加新推理策略

```python
# 在 core/stages.py 中添加
class CustomAgent(MetaAgentBase):
    def _validate_config(self):
        # 验证配置
        pass
    
    def process(self, input_data):
        # 单条处理逻辑
        pass
    
    def process_batch(self, inputs):
        # 批量处理逻辑
        pass
```

### 添加新 Prompt

```python
# 在 templates/prompts.py 中添加
class PromptTemplate:
    NEW_TEMPLATE = Template('Your prompt: $variable')
    
    def get_new_prompt(self, variable: str) -> str:
        return self.NEW_TEMPLATE.substitute(variable=variable)
```

---

## 🧪 测试建议

### 单元测试示例

```python
import unittest
from core.base import ReasoningInput
from core.stages import StageOneAgent

class TestStageOneAgent(unittest.TestCase):
    def setUp(self):
        self.agent = StageOneAgent(test_config)
    
    def test_process_single_input(self):
        inp = ReasoningInput(question="What is 2+2?", answer="4")
        output = self.agent.process(inp)
        self.assertIsNotNone(output.baseline_answer)
        self.assertIsNotNone(output.chosen_answer)
```

### 集成测试

```python
def test_end_to_end_pipeline():
    # 加载测试数据
    processor = DatasetProcessor()
    data = processor.load_dataset('gsm8k', 'test_data.jsonl')
    
    # 运行完整流程
    agent = StageOneAgent(config)
    outputs = agent.process_batch(inputs)
    
    # 验证输出
    assert len(outputs) == len(data)
    assert all(o.chosen_answer for o in outputs)
```

---

## 📝 迁移清单

- [x] 创建核心模块 (`core/`)
- [x] 实现抽象基类 (`core/base.py`)
- [x] 迁移 Stage 1 逻辑 (`core/stages.py::StageOneAgent`)
- [x] 迁移 Stage 2 逻辑 (`core/stages.py::StageTwoAgent`)
- [x] 实现推理 Agent (`core/stages.py::InferenceAgent`)
- [x] 创建数据处理层 (`data/processor.py`)
- [x] 统一推理引擎 (`inference/engine.py`)
- [x] 集中 Prompt 管理 (`templates/prompts.py`)
- [x] 配置管理系统 (`core/config.py`)
- [x] 新主入口 (`run_experiments.py`)
- [x] 编写重构文档

### 后续优化建议

1. **向后兼容**: 保留旧文件，逐步迁移
2. **单元测试**: 为核心模块添加测试
3. **文档完善**: API 文档和使用示例
4. **性能优化**: Profiling 和瓶颈分析
5. **监控日志**: 结构化日志和指标收集

---

## 🎉 总结

本次重构成功将 `metanew3` 从扁平化架构升级为模块化、可扩展的现代架构，参考了 `memr3` 的优秀设计模式：

1. **抽象基类模式**: 统一接口
2. **策略模式**: 多种推理策略
3. **注册表模式**: 数据集/Prompt 管理
4. **适配器模式**: 统一推理接口
5. **配置化**: 集中配置管理

新架构在保持功能等价的前提下，大幅提升了代码的**可维护性**、**可扩展性**和**可测试性**。

**推荐**：后续开发使用新架构 (`run_experiments.py`)，旧文件保留作为向后兼容参考。
