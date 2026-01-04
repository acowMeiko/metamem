# MetaEvo Framework - Refactored Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 🎯 现代化、模块化的元优化框架，基于 memr3 设计模式重构

---

## 📖 简介

MetaEvo 是一个用于生成 DPO (Direct Preference Optimization) 训练数据、管理推理记忆、执行智能推理的完整框架。

本项目已完成从扁平化架构到模块化架构的**完整重构**，采用了多种设计模式，实现了：
- ✅ 数据与逻辑解耦
- ✅ Prompt 集中管理
- ✅ 多策略可切换
- ✅ 易扩展易测试

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 模型配置
export BASE_MODEL_NAME="/path/to/your/model"
export STRONG_MODEL_NAME="DeepSeek-R1"
export STRONG_MODEL_API_URL="https://api.example.com/v1/"
export STRONG_MODEL_KEY="your-api-key"

# 推理参数
export BATCH_SIZE=256
export MAX_WORKERS=20
```

### 3. 运行实验

#### Stage 1: 生成 DPO 训练数据

```bash
python run_experiments.py --stage 1 \
    --dataset gsm8k \
    --dataset-path dataset/gsm8k/test.jsonl \
    --output output/dpo_gsm8k.json
```

#### Stage 2: 更新 Memory

```bash
python run_experiments.py --stage 2 \
    --dpo-file data/dpo_llamafactory/dpo_all_levels_llamafactory.json
```

#### Stage 3: 推理 (带 Memory 引导)

```bash
python run_experiments.py --stage 3 \
    --dataset gsm8k \
    --dataset-path dataset/gsm8k/test.jsonl \
    --output output/inference_gsm8k.json
```

---

## 📂 项目结构

```
metanew3/
├── core/                      # 核心模块
│   ├── base.py               # 抽象基类
│   ├── stages.py             # Stage 1/2/3 Agent
│   └── config.py             # 配置管理
│
├── data/                      # 数据处理
│   └── processor.py          # 数据集处理器
│
├── inference/                 # 推理引擎
│   ├── engine.py             # 统一推理接口
│   ├── local_inference.py    # vLLM 推理
│   └── api_inference.py      # API 推理
│
├── templates/                 # Prompt 管理
│   └── prompts.py            # 集中式 Prompt
│
├── run_experiments.py        # 主入口 (推荐使用)
├── examples/                  # 使用示例
└── docs/                      # 文档
```

---

## 🎯 核心功能

### 三个推理阶段

#### Stage 1: DPO 数据生成
- 生成 Baseline 答案 (弱模型)
- 分析差异
- 提取原则
- 生成 Chosen 答案 (强模型)
- 输出 DPO 格式数据

#### Stage 2: Memory 更新
- 从 DPO 数据提取任务描述和原则
- 语义匹配已存在任务
- 合并或添加原则到 Memory

#### Stage 3: Memory 引导推理
- 根据问题生成任务描述
- 从 Memory 检索相关原则
- 执行带原则引导的推理

### 支持的数据集

- ✅ GSM8K (数学应用题)
- ✅ MATH (高等数学)
- ✅ BBH (Big-Bench Hard)
- ✅ MMLU (多选题)
- ✅ SVAMP (数学应用题)

**易扩展**: 注册新的预处理函数即可支持新数据集

---

## 💡 核心设计

### 1. 抽象基类模式

```python
from core.base import MetaAgentBase, ReasoningInput

class MyAgent(MetaAgentBase):
    def process(self, input_data: ReasoningInput):
        # 实现处理逻辑
        pass
```

### 2. 策略模式

```python
from core.stages import StageOneAgent, StageTwoAgent, InferenceAgent

# 不同策略可切换
agent = StageOneAgent(config)
# agent = StageTwoAgent(config)
# agent = InferenceAgent(config)

outputs = agent.process_batch(inputs)
```

### 3. 注册表模式

```python
from data.processor import DatasetProcessor

processor = DatasetProcessor()
processor.register('my_dataset', preprocess_func)
data = processor.load_dataset('my_dataset', 'path/to/data.json')
```

### 4. 统一推理接口

```python
from inference.engine import InferenceEngineBuilder

engine = (InferenceEngineBuilder()
          .set_weak_model('local', 'qwen2.5-14b')
          .set_strong_model('api', 'DeepSeek-R1', url='...', api_key='...')
          .build())

# 自动适配 vLLM 或 API
results = engine.batch_inference(prompts, model_type='weak')
```

---

## 📚 文档

- 📖 [重构指南](REFACTORING_GUIDE.md) - 详细的重构说明
- 🏗️ [架构对比](docs/ARCHITECTURE_COMPARISON.md) - 新旧架构对比
- ✅ [重构完成报告](REFACTORING_COMPLETE.md) - 交付成果总结
- 🚀 [快速开始示例](examples/quick_start.py) - 代码示例

---

## 🔧 高级用法

### 编程式使用

```python
from core.config import MetaConfig, initialize_config
from core.stages import StageOneAgent
from core.base import ReasoningInput
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
          .set_weak_model('local', config.models.weak_model_name)
          .set_strong_model('api', 'DeepSeek-R1', url='...', api_key='...')
          .build())

# 4. 创建 Agent
agent = StageOneAgent({
    'inference_engine': engine,
    'prompt_template': PromptTemplate(),
    'batch_size': 256
})

# 5. 处理数据
inputs = [ReasoningInput(question=d['question'], answer=d['answer']) for d in data]
outputs = agent.process_batch(inputs)

# 6. 保存结果
agent.save_dpo_format(outputs, 'output/dpo_data.json')
```

### 添加自定义数据集

```python
from data.processor import DatasetProcessor

def preprocess_my_dataset(raw_data):
    return [
        {"question": item['my_q'], "answer": item['my_a']}
        for item in raw_data
    ]

processor = DatasetProcessor()
processor.register('my_dataset', preprocess_my_dataset)
data = processor.load_dataset('my_dataset', 'path/to/data.json')
```

### 添加自定义 Prompt

```python
from templates.prompts import PromptTemplate
from string import Template

class MyPromptTemplate(PromptTemplate):
    CUSTOM_TEMPLATE = Template('Your custom prompt: $variable')
    
    def get_custom_prompt(self, variable: str) -> str:
        return self.CUSTOM_TEMPLATE.substitute(variable=variable)
```

---

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 运行特定测试
pytest tests/test_stages.py

# 查看覆盖率
pytest --cov=core --cov-report=html
```

---

## 🐛 调试

### 启用调试模式

```bash
python run_experiments.py --stage 1 --dataset gsm8k --dataset-path ... --debug
```

或设置环境变量：

```bash
export LOG_LEVEL=DEBUG
export DEBUG_MODE=true
```

### 查看日志

```bash
tail -f logs/metaevo.log
```

---

## 📊 性能优化

### vLLM 批处理

```python
# 增加批处理大小以提高吞吐量
export BATCH_SIZE=512  # 根据 GPU 显存调整
```

### API 并发调用

```python
# 增加并发线程数
export MAX_WORKERS=50  # 根据 API 限流调整
```

---

## 🤝 贡献指南

### 添加新功能

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8
- 添加类型注解
- 编写文档字符串
- 添加单元测试

---

## 📝 更新日志

### v2.0.0 (2026-01-04) - 架构重构

#### 新增
- ✨ 模块化架构 (core, data, inference, templates)
- ✨ 抽象基类和策略模式
- ✨ 统一推理引擎
- ✨ 集中 Prompt 管理
- ✨ 配置管理系统
- ✨ 完整文档和示例

#### 改进
- 🎨 数据与逻辑解耦
- 🎨 标准化输入输出格式
- 🎨 注册表模式支持扩展
- 🎨 依赖注入提高可测试性

#### 兼容性
- ♻️ 保留旧文件以保持向后兼容
- ♻️ 提供迁移指南

### v1.0.0 - 原始版本
- 基础功能实现

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本项目架构设计参考了 memr3 的优秀设计模式，感谢开源社区的贡献。

---

## 📞 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/metanew3/issues)
- Email: your.email@example.com

---

## ⭐ Star History

如果觉得这个项目有帮助，请给一个 Star ⭐️

---

<div align="center">
  
**🎉 现在就开始使用现代化的 MetaEvo 框架吧！**

[快速开始](#-快速开始) • [文档](#-文档) • [示例](examples/) • [贡献](#-贡献指南)

</div>
