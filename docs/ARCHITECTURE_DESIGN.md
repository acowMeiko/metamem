# MetaEvo 架构设计文档

## 目录
- [系统架构](#系统架构)
- [核心组件](#核心组件)
- [数据流](#数据流)
- [类图](#类图)
- [时序图](#时序图)

---

## 系统架构

### 整体架构 (分层视图)

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                    (应用层)                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         run_experiments.py (主入口)                   │  │
│  │  - CLI 参数解析                                       │  │
│  │  - 组件组装和依赖注入                                  │  │
│  │  - 流程编排                                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Business Logic Layer                      │
│                    (业务逻辑层)                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              core/stages.py (策略实现)                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │StageOne    │  │StageTwo    │  │Inference   │    │  │
│  │  │Agent       │  │Agent       │  │Agent       │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            core/base.py (抽象基类)                    │  │
│  │  - MetaAgentBase (定义标准接口)                       │  │
│  │  - ReasoningInput / ReasoningOutput (标准格式)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Service Layer                             │
│                    (服务层)                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐   │
│  │ data/         │  │ inference/    │  │ templates/   │   │
│  │ processor.py  │  │ engine.py     │  │ prompts.py   │   │
│  │               │  │               │  │              │   │
│  │ 数据预处理     │  │ 推理引擎      │  │ Prompt管理   │   │
│  │ (注册表)      │  │ (适配器)      │  │ (模板注册)    │   │
│  └───────────────┘  └───────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Infrastructure Layer                      │
│                    (基础设施层)                               │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐   │
│  │ inference/    │  │ module/       │  │ core/        │   │
│  │ local_        │  │ memory_       │  │ config.py    │   │
│  │ inference.py  │  │ module.py     │  │              │   │
│  │               │  │               │  │ 配置管理      │   │
│  │ vLLM 推理     │  │ Memory 管理   │  │              │   │
│  └───────────────┘  └───────────────┘  └──────────────┘   │
│  ┌───────────────┐                                         │
│  │ inference/    │                                         │
│  │ api_          │                                         │
│  │ inference.py  │                                         │
│  │               │                                         │
│  │ API 推理      │                                         │
│  └───────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. 抽象基类 (core/base.py)

```
┌───────────────────────────────────────────────────┐
│           MetaAgentBase (抽象基类)                 │
├───────────────────────────────────────────────────┤
│ + __init__(config: Dict)                          │
│ + process(input: ReasoningInput) → Output        │
│ + process_batch(inputs: List) → List[Output]     │
│ # _validate_config() → None                       │
│ # _log_processing(stage, msg, level) → None      │
└───────────────────────────────────────────────────┘
                        △
                        │ (继承)
        ┌───────────────┼───────────────┐
        │               │               │
┌───────────────┐ ┌────────────┐ ┌────────────┐
│ StageOne      │ │ StageTwo   │ │ Inference  │
│ Agent         │ │ Agent      │ │ Agent      │
├───────────────┤ ├────────────┤ ├────────────┤
│ DPO数据生成    │ │ Memory更新 │ │ 推理执行    │
└───────────────┘ └────────────┘ └────────────┘
```

### 2. 数据格式 (core/base.py)

```
┌─────────────────────────────────────────────┐
│         ReasoningInput (输入格式)            │
├─────────────────────────────────────────────┤
│ + question: str                             │
│ + answer: str                               │
│ + task_description: Optional[str]           │
│ + principles: Optional[List[str]]           │
│ + metadata: Optional[Dict]                  │
└─────────────────────────────────────────────┘
                    │
                    ▼ (处理)
┌─────────────────────────────────────────────┐
│        ReasoningOutput (输出格式)            │
├─────────────────────────────────────────────┤
│ + question: str                             │
│ + baseline_answer: Optional[str]            │
│ + diff_analysis: Optional[str]              │
│ + principles: Optional[List[str]]           │
│ + chosen_answer: Optional[str]              │
│ + task_description: Optional[str]           │
│ + metadata: Optional[Dict]                  │
└─────────────────────────────────────────────┘
```

### 3. 数据处理器 (data/processor.py)

```
┌─────────────────────────────────────────────┐
│         DatasetProcessor (注册表模式)        │
├─────────────────────────────────────────────┤
│ - _preprocessors: Dict[str, Callable]       │
├─────────────────────────────────────────────┤
│ + register(name, preprocessor)              │
│ + load_dataset(name, path) → List[Dict]    │
│ - _load_file(path) → Any                    │
│ - _preprocess_gsm8k(data) → List[Dict]     │
│ - _preprocess_math(data) → List[Dict]      │
│ - _preprocess_bbh(data) → List[Dict]       │
│ - _preprocess_mmlu(data) → List[Dict]      │
│ - _preprocess_svamp(data) → List[Dict]     │
└─────────────────────────────────────────────┘

数据流:
原始数据 → load_dataset() → 标准格式
  (各种格式)              {"question": str, "answer": str}
```

### 4. 推理引擎 (inference/engine.py)

```
┌─────────────────────────────────────────────┐
│       InferenceEngine (适配器模式)           │
├─────────────────────────────────────────────┤
│ - config: Dict                              │
│ - weak_model: Dict                          │
│ - strong_model: Dict                        │
├─────────────────────────────────────────────┤
│ + single_inference(prompt, ...) → str       │
│ + batch_inference(prompts, ...) → List[str]│
│ + concurrent_api_inference(...) → List     │
│ - _validate_config() → None                 │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌────────────────┐      ┌──────────────┐
│ local_         │      │ api_         │
│ inference.py   │      │ inference.py │
│ (vLLM)         │      │ (OpenAI API) │
└────────────────┘      └──────────────┘
```

### 5. Prompt 管理 (templates/prompts.py)

```
┌─────────────────────────────────────────────┐
│         PromptTemplate (模板注册)            │
├─────────────────────────────────────────────┤
│ 类属性:                                      │
│ - TASK_DESC_TEMPLATE: Template              │
│ - DIRECT_ANSWER_TEMPLATE: Template          │
│ - GUIDED_ANSWER_TEMPLATE: Template          │
│ - DIFF_ANALYSIS_TEMPLATE: Template          │
│ - PRINCIPLE_TEMPLATE: Template              │
│ - PRINCIPLE_MATCH_TEMPLATE: Template        │
│ - DIALOGUE_FORMAT: Template                 │
├─────────────────────────────────────────────┤
│ 方法:                                        │
│ + get_task_description_prompt(q) → str      │
│ + get_direct_answer_prompt(q) → str         │
│ + get_guided_answer_prompt(q, p) → str      │
│ + get_diff_analysis_prompt(q,p,l) → str     │
│ + get_principle_prompt(q, d) → str          │
│ + get_principle_match_prompt(n, o) → str    │
│ + format_dialogue(query) → str              │
└─────────────────────────────────────────────┘
```

---

## 数据流

### Stage 1: DPO 数据生成流程

```
原始数据集 (JSON/JSONL)
    │
    ▼ DatasetProcessor.load_dataset()
标准格式 [{"question": str, "answer": str}]
    │
    ▼ 转换为 ReasoningInput
List[ReasoningInput]
    │
    ▼ StageOneAgent.process_batch()
    │
    ├──► 1. 生成任务描述
    │    │  Prompt: TASK_DESC_TEMPLATE
    │    │  Model: weak (local)
    │    └─► task_descriptions: List[str]
    │
    ├──► 2. 生成 Baseline 答案
    │    │  Prompt: DIRECT_ANSWER_TEMPLATE
    │    │  Model: weak (local)
    │    └─► baselines: List[str]
    │
    ├──► 3. 分析差异
    │    │  Input: question, baseline, ground_truth
    │    │  Prompt: DIFF_ANALYSIS_TEMPLATE
    │    │  Model: weak (local)
    │    └─► diffs: List[str]
    │
    ├──► 4. 提取原则
    │    │  Input: question, diff
    │    │  Prompt: PRINCIPLE_TEMPLATE
    │    │  Model: weak (local)
    │    └─► principles: List[str]
    │
    └──► 5. 生成 Chosen 答案
         │  Input: question, principles
         │  Prompt: GUIDED_ANSWER_TEMPLATE
         │  Model: strong (API, concurrent)
         └─► chosen_answers: List[str]
    │
    ▼
List[ReasoningOutput]
    │
    ▼ save_dpo_format()
DPO 格式 JSON 文件
{
  "input": "Question: ... Error Analysis: ...",
  "rejected": "baseline_answer",
  "chosen": "chosen_answer"
}
```

### Stage 2: Memory 更新流程

```
DPO 数据文件 (JSON)
    │
    ▼ 解析和转换
List[ReasoningInput]
(包含 task_description 和 principles)
    │
    ▼ StageTwoAgent.process_batch()
    │
    └──► For each input:
         │
         ├──► MemoryManager.retrieve(task_desc)
         │    │  使用语义匹配查找已存在任务
         │    └─► matched_task, existing_principles
         │
         ├──► If matched:
         │    └─► MemoryManager.merge_principles()
         │         │  解决冲突 (Redundant/Conflicting/Irrelevant)
         │         └─► 合并原则
         │
         └──► If not matched:
              └─► MemoryManager.add_task()
                   └─► 添加新任务
    │
    ▼ 定期保存
Memory JSON 文件
{
  "task_description_1": [principle1, principle2, ...],
  "task_description_2": [principle3, principle4, ...],
  ...
}
```

### Stage 3: 推理流程

```
测试问题
    │
    ▼ 转换为 ReasoningInput
List[ReasoningInput]
    │
    ▼ InferenceAgent.process_batch()
    │
    ├──► 1. 生成任务描述
    │    │  Prompt: TASK_DESC_TEMPLATE
    │    │  Model: weak (local)
    │    └─► task_descriptions: List[str]
    │
    ├──► 2. 从 Memory 检索原则
    │    │  For each task_description:
    │    │    MemoryManager.retrieve(task_desc)
    │    └─► principles_list: List[List[str]]
    │
    └──► 3. 执行推理
         │  If has principles:
         │    Prompt: GUIDED_ANSWER_TEMPLATE
         │  Else:
         │    Prompt: DIRECT_ANSWER_TEMPLATE
         │  Model: weak (local)
         └─► answers: List[str]
    │
    ▼
List[ReasoningOutput]
(包含 predicted_answer, has_principles 等)
    │
    ▼ 保存结果
推理结果 JSON 文件
```

---

## 类图

### 核心类关系

```
┌──────────────────┐
│  ABC             │
│  (抽象基类)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────┐
│  MetaAgentBase                   │◄──────┐
│  ────────────────────────────    │       │
│  + config: Dict                  │       │ 依赖
│  + logger: Logger                │       │
│  ────────────────────────────    │       │
│  + __init__(config)              │       │
│  + process(input) → output       │       │
│  + process_batch(inputs) → list  │       │
│  # _validate_config()            │       │
│  # _log_processing(...)          │       │
└──────────────────────────────────┘       │
         △                                  │
         │ 继承                             │
         │                                  │
    ┌────┴─────┬──────────┬─────────┐     │
    │          │          │         │     │
    ▼          ▼          ▼         │     │
┌────────┐ ┌────────┐ ┌─────────┐  │     │
│Stage   │ │Stage   │ │Inference│  │     │
│One     │ │Two     │ │Agent    │  │     │
│Agent   │ │Agent   │ │         │  │     │
└────────┘ └────────┘ └─────────┘  │     │
    │          │          │         │     │
    └──────────┴──────────┴─────────┘     │
              │                            │
              │ 使用                        │
              ▼                            │
    ┌──────────────────┐                  │
    │ InferenceEngine  │──────────────────┘
    ├──────────────────┤
    │ PromptTemplate   │
    ├──────────────────┤
    │ MemoryManager    │
    └──────────────────┘
```

### 数据类关系

```
┌──────────────────────────┐
│  @dataclass              │
│  ReasoningInput          │
├──────────────────────────┤
│  + question: str         │
│  + answer: str           │
│  + task_description: ?   │
│  + principles: ?         │
│  + metadata: ?           │
└──────────────────────────┘
          │
          │ 处理
          ▼
┌──────────────────────────┐
│  @dataclass              │
│  ReasoningOutput         │
├──────────────────────────┤
│  + question: str         │
│  + baseline_answer: ?    │
│  + diff_analysis: ?      │
│  + principles: ?         │
│  + chosen_answer: ?      │
│  + task_description: ?   │
│  + metadata: ?           │
├──────────────────────────┤
│  + to_dict() → Dict      │
└──────────────────────────┘
```

---

## 时序图

### Stage 1 处理时序

```
用户        run_experiments    StageOneAgent   InferenceEngine   PromptTemplate
 │                │                  │                │               │
 │   运行命令      │                  │                │               │
 │───────────────►│                  │                │               │
 │                │  初始化配置       │                │               │
 │                │──────────────────►│                │               │
 │                │                  │  创建Agent      │               │
 │                │                  │◄───────────────│               │
 │                │                  │                │               │
 │                │  process_batch() │                │               │
 │                │─────────────────►│                │               │
 │                │                  │  获取Prompt     │               │
 │                │                  │────────────────────────────────►│
 │                │                  │  返回Prompt     │               │
 │                │                  │◄────────────────────────────────│
 │                │                  │  batch_inference()             │
 │                │                  │───────────────►│               │
 │                │                  │  vLLM推理      │               │
 │                │                  │◄───────────────│               │
 │                │                  │                │               │
 │                │                  │  (重复多个步骤)                │
 │                │                  │                │               │
 │                │  返回outputs     │                │               │
 │                │◄─────────────────│                │               │
 │                │  save_dpo_format()                │               │
 │                │─────────────────►│                │               │
 │                │                  │  写入文件       │               │
 │   完成          │◄─────────────────│                │               │
 │◄───────────────│                  │                │               │
```

### Memory 检索时序

```
InferenceAgent   MemoryManager   EmbeddingModel
      │                │               │
      │  retrieve()    │               │
      │───────────────►│               │
      │                │  语义匹配      │
      │                │──────────────►│
      │                │  相似度        │
      │                │◄──────────────│
      │  principles    │               │
      │◄───────────────│               │
```

---

## 设计模式应用

### 1. 策略模式 (Strategy Pattern)

```
┌─────────────────┐
│   Context       │
│  (Driver)       │
└────────┬────────┘
         │
         │ 选择策略
         │
    ┌────┴─────┬──────────┬─────────┐
    │          │          │         │
    ▼          ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌─────────┐
│Strategy│ │Strategy│ │Strategy │
│  1     │ │  2     │ │  3      │
└────────┘ └────────┘ └─────────┘
```

### 2. 适配器模式 (Adapter Pattern)

```
┌────────────────┐
│  Client        │
│  (Agent)       │
└───────┬────────┘
        │ 统一接口
        ▼
┌────────────────┐
│  Adapter       │
│  (Engine)      │
└───────┬────────┘
        │
    ┌───┴───┐
    │       │
    ▼       ▼
┌──────┐ ┌──────┐
│vLLM  │ │ API  │
└──────┘ └──────┘
```

### 3. 注册表模式 (Registry Pattern)

```
┌──────────────────┐
│  Registry        │
│  {              │
│   "key1": func1 │
│   "key2": func2 │
│  }              │
└────────┬─────────┘
         │ register()
         │ get()
         ▼
┌──────────────────┐
│  Processor       │
└──────────────────┘
```

---

## 总结

本架构文档展示了 MetaEvo 框架的：
1. **清晰的分层结构**: 应用层 → 业务逻辑层 → 服务层 → 基础设施层
2. **标准化的数据流**: 输入 → 处理 → 输出，每一步都有明确定义
3. **模块化的组件**: 每个组件职责单一，易于理解和维护
4. **优雅的设计模式**: 多种设计模式的综合应用

这种架构设计确保了系统的**可扩展性**、**可维护性**和**可测试性**。
