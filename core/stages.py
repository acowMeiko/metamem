"""
Concrete implementations of meta-reasoning agents.

This module contains specific strategy implementations:
- StageOneAgent: DPO data generation (Baseline → Diff → Principles → Chosen)
- StageTwoAgent: Memory update from DPO data
- InferenceAgent: Inference with memory-guided principles

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

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .base import MetaAgentBase, ReasoningInput, ReasoningOutput
from inference.engine import InferenceEngine
from module.memory_module import MemoryManager
from templates.prompts import PromptTemplate


# ============================================================================
# 常量定义
# ============================================================================

# DPO 格式常量
DPO_INSTRUCTION = "Based on the comparison of high-quality and low-quality answers, generate reusable problem-solving principles."
DPO_OUTPUT_FORMAT = {"output": []}  # 基础输出格式

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


# ============================================================================
# Stage 1: DPO 训练数据生成代理
# ============================================================================

class StageOneAgent(MetaAgentBase):
    """
    Stage 1: DPO Training Data Generation Agent.
    
    职责：生成用于 DPO (Direct Preference Optimization) 训练的数据对
    
    完整流程：
        1. Task Description  - 生成任务描述（用于记忆检索）
        2. Baseline Answer   - 使用弱模型生成基线答案（作为 rejected）
        3. Diff Analysis     - 分析基线与标准答案的差异
        4. Principles        - 从差异中提取解题原则
        5. Chosen Answer     - 使用强模型基于原则生成优质答案（作为 chosen）
    
    输出格式：
        符合 LlamaFactory DPO 训练格式，包含：
        - instruction: 固定指令
        - chosen: 高质量答案的原则 (JSON 字符串)
        - rejected: 基线答案的原则 (JSON 字符串)
        - question: 原始问题
        - diff: 差异分析
        - task_description: 任务描述
    
    增量更新支持：
        - 文件不存在：完整生成所有字段
        - 文件存在：只更新 rejected、diff、task_description（保留 chosen 节省成本）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Stage 1 代理。
        
        Args:
            config: 配置字典，必须包含：
                - inference_engine: InferenceEngine 实例（推理引擎）
                - prompt_template: PromptTemplate 实例（提示词模板）
                - batch_size: 批处理大小（可选，默认 64）
                - max_workers: 并发 API 调用数（可选，默认 20）
        """
        super().__init__(config)
        
        # 核心组件
        self.engine: InferenceEngine = config['inference_engine']
        self.prompts: PromptTemplate = config['prompt_template']
        
        # 性能配置
        self.batch_size: int = config.get('batch_size', DEFAULT_BATCH_SIZE)
        self.max_workers: int = config.get('max_workers', DEFAULT_MAX_WORKERS)
    
    def _validate_config(self) -> None:
        """验证必需的配置项。"""
        required = ['inference_engine', 'prompt_template']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    # ------------------------------------------------------------------------
    # 公共接口方法
    # ------------------------------------------------------------------------
    
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        处理单个输入（便捷方法）。
        
        Args:
            input_data: 必须包含 question 和 answer（标准答案）
            
        Returns:
            包含所有 Stage 1 输出的 ReasoningOutput
        """
        results = self.process_batch([input_data])
        return results[0]
    
    def process_batch(
        self, 
        inputs: List[ReasoningInput],
        skip_chosen: bool = False
    ) -> List[ReasoningOutput]:
        """
        批量处理输入，执行完整的 Stage 1 流水线。
        
        流水线阶段：
            1. 生成任务描述（可选，用于未来的记忆查找）
            2. 生成基线答案（弱模型，作为 rejected）
            3. 分析差异（对比基线与标准答案）
            4. 提取原则（从差异分析中提取）
            5. 生成优质答案（强模型，并发调用，作为 chosen）- 可跳过
        
        Args:
            inputs: 输入列表，每个包含 question 和 answer
            skip_chosen: 是否跳过生成 chosen 答案（增量更新模式时使用）
            
        Returns:
            ReasoningOutput 列表，包含完整的 DPO 数据
        """
        questions = [inp.question for inp in inputs]
        answers = [inp.answer for inp in inputs]
        
        self._log_processing("INIT", f"Processing batch of {len(inputs)} items")
        if skip_chosen:
            self._log_processing("MODE", "增量更新模式: 将跳过 chosen 答案生成（保留现有数据）")
        
        # ===== 阶段 1.1: 生成任务描述 =====
        self._log_processing("TASK_DESC", "Generating task descriptions...")
        task_descs = self._generate_task_descriptions(questions)
        
        # ===== 阶段 1.2: 生成基线答案（rejected）=====
        self._log_processing("BASELINE", "Generating baseline answers...")
        baselines = self._generate_baseline_answers(questions)
        
        # ===== 阶段 1.3: 分析差异 =====
        self._log_processing("DIFF", "Analyzing differences...")
        diffs = self._analyze_differences(questions, baselines, answers)
        
        # ===== 阶段 1.4: 提取原则 =====
        self._log_processing("PRINCIPLES", "Extracting principles...")
        principles = self._extract_principles(questions, diffs)
        
        # ===== 阶段 1.5: 生成优质答案（chosen）=====
        # 增量更新模式下跳过此步骤（保留现有 chosen）
        if skip_chosen:
            self._log_processing("CHOSEN", "Skipping chosen answers generation (incremental mode)")
            chosen_answers = [""] * len(questions)  # 占位符，后续会从现有数据读取
        else:
            self._log_processing("CHOSEN", "Generating chosen answers...")
            chosen_answers = self._generate_chosen_answers(questions, principles)
        
        # ===== 组装输出结果 =====
        outputs = []
        for i, inp in enumerate(inputs):
            output = ReasoningOutput(
                question=inp.question,
                baseline_answer=baselines[i],
                diff_analysis=diffs[i],
                principles=self._parse_principles(principles[i]),
                chosen_answer=chosen_answers[i],
                task_description=task_descs[i],
                metadata={
                    'ground_truth': answers[i],
                    'stage': 'stage1',
                    'skip_chosen': skip_chosen
                }
            )
            outputs.append(output)
        
        self._log_processing("COMPLETE", f"Stage 1 completed for {len(outputs)} items")
        return outputs
    
    # ------------------------------------------------------------------------
    # Stage 1 子流程：模型调用方法
    # ------------------------------------------------------------------------
    
    def _generate_task_descriptions(self, questions: List[str]) -> List[str]:
        """
        生成任务描述（用于记忆系统的语义匹配）。
        
        使用弱模型生成抽象的任务描述，便于后续在记忆中检索相似任务。
        
        Args:
            questions: 问题列表
            
        Returns:
            任务描述列表
        """
        prompts = [self.prompts.get_task_description_prompt(q) for q in questions]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=DEFAULT_MAX_TOKENS_TASK_DESC,
            temperature=0.1
        )
    
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
        prompts = [self.prompts.get_direct_answer_prompt(q) for q in questions]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=DEFAULT_MAX_TOKENS_BASELINE,
            temperature=0.0
        )
    
    def _analyze_differences(
        self, 
        questions: List[str], 
        predictions: List[str], 
        labels: List[str]
    ) -> List[str]:
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
        prompts = [
            self.prompts.get_diff_analysis_prompt(q, p, l)
            for q, p, l in zip(questions, predictions, labels)
        ]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=DEFAULT_MAX_TOKENS_DIFF,
            temperature=0.1
        )
    
    def _extract_principles(self, questions: List[str], diffs: List[str]) -> List[str]:
        """
        从差异分析中提取解题原则。
        
        基于差异分析结果，归纳出可复用的解题原则，
        这些原则将用于指导强模型生成更好的答案。
        
        Args:
            questions: 问题列表
            diffs: 差异分析列表
            
        Returns:
            提取的原则列表（文本格式）
        """
        prompts = [
            self.prompts.get_principle_prompt(q, d)
            for q, d in zip(questions, diffs)
        ]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=DEFAULT_MAX_TOKENS_PRINCIPLES,
            temperature=0.1
        )
    
    def _generate_chosen_answers(self, questions: List[str], principles: List[str]) -> List[str]:
        """
        生成优质答案（将作为 DPO 的 chosen 答案）。
        
        使用强模型（如 GPT-4）在原则指导下生成高质量答案。
        采用并发 API 调用以提高效率。
        
        Args:
            questions: 问题列表
            principles: 指导原则列表
            
        Returns:
            优质答案列表
        """
        prompts = [
            self.prompts.get_guided_answer_prompt(q, p)
            for q, p in zip(questions, principles)
        ]
        return self.engine.concurrent_api_inference(
            prompts=prompts,
            model_type='strong',
            max_workers=self.max_workers,
            max_tokens=DEFAULT_MAX_TOKENS_CHOSEN,
            temperature=0.0
        )
    
    def _parse_principles(self, principle_text: str) -> List[str]:
        """
        解析原则文本为列表。
        
        简单的文本解析：按行分割并清理空白。
        未来可以扩展支持更复杂的格式（JSON、Markdown 等）。
        
        Args:
            principle_text: 原则文本（多行）
            
        Returns:
            原则列表
        """
        lines = principle_text.strip().split('\n')
        principles = [line.strip() for line in lines if line.strip()]
        return principles
    
    # ------------------------------------------------------------------------
    # DPO 数据保存：支持完整生成和增量更新
    # ------------------------------------------------------------------------
    
    def save_dpo_format(self, outputs: List[ReasoningOutput], output_path: Path) -> None:
        """
        保存为 DPO 格式（兼容 LlamaFactory）。
        
        标准格式（与 dpo_level1.json 一致）：
        {
            "instruction": "Based on the comparison...",
            "chosen": "{\"output\": [{\"Principle\": \"...\"}]}",
            "rejected": "{\"output\": [{\"Principle\": \"...\"}]}",
            "question": "...",
            "diff": "...",
            "task_description": "..."
        }
        
        智能模式选择：
            - 文件不存在或为空 → 完整生成所有字段
            - 文件存在且非空   → 增量更新（保留 chosen，更新 rejected/diff/task_description）
        
        增量更新的好处：
            - 保留 chosen（来自强模型，成本高）
            - 只更新可能变化的字段（节省 API 调用）
        
        Args:
            outputs: ReasoningOutput 列表
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ===== 检测文件状态，决定生成模式 =====
        existing_data = []
        is_incremental = False
        
        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                if isinstance(existing_data, list) and len(existing_data) > 0:
                    is_incremental = True
                    self.logger.info(f"检测到现有 DPO 文件包含 {len(existing_data)} 条数据，启用增量更新模式")
                    self.logger.info("将更新: task_description, rejected 中的 principle, diff")
                    self.logger.info("将保留: chosen（来自强模型，避免重复调用）")
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"无法读取现有文件，将完整生成: {e}")
                is_incremental = False
        
        # ===== 根据模式生成数据 =====
        if is_incremental:
            dpo_data = self._incremental_update(existing_data, outputs)
            self.logger.info(f"✓ 增量更新完成: 更新了 {len(dpo_data)} 条数据")
        else:
            dpo_data = self._generate_full_dpo_data(outputs)
            self.logger.info(f"✓ 完整生成完成: 创建了 {len(dpo_data)} 条新数据")
        
        # ===== 保存为 JSON =====
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"DPO 数据已保存到 {output_path}")
    
    def _extract_principles_to_json(self, principles_list: List[str]) -> str:
        """
        将原则列表转换为标准 JSON 字符串格式。
        
        输入: ["Principle 1", "Principle 2"]
        输出: '{"output": [{"Principle": "Principle 1"}, {"Principle": "Principle 2"}]}'
        
        Args:
            principles_list: 原则列表
            
        Returns:
            JSON 字符串（符合 DPO 格式）
        """
        if not principles_list:
            return json.dumps(DPO_OUTPUT_FORMAT, ensure_ascii=False)
        
        output_list = [
            {"Principle": p.strip()} 
            for p in principles_list 
            if p.strip()
        ]
        return json.dumps({"output": output_list}, ensure_ascii=False)
    
    def _incremental_update(
        self, 
        existing_data: List[Dict], 
        outputs: List[ReasoningOutput]
    ) -> List[Dict]:
        """
        增量更新模式：只更新特定字段。
        
        保留字段（不更新）：
            - instruction: 固定文本
            - chosen: 来自强模型，成本高，通常不需要更新
            - question: 原始问题，不会改变
        
        更新字段：
            - rejected: 基线模型可能改进，需要重新提取
            - diff: 差异分析可能优化
            - task_description: 任务描述可能改进
        
        Args:
            existing_data: 现有 DPO 数据
            outputs: 新生成的输出
            
        Returns:
            更新后的 DPO 数据
        """
        # 数据长度校验
        if len(existing_data) != len(outputs):
            self.logger.warning(
                f"数据长度不匹配: 现有 {len(existing_data)} 条，新生成 {len(outputs)} 条。"
                f"将按照最小长度进行更新。"
            )
        
        updated_data = []
        min_len = min(len(existing_data), len(outputs))
        
        # 逐项更新
        for i in range(min_len):
            existing_item = existing_data[i]
            new_output = outputs[i]
            
            # 保留现有字段
            updated_item = {
                "instruction": existing_item.get("instruction", DPO_INSTRUCTION),
                "chosen": existing_item.get("chosen", ""),
                "question": existing_item.get("question", new_output.question),
            }
            
            # 更新字段
            rejected_principles = self._parse_principles(new_output.baseline_answer or "")
            updated_item["rejected"] = self._extract_principles_to_json(rejected_principles)
            updated_item["diff"] = new_output.diff_analysis or ""
            updated_item["task_description"] = new_output.task_description or ""
            
            updated_data.append(updated_item)
        
        # 保留额外的现有数据（如果有）
        if len(existing_data) > min_len:
            extra_count = len(existing_data) - min_len
            self.logger.info(f"保留额外的 {extra_count} 条现有数据")
            updated_data.extend(existing_data[min_len:])
        
        return updated_data
    
    def _generate_full_dpo_data(self, outputs: List[ReasoningOutput]) -> List[Dict]:
        """
        完整生成模式：生成所有 DPO 数据字段。
        
        用于首次生成或完全重建数据。
        
        Args:
            outputs: ReasoningOutput 列表
            
        Returns:
            完整的 DPO 数据列表
        """
        dpo_data = []
        
        for output in outputs:
            # 提取并转换原则
            chosen_principles = output.principles or []
            rejected_principles = self._parse_principles(output.baseline_answer or "")
            
            chosen_json = self._extract_principles_to_json(chosen_principles)
            rejected_json = self._extract_principles_to_json(rejected_principles)
            
            # 组装 DPO 数据项
            dpo_item = {
                "instruction": DPO_INSTRUCTION,
                "chosen": chosen_json,
                "rejected": rejected_json,
                "question": output.question or "",
                "diff": output.diff_analysis or "",
                "task_description": output.task_description or ""
            }
            dpo_data.append(dpo_item)
        
        return dpo_data


# ============================================================================
# Stage 2: 记忆系统更新代理
# ============================================================================

class StageTwoAgent(MetaAgentBase):
    """
    Stage 2: Memory Update Agent.
    
    职责：从 DPO 数据中提取原则并更新记忆系统
    
    工作流程：
        1. 语义匹配 - 检查任务是否已存在于记忆中
        2. 合并原则 - 如果存在，合并新原则
        3. 添加任务 - 如果不存在，添加新任务
        4. 定期保存 - 每处理 N 条数据保存一次
    
    记忆系统特性：
        - 使用语义匹配识别相似任务（避免重复）
        - 智能合并原则（去重、冲突解决）
        - 增量更新（不影响已有数据）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Stage 2 代理。
        
        Args:
            config: 配置字典，必须包含：
                - memory_manager: MemoryManager 实例（记忆管理器）
                - save_frequency: 保存频率（可选，默认每 50 条保存一次）
        """
        super().__init__(config)
        
        # 核心组件
        self.memory: MemoryManager = config['memory_manager']
        
        # 保存策略
        self.save_frequency: int = config.get('save_frequency', DEFAULT_SAVE_FREQUENCY)
    
    def _validate_config(self) -> None:
        """验证必需的配置项。"""
        if 'memory_manager' not in self.config:
            raise ValueError("Missing required config: memory_manager")
    
    # ------------------------------------------------------------------------
    # 公共接口方法
    # ------------------------------------------------------------------------
    
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        处理单个输入（便捷方法）。
        
        Args:
            input_data: 必须包含 task_description 和 principles
            
        Returns:
            包含记忆更新状态的 ReasoningOutput
        """
        results = self.process_batch([input_data])
        return results[0]
    
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        批量处理输入，更新记忆系统（优化版：使用批量推理）。
        
        处理逻辑：
            1. 第一步：收集所有 task_description
            2. 第二步：批量语义匹配（使用 vLLM 批量推理）
            3. 第三步：批量更新 memory
                - 有匹配：批量解决冲突
                - 无匹配：直接添加
            4. 保存记忆
        
        Args:
            inputs: 输入列表，每个包含 task_description 和 principles
            
        Returns:
            ReasoningOutput 列表（透传输入，附加记忆状态信息）
        """
        # ===== 第一步：数据验证与收集 =====
        valid_inputs = []
        valid_indices = []
        outputs = [None] * len(inputs)
        
        for i, inp in enumerate(inputs):
            if not inp.task_description or not inp.principles:
                self._log_processing("SKIP", f"Item {i}: Missing task_description or principles")
                outputs[i] = ReasoningOutput(
                    question=inp.question,
                    metadata={
                        'memory_status': 'skipped', 
                        'reason': 'missing_data'
                    }
                )
            else:
                valid_inputs.append(inp)
                valid_indices.append(i)
        
        if not valid_inputs:
            self._log_processing("COMPLETE", "No valid inputs to process")
            return outputs
        
        # ===== 第二步：批量语义匹配 =====
        self._log_processing("MATCH", f"Batch matching {len(valid_inputs)} tasks...")
        task_descs = [inp.task_description for inp in valid_inputs]
        match_results = self.memory.batch_retrieve(task_descs)
        
        # ===== 第三步：批量更新 memory =====
        # 分类：需要合并的 vs 需要添加的
        to_merge = []  # [(matched_task, new_principles, input_idx)]
        to_add = []    # [(task_desc, principles, input_idx)]
        
        for i, (input_task, matched_task, existing_principles) in enumerate(match_results):
            original_idx = valid_indices[i]
            inp = valid_inputs[i]
            
            if matched_task:
                to_merge.append((matched_task, existing_principles, inp.principles, original_idx))
            else:
                to_add.append((input_task, inp.principles, original_idx))
        
        # 3.1 批量解决冲突（有匹配的任务）
        if to_merge:
            self._log_processing("MERGE", f"Batch resolving conflicts for {len(to_merge)} tasks...")
            conflict_pairs = [(old_prins, new_prins) for _, old_prins, new_prins, _ in to_merge]
            resolved_principles = self.memory.batch_resolve_conflicts(conflict_pairs)
            
            # 更新 memory 并记录结果
            for idx, (matched_task, _, _, original_idx) in enumerate(to_merge):
                self.memory.memory[matched_task] = resolved_principles[idx]
                outputs[original_idx] = ReasoningOutput(
                    question=valid_inputs[valid_indices.index(original_idx)].question,
                    task_description=valid_inputs[valid_indices.index(original_idx)].task_description,
                    principles=valid_inputs[valid_indices.index(original_idx)].principles,
                    metadata={
                        'memory_status': 'merged',
                        'matched_task': matched_task
                    }
                )
        
        # 3.2 批量添加新任务（无匹配的）
        if to_add:
            self._log_processing("ADD", f"Batch adding {len(to_add)} new tasks...")
            for task_desc, principles, original_idx in to_add:
                self.memory.add_task(task_desc, principles)
                outputs[original_idx] = ReasoningOutput(
                    question=valid_inputs[valid_indices.index(original_idx)].question,
                    task_description=task_desc,
                    principles=principles,
                    metadata={
                        'memory_status': 'added',
                        'matched_task': None
                    }
                )
        
        # ===== 最终保存 =====
        self.memory.save()
        self._log_processing("COMPLETE", f"Memory updated: {len(to_merge)} merged, {len(to_add)} added")
        
        return outputs


# ============================================================================
# Inference: 基于记忆的推理代理
# ============================================================================

class InferenceAgent(MetaAgentBase):
    """
    Inference Agent with Memory-Guided Principles.
    
    职责：使用记忆系统中的原则指导推理
    
    工作流程：
        1. 任务识别 - 生成任务描述
        2. 记忆检索 - 查找相关原则
        3. 引导推理 - 使用原则指导模型生成答案
    
    推理模式：
        - 有原则：使用原则引导（guided inference）
        - 无原则：直接推理（direct inference）
    
    优势：
        - 利用历史经验（从记忆中学习）
        - 提高答案质量（基于最佳实践）
        - 保持一致性（复用成功策略）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化推理代理。
        
        Args:
            config: 配置字典，必须包含：
                - inference_engine: InferenceEngine 实例（推理引擎）
                - prompt_template: PromptTemplate 实例（提示词模板）
                - memory_manager: MemoryManager 实例（记忆管理器）
                - batch_size: 批处理大小（可选，默认 64）
        """
        super().__init__(config)
        
        # 核心组件
        self.engine: InferenceEngine = config['inference_engine']
        self.prompts: PromptTemplate = config['prompt_template']
        self.memory: MemoryManager = config['memory_manager']
        
        # 性能配置
        self.batch_size: int = config.get('batch_size', DEFAULT_BATCH_SIZE)
    
    def _validate_config(self) -> None:
        """验证必需的配置项。"""
        required = ['inference_engine', 'prompt_template', 'memory_manager']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    # ------------------------------------------------------------------------
    # 公共接口方法
    # ------------------------------------------------------------------------
    
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        处理单个推理请求（便捷方法）。
        
        Args:
            input_data: 包含问题的输入
            
        Returns:
            包含答案的 ReasoningOutput
        """
        results = self.process_batch([input_data])
        return results[0]
    
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        批量处理推理请求，使用记忆指导。
        
        完整流程：
            1. 生成任务描述（识别问题类型）
            2. 从记忆中检索相关原则
            3. 执行引导推理（如果有原则）或直接推理（如果没有）
        
        Args:
            inputs: 输入列表，每个包含问题
            
        Returns:
            ReasoningOutput 列表，包含最终答案
        """
        questions = [inp.question for inp in inputs]
        
        self._log_processing("INIT", f"Inference batch of {len(inputs)} items")
        
        # ===== 阶段 1: 生成任务描述 =====
        self._log_processing("TASK_DESC", "Generating task descriptions...")
        task_desc_prompts = [
            self.prompts.get_task_description_prompt(q) 
            for q in questions
        ]
        task_descs = self.engine.batch_inference(
            prompts=task_desc_prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=DEFAULT_MAX_TOKENS_TASK_DESC
        )
        
        # ===== 阶段 2: 从记忆中检索原则 =====
        self._log_processing("MEMORY", "Retrieving principles from memory...")
        principles_list = []
        for task_desc in task_descs:
            _, principles = self.memory.retrieve(task_desc)
            principles_list.append(principles)
        
        # ===== 阶段 3: 生成答案（引导 or 直接）=====
        self._log_processing("INFERENCE", "Generating answers...")
        outputs = []
        
        for i, (question, task_desc, principles) in enumerate(
            zip(questions, task_descs, principles_list)
        ):
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
            
            # 执行推理
            answer = self.engine.single_inference(
                prompt=prompt,
                model_type='weak',
                max_tokens=DEFAULT_MAX_TOKENS_INFERENCE
            )
            
            # 记录结果
            outputs.append(ReasoningOutput(
                question=question,
                chosen_answer=answer,
                task_description=task_desc,
                principles=principles,
                metadata={
                    'inference_type': inference_type,
                    'has_principles': len(principles) > 0
                }
            ))
        
        self._log_processing("COMPLETE", f"Inference completed for {len(outputs)} items")
        return outputs
