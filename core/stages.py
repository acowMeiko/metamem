"""
Concrete implementations of meta-reasoning agents.

This module contains specific strategy implementations:
- StageOneAgent: DPO data generation (Baseline → Diff → Principles → Chosen)
- StageTwoAgent: Memory update from DPO data
- InferenceAgent: Inference with memory-guided principles
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .base import MetaAgentBase, ReasoningInput, ReasoningOutput
from inference.engine import InferenceEngine
from module.memory_module import MemoryManager
from templates.prompts import PromptTemplate


class StageOneAgent(MetaAgentBase):
    """
    Stage 1: DPO Training Data Generation Agent.
    
    Flow: Baseline → Diff Analysis → Principles (weak) → Chosen (strong) → DPO Format
    
    This agent generates DPO training data by:
    1. Generating baseline answers with the weak model
    2. Analyzing differences between baseline and ground truth
    3. Extracting principles from differences
    4. Generating chosen answers with strong model guided by principles
    5. Formatting as DPO data (rejected=baseline, chosen=strong answer)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Stage 1 agent.
        
        Args:
            config: Must contain:
                - inference_engine: InferenceEngine instance
                - prompt_template: PromptTemplate instance
                - batch_size: Batch size for inference
                - max_workers: Concurrent workers for API calls
        """
        super().__init__(config)
        self.engine = config['inference_engine']
        self.prompts = config['prompt_template']
        self.batch_size = config.get('batch_size', 64)
        self.max_workers = config.get('max_workers', 20)
    
    def _validate_config(self) -> None:
        """Validate required configuration."""
        required = ['inference_engine', 'prompt_template']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        Process a single input through Stage 1.
        
        Args:
            input_data: Must contain question and answer (ground truth)
            
        Returns:
            ReasoningOutput with all Stage 1 outputs
        """
        results = self.process_batch([input_data])
        return results[0]
    
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        Process a batch of inputs through Stage 1 pipeline.
        
        Pipeline stages:
        1. Generate task descriptions (optional, for memory)
        2. Generate baseline answers (weak model)
        3. Analyze differences (weak model)
        4. Extract principles (weak model)
        5. Generate chosen answers (strong model, concurrent)
        
        Args:
            inputs: List of inputs with question and answer
            
        Returns:
            List of ReasoningOutput with DPO data
        """
        questions = [inp.question for inp in inputs]
        answers = [inp.answer for inp in inputs]
        
        self._log_processing("INIT", f"Processing batch of {len(inputs)} items")
        
        # Stage 1.1: Generate task descriptions (for future memory lookup)
        self._log_processing("TASK_DESC", "Generating task descriptions...")
        task_descs = self._generate_task_descriptions(questions)
        
        # Stage 1.2: Generate baseline answers (rejected answers)
        self._log_processing("BASELINE", "Generating baseline answers...")
        baselines = self._generate_baseline_answers(questions)
        
        # Stage 1.3: Analyze differences
        self._log_processing("DIFF", "Analyzing differences...")
        diffs = self._analyze_differences(questions, baselines, answers)
        
        # Stage 1.4: Extract principles from diffs
        self._log_processing("PRINCIPLES", "Extracting principles...")
        principles = self._extract_principles(questions, diffs)
        
        # Stage 1.5: Generate chosen answers (strong model)
        self._log_processing("CHOSEN", "Generating chosen answers...")
        chosen_answers = self._generate_chosen_answers(questions, principles)
        
        # Assemble outputs
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
                    'stage': 'stage1'
                }
            )
            outputs.append(output)
        
        self._log_processing("COMPLETE", f"Stage 1 completed for {len(outputs)} items")
        return outputs
    
    def _generate_task_descriptions(self, questions: List[str]) -> List[str]:
        """Generate task descriptions using the weak model."""
        prompts = [self.prompts.get_task_description_prompt(q) for q in questions]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=2560,
            temperature=0.1
        )
    
    def _generate_baseline_answers(self, questions: List[str]) -> List[str]:
        """Generate baseline (rejected) answers using the weak model."""
        prompts = [self.prompts.get_direct_answer_prompt(q) for q in questions]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=2048,
            temperature=0.0
        )
    
    def _analyze_differences(
        self, 
        questions: List[str], 
        predictions: List[str], 
        labels: List[str]
    ) -> List[str]:
        """Analyze differences between baseline and ground truth."""
        prompts = [
            self.prompts.get_diff_analysis_prompt(q, p, l)
            for q, p, l in zip(questions, predictions, labels)
        ]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=1024,
            temperature=0.1
        )
    
    def _extract_principles(self, questions: List[str], diffs: List[str]) -> List[str]:
        """Extract principles from difference analysis."""
        prompts = [
            self.prompts.get_principle_prompt(q, d)
            for q, d in zip(questions, diffs)
        ]
        return self.engine.batch_inference(
            prompts=prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=2560,
            temperature=0.1
        )
    
    def _generate_chosen_answers(self, questions: List[str], principles: List[str]) -> List[str]:
        """Generate chosen answers using strong model with concurrent API calls."""
        prompts = [
            self.prompts.get_guided_answer_prompt(q, p)
            for q, p in zip(questions, principles)
        ]
        return self.engine.concurrent_api_inference(
            prompts=prompts,
            model_type='strong',
            max_workers=self.max_workers,
            max_tokens=4096,
            temperature=0.0
        )
    
    def _parse_principles(self, principle_text: str) -> List[str]:
        """Parse principles from model output."""
        # Simple parsing - extract lines or split by markers
        lines = principle_text.strip().split('\n')
        principles = [line.strip() for line in lines if line.strip()]
        return principles
    
    def save_dpo_format(self, outputs: List[ReasoningOutput], output_path: Path) -> None:
        """
        Save outputs in DPO format (for LlamaFactory).
        
        DPO Format:
        {
            "input": "Question: ... Error Analysis: ...",
            "rejected": "baseline_answer",
            "chosen": "chosen_answer"
        }
        """
        dpo_data = []
        for output in outputs:
            # Format input with question and diff
            input_text = f"Question: {output.question}\n\nError Analysis:\n{output.diff_analysis}"
            
            dpo_item = {
                "input": input_text,
                "rejected": output.baseline_answer or "",
                "chosen": output.chosen_answer or ""
            }
            dpo_data.append(dpo_item)
        
        # Save as JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(dpo_data)} DPO items to {output_path}")


class StageTwoAgent(MetaAgentBase):
    """
    Stage 2: Memory Update Agent.
    
    This agent updates the memory system with principles extracted from DPO data.
    It performs semantic matching to merge similar tasks and avoid redundancy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Stage 2 agent.
        
        Args:
            config: Must contain:
                - memory_manager: MemoryManager instance
                - save_frequency: How often to save memory (default: 50)
        """
        super().__init__(config)
        self.memory = config['memory_manager']
        self.save_frequency = config.get('save_frequency', 50)
    
    def _validate_config(self) -> None:
        """Validate required configuration."""
        if 'memory_manager' not in self.config:
            raise ValueError("Missing required config: memory_manager")
    
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """Process a single input for memory update."""
        results = self.process_batch([input_data])
        return results[0]
    
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        Process a batch of inputs to update memory.
        
        For each input:
        1. Check if task exists in memory (semantic matching)
        2. If exists: merge principles
        3. If not: add new task
        
        Args:
            inputs: List of inputs with task_description and principles
            
        Returns:
            List of ReasoningOutput (pass-through with memory status)
        """
        outputs = []
        
        for i, inp in enumerate(tqdm(inputs, desc="Updating Memory")):
            if not inp.task_description or not inp.principles:
                self._log_processing("SKIP", f"Item {i}: Missing task_description or principles")
                outputs.append(ReasoningOutput(
                    question=inp.question,
                    metadata={'memory_status': 'skipped', 'reason': 'missing_data'}
                ))
                continue
            
            # Semantic matching
            matched_task, existing_principles = self.memory.retrieve(inp.task_description)
            
            if matched_task:
                # Merge with existing principles
                self._log_processing("MERGE", f"Item {i}: Merging with task '{matched_task}'")
                self.memory.merge_principles(matched_task, inp.principles)
                status = 'merged'
            else:
                # Add as new task
                self._log_processing("ADD", f"Item {i}: Adding new task")
                self.memory.add_task(inp.task_description, inp.principles)
                status = 'added'
            
            # Periodic save
            if (i + 1) % self.save_frequency == 0:
                self.memory.save()
                self._log_processing("SAVE", f"Saved memory at item {i+1}")
            
            outputs.append(ReasoningOutput(
                question=inp.question,
                task_description=inp.task_description,
                principles=inp.principles,
                metadata={
                    'memory_status': status,
                    'matched_task': matched_task
                }
            ))
        
        # Final save
        self.memory.save()
        self._log_processing("COMPLETE", f"Memory updated with {len(outputs)} items")
        
        return outputs


class InferenceAgent(MetaAgentBase):
    """
    Inference Agent with Memory-Guided Principles.
    
    This agent performs inference using principles retrieved from memory.
    Flow: Task Description → Memory Retrieval → Guided Inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Inference agent.
        
        Args:
            config: Must contain:
                - inference_engine: InferenceEngine instance
                - prompt_template: PromptTemplate instance
                - memory_manager: MemoryManager instance
                - batch_size: Batch size for inference
        """
        super().__init__(config)
        self.engine = config['inference_engine']
        self.prompts = config['prompt_template']
        self.memory = config['memory_manager']
        self.batch_size = config.get('batch_size', 64)
    
    def _validate_config(self) -> None:
        """Validate required configuration."""
        required = ['inference_engine', 'prompt_template', 'memory_manager']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """Process a single inference request."""
        results = self.process_batch([input_data])
        return results[0]
    
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        Process a batch of inference requests with memory guidance.
        
        Pipeline:
        1. Generate task descriptions
        2. Retrieve relevant principles from memory
        3. Perform guided inference with principles
        
        Args:
            inputs: List of inputs with questions
            
        Returns:
            List of ReasoningOutput with final answers
        """
        questions = [inp.question for inp in inputs]
        
        self._log_processing("INIT", f"Inference batch of {len(inputs)} items")
        
        # Step 1: Generate task descriptions
        self._log_processing("TASK_DESC", "Generating task descriptions...")
        task_desc_prompts = [self.prompts.get_task_description_prompt(q) for q in questions]
        task_descs = self.engine.batch_inference(
            prompts=task_desc_prompts,
            model_type='weak',
            batch_size=self.batch_size,
            max_tokens=2560
        )
        
        # Step 2: Retrieve principles from memory
        self._log_processing("MEMORY", "Retrieving principles from memory...")
        principles_list = []
        for task_desc in task_descs:
            _, principles = self.memory.retrieve(task_desc)
            principles_list.append(principles)
        
        # Step 3: Generate answers (with or without principles)
        self._log_processing("INFERENCE", "Generating answers...")
        outputs = []
        
        for i, (question, task_desc, principles) in enumerate(zip(questions, task_descs, principles_list)):
            if principles:
                # Guided inference with principles
                principles_text = "\n".join(f"- {p}" for p in principles)
                prompt = self.prompts.get_guided_answer_prompt(question, principles_text)
                inference_type = "guided"
            else:
                # Direct inference without principles
                prompt = self.prompts.get_direct_answer_prompt(question)
                inference_type = "direct"
            
            answer = self.engine.single_inference(
                prompt=prompt,
                model_type='weak',
                max_tokens=2048
            )
            
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
