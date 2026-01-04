"""
MetaEvo Framework - Main Entry Point (Refactored Architecture)

This is the new main entry point following memr3-style architecture.
Features:
- Clean separation of concerns
- Data/logic decoupling
- Strategy pattern for different stages
- Centralized configuration
- Type-safe interfaces

Usage:
    python run_experiments.py --stage 1 --dataset gsm8k --dataset-path dataset/gsm8k/test.jsonl
    python run_experiments.py --stage 2
    python run_experiments.py --stage 3 --dataset gsm8k --dataset-path dataset/gsm8k/test.jsonl
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json
import re

from core.config import MetaConfig, initialize_config, get_config
from core.base import ReasoningInput
from core.stages import StageOneAgent, StageTwoAgent, InferenceAgent
from data.processor import DatasetProcessor
from inference.engine import InferenceEngineBuilder
from templates.prompts import PromptTemplate
from module.memory_module import MemoryManager

logger = logging.getLogger(__name__)


def setup_inference_engine(config: MetaConfig):
    """
    Setup the inference engine from configuration.
    
    Args:
        config: MetaConfig instance
        
    Returns:
        InferenceEngine instance
    """
    builder = InferenceEngineBuilder()
    
    # Configure weak model
    builder.set_weak_model(
        model_type=config.models.weak_model_type,
        name=config.models.weak_model_name,
        url=config.models.weak_model_url,
        api_key=config.models.weak_model_key
    )
    
    # Configure strong model
    builder.set_strong_model(
        model_type=config.models.strong_model_type,
        name=config.models.strong_model_name,
        url=config.models.strong_model_url,
        api_key=config.models.strong_model_key
    )
    
    # Set defaults
    builder.set_defaults(
        temperature=config.inference.default_temperature,
        top_p=config.inference.default_top_p,
        max_tokens=config.inference.default_max_tokens
    )
    
    return builder.build()


def run_stage1(
    dataset_name: str,
    dataset_path: str,
    output_path: Optional[str] = None
) -> None:
    """
    Run Stage 1: DPO Training Data Generation.
    
    Args:
        dataset_name: Name of the dataset (gsm8k, math, bbh, etc.)
        dataset_path: Path to the dataset file
        output_path: Optional output path for DPO data
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DPO Training Data Generation")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load and preprocess dataset
    processor = DatasetProcessor()
    data = processor.load_dataset(dataset_name, dataset_path)
    logger.info(f"Loaded {len(data)} items from {dataset_name}")
    
    # Convert to ReasoningInput format
    inputs = [
        ReasoningInput(
            question=item['question'],
            answer=item['answer'],
            metadata={'dataset': dataset_name}
        )
        for item in data
    ]
    
    # Setup components
    engine = setup_inference_engine(config)
    prompts = PromptTemplate()
    
    # Create Stage 1 agent
    agent_config = {
        'inference_engine': engine,
        'prompt_template': prompts,
        'batch_size': config.inference.batch_size,
        'max_workers': config.inference.max_workers
    }
    agent = StageOneAgent(agent_config)
    
    # Process data
    logger.info("Starting Stage 1 processing...")
    outputs = agent.process_batch(inputs)
    
    # Save DPO format
    if output_path is None:
        output_path = config.paths.output_dir / f"dpo_{dataset_name}.json"
    else:
        output_path = Path(output_path)
    
    agent.save_dpo_format(outputs, output_path)
    logger.info(f"Stage 1 completed. Output saved to {output_path}")


def run_stage2(dpo_file: Optional[str] = None) -> None:
    """
    Run Stage 2: Memory Update from DPO Data.
    
    Args:
        dpo_file: Path to DPO data file (optional, uses default if not provided)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Memory Update from DPO Data")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Determine DPO file path
    if dpo_file is None:
        # Use default from data directory
        dpo_file = config.paths.data_dir / 'dpo_llamafactory' / 'dpo_level1.json'
    else:
        dpo_file = Path(dpo_file)
    
    if not dpo_file.exists():
        logger.error(f"DPO file not found: {dpo_file}")
        return
    
    # Load DPO data
    with open(dpo_file, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    logger.info(f"Loaded {len(dpo_data)} DPO items")
    
    if not isinstance(dpo_data, list):
        logger.error(f"数据格式错误: 期望列表格式，实际为 {type(dpo_data)}")
        return
    
    # ===== 阶段1: 数据预处理和过滤 =====
    logger.info("阶段1/3: 数据预处理和过滤")
    batch_data = []
    for i, item in enumerate(dpo_data):
        # 直接从数据集提取 question 和 diff 字段
        question = item.get("question")
        diff = item.get("diff")
        
        if not question or not diff:
            logger.warning(f"第 {i} 项数据缺少 question 或 diff，跳过")
            continue
        
        batch_data.append({
            'index': i,
            'question': question,
            'diff': diff,
            'chosen': item.get('chosen', '')
        })
    
    logger.info(f"准备批处理 {len(batch_data)} 条数据")
    
    # ===== 阶段2: 批量生成任务描述 =====
    logger.info("阶段2/3: 批量生成任务描述")
    
    # Setup inference engine for task description generation
    engine = setup_inference_engine(config)
    prompts = PromptTemplate()
    
    # 批量生成任务描述
    questions = [item['question'] for item in batch_data]
    
    logger.info(f"批量生成 {len(questions)} 个任务描述...")
    task_desc_prompts = [prompts.get_task_description_prompt(q) for q in questions]
    task_descs_raw = engine.batch_inference(
        prompts=task_desc_prompts,
        model_type='weak',
        batch_size=config.inference.batch_size,
        max_tokens=config.inference.task_desc_max_tokens,
        temperature=0.1
    )
    
    # 从 chosen 字段提取原则
    logger.info("提取 chosen 原则...")
    regenerated_list = [item['chosen'] for item in batch_data]
    
    # 保存中间生成结果
    intermediate_output_file = config.paths.output_dir / "stage_generated.json"
    logger.info(f"保存中间生成结果到: {intermediate_output_file}")
    
    def clean_markdown(text):
        """清理 markdown 代码块标记"""
        text_clean = text.strip()
        if text_clean.startswith('```json'):
            text_clean = text_clean[7:]
        elif text_clean.startswith('```'):
            text_clean = text_clean[3:]
        if text_clean.endswith('```'):
            text_clean = text_clean[:-3]
        return text_clean.strip()
    
    intermediate_data = []
    for i, (item, td, rl) in enumerate(zip(batch_data, task_descs_raw, regenerated_list)):
        intermediate_data.append({
            "index": item['index'],
            "question": item['question'],
            "diff": item['diff'],
            "task_desc": clean_markdown(td),
            "regenerated_principles": clean_markdown(rl)
        })
    
    intermediate_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(intermediate_output_file, 'w', encoding='utf-8') as f:
        json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
    
    # ===== 阶段3: 解析结果并构建 ReasoningInput =====
    logger.info("阶段3/3: 解析结果并更新 Memory")
    
    def parse_task_description(task_desc_raw):
        """解析任务描述 JSON"""
        task_desc_clean = clean_markdown(task_desc_raw)
        
        try:
            task_obj = json.loads(task_desc_clean)
            return task_obj.get("taskDescription", {}).get("description")
        except (json.JSONDecodeError, KeyError, AttributeError):
            # 尝试正则提取
            pattern = r'\{\s*"taskDescription"\s*:\s*\{\s*"description"\s*:\s*"([^"]+)"\s*\}\s*\}'
            matches = list(re.finditer(pattern, task_desc_raw, re.DOTALL))
            if matches:
                json_str = matches[-1].group(0)
                task_obj = json.loads(json_str)
                return task_obj.get("taskDescription", {}).get("description")
        return None
    
    def parse_principles(regenerated_raw):
        """解析原则 JSON"""
        regenerated_clean = clean_markdown(regenerated_raw)
        
        try:
            principles_obj = json.loads(regenerated_clean)
            output_list = principles_obj.get("output", [])
            return [x.get("Principle") for x in output_list 
                   if isinstance(x, dict) and "Principle" in x]
        except (json.JSONDecodeError, KeyError, AttributeError):
            # 尝试正则提取
            json_match = re.search(
                r'\{\s*"output"\s*:\s*\[[^\]]*?\{[^}]*?"Principle"[^}]*?\}[^\]]*?\]\s*\}',
                regenerated_raw, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
                principles_obj = json.loads(json_str)
                output_list = principles_obj.get("output", [])
                return [x.get("Principle") for x in output_list 
                       if isinstance(x, dict) and "Principle" in x]
        return []
    
    # 解析并构建输入
    inputs = []
    for i, (item, task_desc_raw, regenerated_raw) in enumerate(zip(batch_data, task_descs_raw, regenerated_list)):
        task_desc = parse_task_description(task_desc_raw)
        principles = parse_principles(regenerated_raw)
        
        if not task_desc:
            logger.warning(f"第 {item['index']} 项无法解析任务描述，跳过")
            continue
        
        if not principles:
            logger.warning(f"第 {item['index']} 项无法解析原则，跳过")
            continue
        
        inputs.append(ReasoningInput(
            question=item['question'],
            answer="",
            task_description=task_desc,
            principles=principles,
            metadata={'dpo_file': str(dpo_file), 'index': item['index']}
        ))
    
    logger.info(f"成功解析 {len(inputs)} 个有效项")
    
    # 保存完整的生成内容到 output
    full_results_file = config.paths.output_dir / f"stage2_full_results_{Path(dpo_file).stem}.json"
    logger.info(f"保存完整生成结果到: {full_results_file}")
    
    full_results = []
    for i, (item, task_desc_raw, regenerated_raw, reasoning_input) in enumerate(
        zip(batch_data, task_descs_raw, regenerated_list, inputs)
    ):
        full_results.append({
            "index": item['index'],
            "question": item['question'],
            "diff": item['diff'],
            "chosen": item['chosen'],
            "task_description_raw": task_desc_raw,
            "task_description_parsed": reasoning_input.task_description,
            "principles_raw": regenerated_raw,
            "principles_parsed": reasoning_input.principles,
            "metadata": reasoning_input.metadata
        })
    
    full_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 已保存 {len(full_results)} 条完整记录")
    
    # # Setup memory manager
    # memory = MemoryManager(path=str(config.paths.memory_file))
    
    # # Create Stage 2 agent
    # agent_config = {
    #     'memory_manager': memory,
    #     'save_frequency': config.memory.save_frequency
    # }
    # agent = StageTwoAgent(agent_config)
    
    # # Process data
    # logger.info("更新 Memory...")
    # outputs = agent.process_batch(inputs)
    
    # # 保存最终的 Memory 更新结果
    # memory_update_results_file = config.paths.output_dir / f"stage2_memory_updates_{Path(dpo_file).stem}.json"
    # logger.info(f"保存 Memory 更新结果到: {memory_update_results_file}")
    
    # memory_updates = []
    # for inp, out in zip(inputs, outputs):
    #     memory_updates.append({
    #         "question": inp.question,
    #         "task_description": out.task_description,
    #         "principles": out.principles,
    #         "memory_status": out.metadata.get('memory_status'),
    #         "matched_task": out.metadata.get('matched_task'),
    #         "index": inp.metadata.get('index')
    #     })
    
    # with open(memory_update_results_file, 'w', encoding='utf-8') as f:
    #     json.dump(memory_updates, f, indent=2, ensure_ascii=False)
    # logger.info(f"✓ 已保存 {len(memory_updates)} 条 Memory 更新记录")
    
    # logger.info(f"Stage 2 completed. Memory updated with {len(outputs)} items")


def run_stage3(
    dataset_name: str,
    dataset_path: str,
    output_path: Optional[str] = None
) -> None:
    """
    Run Stage 3: Inference with Memory-Guided Principles.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset file
        output_path: Optional output path for inference results
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Inference with Memory-Guided Principles")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load and preprocess dataset
    processor = DatasetProcessor()
    data = processor.load_dataset(dataset_name, dataset_path)
    logger.info(f"Loaded {len(data)} items from {dataset_name}")
    
    # Convert to ReasoningInput format
    inputs = [
        ReasoningInput(
            question=item['question'],
            answer=item['answer'],  # For evaluation
            metadata={'dataset': dataset_name}
        )
        for item in data
    ]
    
    # Setup components
    engine = setup_inference_engine(config)
    prompts = PromptTemplate()
    memory = MemoryManager(path=str(config.paths.memory_file))
    
    # Create Inference agent
    agent_config = {
        'inference_engine': engine,
        'prompt_template': prompts,
        'memory_manager': memory,
        'batch_size': config.inference.batch_size
    }
    agent = InferenceAgent(agent_config)
    
    # Process data
    logger.info("Starting Stage 3 processing...")
    outputs = agent.process_batch(inputs)
    
    # Save results
    if output_path is None:
        output_path = config.paths.output_dir / f"inference_{dataset_name}.json"
    else:
        output_path = Path(output_path)
    
    # Convert outputs to JSON
    results = []
    for inp, out in zip(inputs, outputs):
        results.append({
            'question': out.question,
            'ground_truth': inp.answer,
            'predicted_answer': out.chosen_answer,
            'task_description': out.task_description,
            'has_principles': out.metadata.get('has_principles', False),
            'inference_type': out.metadata.get('inference_type', 'unknown')
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Stage 3 completed. Results saved to {output_path}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='MetaEvo Framework - Refactored Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Generate DPO data
  python run_experiments.py --stage 1 --dataset gsm8k --dataset-path dataset/gsm8k/test.jsonl
  
  # Stage 2: Update memory
  python run_experiments.py --stage 2
  
  # Stage 3: Inference with memory
  python run_experiments.py --stage 3 --dataset gsm8k --dataset-path dataset/gsm8k/test.jsonl
        """
    )
    
    parser.add_argument(
        '--stage',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Stage to run (1: DPO generation, 2: Memory update, 3: Inference)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (gsm8k, math, bbh, mmlu, svamp)'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to dataset file'
    )
    
    parser.add_argument(
        '--dpo-file',
        type=str,
        help='Path to DPO file (for stage 2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = MetaConfig.from_env()
    if args.debug:
        config.debug_mode = True
        config.logging.log_level = 'DEBUG'
    
    initialize_config(config)
    
    logger.info("MetaEvo Framework - Refactored Architecture")
    logger.info(f"Stage: {args.stage}")
    
    if config.debug_mode:
        logger.debug("Configuration:")
        logger.debug(json.dumps(config.to_dict(), indent=2))
    
    # Run the appropriate stage
    try:
        if args.stage == 1:
            if not args.dataset or not args.dataset_path:
                parser.error("Stage 1 requires --dataset and --dataset-path")
            run_stage1(args.dataset, args.dataset_path, args.output)
        
        elif args.stage == 2:
            run_stage2(args.dpo_file)
        
        elif args.stage == 3:
            if not args.dataset or not args.dataset_path:
                parser.error("Stage 3 requires --dataset and --dataset-path")
            run_stage3(args.dataset, args.dataset_path, args.output)
    
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
