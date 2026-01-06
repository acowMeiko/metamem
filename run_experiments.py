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

import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import argparse
import logging
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
        max_tokens=config.inference.default_max_tokens,
        repetition_penalty=config.inference.default_repetition_penalty,
        frequency_penalty=config.inference.default_frequency_penalty
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
    
    # Determine output path
    if output_path is None:
        output_path = config.paths.output_dir / f"dpo_{dataset_name}.json"
    else:
        output_path = Path(output_path)
    
    # Check if DPO file exists and determine processing mode
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    skip_chosen = False
    
    if file_exists:
        logger.info("=" * 60)
        logger.info("检测到现有 DPO 文件")
        logger.info("=" * 60)
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                logger.info(f"现有数据条数: {len(existing_data)}")
                skip_chosen = True
        except Exception as e:
            logger.warning(f"无法读取现有文件: {e}")
        
        if skip_chosen:
            logger.info("将启用增量更新模式:")
    else:
        logger.info("=" * 60)
        logger.info("完整生成模式")
        logger.info("=" * 60)
        logger.info("将生成所有字段（首次创建 DPO 数据）")
        logger.info("=" * 60)
    
    # Process data with skip_chosen flag
    logger.info("Starting Stage 1 processing...")
    outputs = agent.process_batch(inputs, skip_chosen=skip_chosen)
    
    # Save DPO format (自动判断增量更新 or 完整生成)
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
    
    # ===== 阶段1: 数据预处理和验证 =====
    logger.info("阶段1/2: 数据预处理和验证")
    
    # 检查是否所有数据都有 task_description
    missing_taskdesc = [i for i, item in enumerate(dpo_data) if not item.get('task_description')]
    if missing_taskdesc:
        logger.error(f"发现 {len(missing_taskdesc)} 条数据缺少 task_description 字段")
        return
    logger.info("✓ 所有数据都包含 task_description 字段")
    # 准备批处理数据
    batch_data = []
    for i, item in enumerate(dpo_data):
        question = item.get("question")
        task_description = item.get("task_description")
        rejected = item.get("rejected", "")
        
        if not question or not task_description:
            logger.warning(f"第 {i} 项数据缺少 question 或 task_description，跳过")
            continue
        if not rejected:
            logger.warning(f"第 {i} 项数据缺少 rejected 原则，跳过")
            continue
        
        batch_data.append({
            'index': i,
            'question': question,
            'task_description': task_description,
            'rejected': rejected
        })
    
    logger.info(f"准备批处理 {len(batch_data)} 条数据")
    
    # ===== 阶段2: 构建 ReasoningInput 并更新 Memory =====
    logger.info("阶段2/2: 构建输入并更新 Memory")
    
    # def clean_markdown(text):
    #     """清理 markdown 代码块标记"""
    #     text_clean = text.strip()
    #     if text_clean.startswith('```json'):
    #         text_clean = text_clean[7:]
    #     elif text_clean.startswith('```'):
    #         text_clean = text_clean[3:]
    #     if text_clean.endswith('```'):
    #         text_clean = text_clean[:-3]
    #     return text_clean.strip()
    
    # 构建输入
    inputs = []
    for item in batch_data:
        task_desc_raw = item['task_description']
        principles_raw = item['rejected']
        
        # 解析 task_description：提取 JSON 中的 description 字段
        task_desc = None
        if isinstance(task_desc_raw, str):
            try:
                # 尝试找到第一个完整的 JSON 对象
                import re
                json_match = re.search(r'\{[^}]*"taskDescription"[^}]*\{[^}]*"description"[^}]*:([^}]*)\}[^}]*\}', task_desc_raw, re.DOTALL)
                if json_match:
                    # 提取 JSON 部分
                    json_start = task_desc_raw.find('{')
                    json_part = task_desc_raw[json_start:]
                    # 找到第一个完整的 JSON 对象
                    brace_count = 0
                    for i, char in enumerate(json_part):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = json_part[:i+1]
                                break
                    
                    task_data = json.loads(json_str)
                    task_desc = task_data.get('taskDescription', {}).get('description', '')
                else:
                    logger.warning(f"第 {item['index']} 项的 task_description 格式不符合预期")
                    task_desc = task_desc_raw
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.warning(f"第 {item['index']} 项解析 task_description 失败: {e}，使用原始值")
                task_desc = task_desc_raw
        else:
            task_desc = task_desc_raw
        
        # 解析 rejected：提取 JSON 中的 Principle 列表
        principles = []
        if isinstance(principles_raw, str):
            try:
                # 尝试解析为 JSON
                principles_data = json.loads(principles_raw)
                # 提取 output 中的 Principle
                if isinstance(principles_data, dict) and 'output' in principles_data:
                    output_list = principles_data['output']
                    if isinstance(output_list, list):
                        principles = [item.get('Principle', '') for item in output_list if isinstance(item, dict) and 'Principle' in item]
                        # 过滤空字符串
                        principles = [p for p in principles if p.strip()]
            except json.JSONDecodeError:
                # 如果不是 JSON，按原来的方式处理
                principles = [p.strip() for p in principles_raw.split('\n') if p.strip()]
                if not principles:
                    principles = [principles_raw] if principles_raw else []
        elif isinstance(principles_raw, list):
            principles = principles_raw
        
        if not task_desc or not principles:
            logger.warning(f"第 {item['index']} 项缺少任务描述或原则，跳过")
            continue
        
        inputs.append(ReasoningInput(
            question=item['question'],
            answer="",
            task_description=task_desc,
            principles=principles,  # 现在是 list 类型
            metadata={'dpo_file': str(dpo_file), 'index': item['index']}
        ))
    
    logger.info(f"成功构建 {len(inputs)} 个有效输入项")
    
    # Setup memory manager
    memory = MemoryManager(path=str(config.paths.memory_file))
    # Create Stage 2 agent
    agent_config = {
        'memory_manager': memory,
        'save_frequency': config.memory.save_frequency
    }
    agent = StageTwoAgent(agent_config)
    
    # Process data
    logger.info("更新 Memory...")
    outputs = agent.process_batch(inputs)
    
    # 保存最终的 Memory 更新结果
    memory_update_results_file = config.paths.output_dir / f"stage2_memory_updates_{Path(dpo_file).stem}.json"
    logger.info(f"保存 Memory 更新结果到: {memory_update_results_file}")
    
    memory_updates = []
    for inp, out in zip(inputs, outputs):
        memory_updates.append({
            "question": inp.question,
            "task_description": out.task_description,
            "principles": out.principles,
            "memory_status": out.metadata.get('memory_status'),
            "matched_task": out.metadata.get('matched_task'),
            "index": inp.metadata.get('index')
        })
    
    with open(memory_update_results_file, 'w', encoding='utf-8') as f:
        json.dump(memory_updates, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 已保存 {len(memory_updates)} 条 Memory 更新记录")
    
    logger.info(f"Stage 2 completed. Memory updated with {len(outputs)} items")


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
            'predicted_answer': out.chosen_answer,#这就是模型输出的答案
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
