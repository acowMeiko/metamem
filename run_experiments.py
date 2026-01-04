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
    
    # Convert DPO format to ReasoningInput
    # DPO format: {"input": "Question: ... Error Analysis: ...", "rejected": "...", "chosen": "..."}
    inputs = []
    for item in dpo_data:
        # Extract question and diff from input field
        input_text = item.get('input', '')
        
        # Simple parsing (you may need more robust parsing)
        import re
        q_match = re.search(r'Question:\s*(.*?)\s*(?:Error Analysis:|Error Points:)', input_text, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        
        # Extract task description if available
        task_desc_match = re.search(r'Task Description:\s*(.*?)(?:\n\n|$)', input_text, re.DOTALL)
        task_desc = task_desc_match.group(1).strip() if task_desc_match else None
        
        # Extract principles from chosen answer or input
        # This depends on your DPO format
        principles = []
        principle_match = re.findall(r'Principle:\s*"([^"]+)"', item.get('chosen', ''))
        if principle_match:
            principles = principle_match
        
        if question and task_desc and principles:
            inputs.append(ReasoningInput(
                question=question,
                answer="",
                task_description=task_desc,
                principles=principles,
                metadata={'dpo_file': str(dpo_file)}
            ))
    
    logger.info(f"Extracted {len(inputs)} valid items with task descriptions and principles")
    
    # Setup memory manager
    memory = MemoryManager(path=str(config.paths.memory_file))
    
    # Create Stage 2 agent
    agent_config = {
        'memory_manager': memory,
        'save_frequency': config.memory.save_frequency
    }
    agent = StageTwoAgent(agent_config)
    
    # Process data
    logger.info("Starting Stage 2 processing...")
    outputs = agent.process_batch(inputs)
    
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
