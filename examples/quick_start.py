"""
Quick Start Example for MetaEvo Refactored Architecture

This script demonstrates how to use the new modular architecture
for each stage of the MetaEvo pipeline.
"""

from pathlib import Path
from core.config import MetaConfig, initialize_config
from core.base import ReasoningInput
from core.stages import StageOneAgent, StageTwoAgent, InferenceAgent
from data.processor import DatasetProcessor
from inference.engine import InferenceEngineBuilder
from templates.prompts import PromptTemplate
from module.memory_module import MemoryManager
import logging

logger = logging.getLogger(__name__)


def example_stage1():
    """Example: Generate DPO training data."""
    print("\n" + "="*60)
    print("EXAMPLE: Stage 1 - DPO Data Generation")
    print("="*60)
    
    # 1. Initialize configuration
    config = MetaConfig.from_env()
    initialize_config(config)
    
    # 2. Load dataset
    processor = DatasetProcessor()
    # Using a small sample for demonstration
    sample_data = [
        {"question": "What is 2 + 3?", "answer": "5"},
        {"question": "What is 5 * 6?", "answer": "30"},
    ]
    
    # 3. Convert to standard format
    inputs = [
        ReasoningInput(question=item['question'], answer=item['answer'])
        for item in sample_data
    ]
    
    # 4. Setup inference engine
    engine = (InferenceEngineBuilder()
              .set_weak_model('local', config.models.weak_model_name)
              .set_strong_model(
                  'api',
                  config.models.strong_model_name,
                  config.models.strong_model_url,
                  config.models.strong_model_key
              )
              .set_defaults(temperature=0.0)
              .build())
    
    # 5. Create Stage 1 agent
    agent = StageOneAgent({
        'inference_engine': engine,
        'prompt_template': PromptTemplate(),
        'batch_size': 2,
        'max_workers': 2
    })
    
    # 6. Process data
    outputs = agent.process_batch(inputs)
    
    # 7. Save DPO format
    output_path = Path('output/example_dpo.json')
    agent.save_dpo_format(outputs, output_path)
    
    print(f"✓ Stage 1 completed. Output saved to {output_path}")
    print(f"✓ Generated {len(outputs)} DPO items")


def example_stage2():
    """Example: Update memory from DPO data."""
    print("\n" + "="*60)
    print("EXAMPLE: Stage 2 - Memory Update")
    print("="*60)
    
    # 1. Initialize configuration
    config = MetaConfig.from_env()
    initialize_config(config)
    
    # 2. Prepare sample data (simulating DPO results)
    sample_inputs = [
        ReasoningInput(
            question="What is 2 + 3?",
            answer="5",
            task_description="Basic arithmetic addition",
            principles=[
                "Always verify the sum by counting",
                "State the final answer clearly"
            ]
        ),
        ReasoningInput(
            question="What is 5 * 6?",
            answer="30",
            task_description="Basic arithmetic multiplication",
            principles=[
                "Use multiplication tables for accuracy",
                "Double-check the result"
            ]
        )
    ]
    
    # 3. Setup memory manager
    memory = MemoryManager(path=str(config.paths.memory_file))
    
    # 4. Create Stage 2 agent
    agent = StageTwoAgent({
        'memory_manager': memory,
        'save_frequency': 10
    })
    
    # 5. Process data
    outputs = agent.process_batch(sample_inputs)
    
    print(f"✓ Stage 2 completed")
    print(f"✓ Updated memory with {len(outputs)} items")
    for out in outputs:
        print(f"  - Task: {out.task_description}")
        print(f"    Status: {out.metadata.get('memory_status')}")


def example_stage3():
    """Example: Inference with memory guidance."""
    print("\n" + "="*60)
    print("EXAMPLE: Stage 3 - Memory-Guided Inference")
    print("="*60)
    
    # 1. Initialize configuration
    config = MetaConfig.from_env()
    initialize_config(config)
    
    # 2. Prepare test questions
    test_questions = [
        {"question": "What is 7 + 8?", "answer": "15"},
        {"question": "What is 9 * 4?", "answer": "36"},
    ]
    
    inputs = [
        ReasoningInput(question=item['question'], answer=item['answer'])
        for item in test_questions
    ]
    
    # 3. Setup components
    engine = (InferenceEngineBuilder()
              .set_weak_model('local', config.models.weak_model_name)
              .set_strong_model(
                  'api',
                  config.models.strong_model_name,
                  config.models.strong_model_url,
                  config.models.strong_model_key
              )
              .build())
    
    memory = MemoryManager(path=str(config.paths.memory_file))
    
    # 4. Create Inference agent
    agent = InferenceAgent({
        'inference_engine': engine,
        'prompt_template': PromptTemplate(),
        'memory_manager': memory,
        'batch_size': 2
    })
    
    # 5. Process data
    outputs = agent.process_batch(inputs)
    
    print(f"✓ Stage 3 completed")
    print(f"✓ Generated {len(outputs)} predictions")
    for inp, out in zip(inputs, outputs):
        print(f"\n  Question: {inp.question}")
        print(f"  Ground Truth: {inp.answer}")
        print(f"  Predicted: {out.chosen_answer}")
        print(f"  Has Principles: {out.metadata.get('has_principles', False)}")
        print(f"  Inference Type: {out.metadata.get('inference_type', 'unknown')}")


def example_custom_dataset():
    """Example: Register and use a custom dataset."""
    print("\n" + "="*60)
    print("EXAMPLE: Custom Dataset Registration")
    print("="*60)
    
    # 1. Define custom preprocessor
    def preprocess_my_dataset(raw_data):
        """Custom preprocessor for your dataset format."""
        return [
            {
                "question": item['my_question_field'],
                "answer": item['my_answer_field']
            }
            for item in raw_data
        ]
    
    # 2. Register the preprocessor
    processor = DatasetProcessor()
    processor.register('my_dataset', preprocess_my_dataset)
    
    print("✓ Custom dataset registered")
    print(f"✓ Available datasets: {list(processor._preprocessors.keys())}")
    
    # 3. Now you can use it
    # data = processor.load_dataset('my_dataset', 'path/to/my_data.json')


def example_custom_prompt():
    """Example: Add a custom prompt template."""
    print("\n" + "="*60)
    print("EXAMPLE: Custom Prompt Template")
    print("="*60)
    
    from string import Template
    
    # 1. Extend PromptTemplate
    class MyPromptTemplate(PromptTemplate):
        CUSTOM_TEMPLATE = Template('''
        You are an expert in $domain.
        
        Question: $question
        
        Please provide a detailed answer.
        ''')
        
        def get_custom_prompt(self, question: str, domain: str) -> str:
            return self.CUSTOM_TEMPLATE.substitute(
                question=question,
                domain=domain
            )
    
    # 2. Use the custom template
    prompts = MyPromptTemplate()
    custom_prompt = prompts.get_custom_prompt(
        question="What is machine learning?",
        domain="artificial intelligence"
    )
    
    print("✓ Custom prompt created")
    print(f"✓ Prompt preview:\n{custom_prompt[:100]}...")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MetaEvo Framework - Quick Start Examples")
    print("="*60)
    
    try:
        # Stage examples
        print("\n[INFO] Running Stage 1 example...")
        # example_stage1()  # Commented out to avoid actual model calls
        print("[SKIP] Stage 1 example (requires model access)")
        
        print("\n[INFO] Running Stage 2 example...")
        # example_stage2()  # Commented out to avoid file writes
        print("[SKIP] Stage 2 example (requires file access)")
        
        print("\n[INFO] Running Stage 3 example...")
        # example_stage3()  # Commented out to avoid actual model calls
        print("[SKIP] Stage 3 example (requires model access)")
        
        # Extension examples (safe to run)
        example_custom_dataset()
        example_custom_prompt()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        print("\nTo run the full pipeline:")
        print("  python run_experiments.py --stage 1 --dataset gsm8k --dataset-path ...")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
