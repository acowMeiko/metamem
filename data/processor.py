"""
Data processing module for different datasets.

This module provides a unified interface for loading and preprocessing
various datasets (GSM8K, MATH, BBH, MMLU, SVAMP, etc.) into a standard format.

Design Pattern: Registry Pattern
- Each dataset has a dedicated preprocessor function
- New datasets can be added by registering a preprocessor
- Standard output format: {"question": str, "answer": str}
"""

from typing import List, Dict, Any, Callable
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Unified dataset processor with registry pattern.
    
    Usage:
        processor = DatasetProcessor()
        data = processor.load_dataset('gsm8k', 'path/to/data.jsonl')
    """
    
    def __init__(self):
        """Initialize the processor with registered preprocessors."""
        self._preprocessors: Dict[str, Callable] = {}
        self._register_default_preprocessors()
    
    def _register_default_preprocessors(self) -> None:
        """Register all default dataset preprocessors."""
        self.register('gsm8k', self._preprocess_gsm8k)
        self.register('math', self._preprocess_math)
        self.register('bbh', self._preprocess_bbh)
        self.register('mmlu', self._preprocess_mmlu)
        self.register('svamp', self._preprocess_svamp)
        self.register('test_filter', self._preprocess_test_filter)
    
    def register(self, dataset_name: str, preprocessor: Callable) -> None:
        """
        Register a new dataset preprocessor.
        
        Args:
            dataset_name: Name of the dataset
            preprocessor: Function that takes raw data and returns List[Dict[str, str]]
        """
        self._preprocessors[dataset_name] = preprocessor
        logger.info(f"Registered preprocessor for dataset: {dataset_name}")
    
    def load_dataset(self, dataset_name: str, dataset_path: str) -> List[Dict[str, str]]:
        """
        Load and preprocess a dataset.
        
        Args:
            dataset_name: Name of the dataset (must be registered)
            dataset_path: Path to the dataset file
            
        Returns:
            List of standardized data: [{"question": str, "answer": str}, ...]
            
        Raises:
            ValueError: If dataset_name is not registered
            FileNotFoundError: If dataset file doesn't exist
        """
        if dataset_name not in self._preprocessors:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {list(self._preprocessors.keys())}"
            )
        
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info("=" * 60)
        logger.info(f"[DatasetProcessor] Loading dataset: {dataset_name}")
        logger.info(f"[DatasetProcessor] Path: {dataset_path}")
        logger.info("=" * 60)
        
        # Load raw data
        raw_data = self._load_file(path)
        
        # Preprocess
        preprocessor = self._preprocessors[dataset_name]
        processed_data = preprocessor(raw_data)
        
        logger.info(f"[DatasetProcessor] Loaded {len(processed_data)} items")
        return processed_data
    
    def _load_file(self, path: Path) -> Any:
        """
        Load raw data from file based on extension.
        
        Supports:
        - .jsonl: One JSON object per line
        - .json: Single JSON object or array
        """
        try:
            if path.suffix == '.jsonl':
                raw_data = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            raw_data.append(json.loads(line))
                logger.info(f"Loaded JSONL file: {len(raw_data)} lines")
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                logger.info(f"Loaded JSON file")
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            return raw_data
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise
    
    # ==================== Dataset-Specific Preprocessors ====================
    
    @staticmethod
    def _preprocess_gsm8k(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Preprocess GSM8K dataset.
        
        Format: {"question": str, "answer": str}
        Already in standard format.
        """
        logger.info(f"Preprocessing GSM8K: {len(raw_data)} items")
        return [
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", "")
            }
            for item in raw_data
            if item.get("question") and item.get("answer")
        ]
    
    @staticmethod
    def _preprocess_math(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Preprocess MATH dataset.
        
        Format: {"problem": str, "answer": str}
        Map "problem" → "question"
        """
        logger.info(f"Preprocessing MATH: {len(raw_data)} items")
        return [
            {
                "question": item.get("problem", ""),
                "answer": item.get("answer", "")
            }
            for item in raw_data
            if item.get("problem") and item.get("answer")
        ]
    
    @staticmethod
    def _preprocess_bbh(raw_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Preprocess BBH (Big-Bench Hard) dataset.
        
        Format: {"examples": [{"input": str, "target": str}, ...]}
        Extract all examples.
        """
        examples = raw_data.get("examples", [])
        logger.info(f"Preprocessing BBH: {len(examples)} examples")
        return [
            {
                "question": example.get("input", ""),
                "answer": example.get("target", "")
            }
            for example in examples
            if example.get("input") and example.get("target")
        ]
    
    @staticmethod
    def _preprocess_mmlu(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Preprocess MMLU dataset.
        
        Format: {
            "question": str,
            "choices": [str, str, str, str],
            "answer": "A"/"B"/"C"/"D" or answer_idx: int
        }
        
        Construct full question with choices, map answer letter to choice text.
        """
        logger.info(f"Preprocessing MMLU: {len(raw_data)} items")
        processed = []
        
        for item in raw_data:
            # Build question
            question = item.get("full_question")
            if not question:
                q = item.get("question", "")
                choices = item.get("choices", [])
                if q and choices:
                    choices_text = "\n".join([
                        f"{chr(65+i)}. {choice}" 
                        for i, choice in enumerate(choices)
                    ])
                    question = f"{q}\n{choices_text}"
            
            # Extract answer
            answer = ""
            choices = item.get("choices", [])
            
            if "answer_idx" in item and choices:
                idx = item["answer_idx"]
                if 0 <= idx < len(choices):
                    answer = choices[idx]
            elif "answer" in item and choices:
                answer_letter = item["answer"]
                idx = ord(answer_letter.upper()) - ord('A')
                if 0 <= idx < len(choices):
                    answer = choices[idx]
            
            if question and answer:
                processed.append({
                    "question": question,
                    "answer": answer
                })
        
        return processed
    
    @staticmethod
    def _preprocess_svamp(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Preprocess SVAMP dataset.
        
        Format: {
            "Body": str,
            "Question": str,
            "Answer": float
        }
        
        Concatenate Body + Question, convert Answer to string.
        """
        logger.info(f"Preprocessing SVAMP: {len(raw_data)} items")
        processed = []
        
        for item in raw_data:
            body = item.get("Body", "").strip()
            question = item.get("Question", "").strip()
            answer = item.get("Answer")
            
            # Combine body and question
            full_question = f"{body} {question}".strip() if body else question
            
            if full_question and answer is not None:
                processed.append({
                    "question": full_question,
                    "answer": str(answer)
                })
        
        return processed
    
    @staticmethod
    def _preprocess_test_filter(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Preprocess test_filter dataset.
        
        Format: {
            "problem": str,
            "answer": str,
            ...
        }
        
        Map "problem" → "question"
        """
        logger.info(f"Preprocessing test_filter: {len(raw_data)} items")
        return [
            {
                "question": item.get("problem", ""),
                "answer": item.get("answer", "")
            }
            for item in raw_data
            if item.get("problem") and item.get("answer")
        ]


class DatasetRegistry:
    """
    Global registry for dataset paths and configurations.
    
    Usage:
        registry = DatasetRegistry()
        registry.register_dataset('gsm8k', 'dataset/gsm8k/test.jsonl')
        path = registry.get_path('gsm8k')
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._datasets: Dict[str, Dict[str, Any]] = {}
    
    def register_dataset(
        self, 
        name: str, 
        path: str, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Register a dataset.
        
        Args:
            name: Dataset name
            path: Path to dataset file
            metadata: Optional metadata (e.g., description, version)
        """
        self._datasets[name] = {
            'path': path,
            'metadata': metadata or {}
        }
        logger.info(f"Registered dataset: {name} -> {path}")
    
    def get_path(self, name: str) -> str:
        """
        Get the path for a registered dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Path to the dataset file
            
        Raises:
            KeyError: If dataset is not registered
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset not registered: {name}")
        return self._datasets[name]['path']
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._datasets.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a dataset."""
        if name not in self._datasets:
            raise KeyError(f"Dataset not registered: {name}")
        return self._datasets[name]['metadata']


# Global instances
_processor = DatasetProcessor()
_registry = DatasetRegistry()


def get_processor() -> DatasetProcessor:
    """Get the global DatasetProcessor instance."""
    return _processor


def get_registry() -> DatasetRegistry:
    """Get the global DatasetRegistry instance."""
    return _registry
