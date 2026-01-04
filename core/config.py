"""
Configuration management for MetaEvo framework.

This module provides a centralized configuration system that:
- Loads settings from environment variables and config files
- Provides type-safe access to configuration values
- Supports configuration validation
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """Path-related configuration."""
    project_root: Path
    checkpoint_dir: Path
    output_dir: Path
    data_dir: Path
    log_dir: Path
    memory_file: Path
    
    @classmethod
    def from_project_root(cls, root: Path) -> 'PathConfig':
        """Create PathConfig from project root."""
        return cls(
            project_root=root,
            checkpoint_dir=root / 'checkpoints',
            output_dir=root / 'output',
            data_dir=root / 'data',
            log_dir=root / 'logs',
            memory_file=root / 'memory' / 'memory_round4.json'
        )


@dataclass
class ModelConfig:
    """Model-related configuration."""
    # Weak model (usually local vLLM)
    weak_model_type: str = 'local'  # 'local' or 'api'
    weak_model_name: str = '/home/share/hcz/qwen2.5-14b-awq'
    weak_model_url: Optional[str] = None
    weak_model_key: Optional[str] = None
    
    # Strong model (usually API)
    strong_model_type: str = 'api'
    strong_model_name: str = 'DeepSeek-R1'
    strong_model_url: str = 'https://llmapi.paratera.com/v1/'
    strong_model_key: str = 'sk-0tKGY03c9OJPODlWGzAGPw'
    
    # LoRA (if applicable)
    lora_model_path: Optional[str] = None
    
    # Model parameters
    max_model_len: int = 32768
    tensor_parallel_size: int = 4


@dataclass
class InferenceConfig:
    """Inference-related configuration."""
    # Generation parameters
    default_temperature: float = 0.0
    default_top_p: float = 0.95
    default_max_tokens: int = 4096
    
    # Task-specific max tokens
    task_desc_max_tokens: int = 2560
    diff_analysis_max_tokens: int = 1024
    principle_max_tokens: int = 2560
    answer_max_tokens: int = 2048
    
    # Batch and concurrency
    batch_size: int = 256
    max_workers: int = 20


@dataclass
class MemoryConfig:
    """Memory-related configuration."""
    memory_file: Path = Path('memory/memory_round4.json')
    save_frequency: int = 50


@dataclass
class LogConfig:
    """Logging configuration."""
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file: Optional[Path] = None


@dataclass
class MetaConfig:
    """Main configuration class."""
    paths: PathConfig
    models: ModelConfig
    inference: InferenceConfig
    memory: MemoryConfig
    logging: LogConfig
    debug_mode: bool = False
    
    @classmethod
    def from_env(cls, project_root: Optional[Path] = None) -> 'MetaConfig':
        """
        Create configuration from environment variables.
        
        Args:
            project_root: Project root path (default: current file's parent's parent)
            
        Returns:
            MetaConfig instance
        """
        if project_root is None:
            # Go up one level from core/ to get to metanew3/
            project_root = Path(__file__).parent.parent.absolute()
        
        paths = PathConfig.from_project_root(project_root)
        
        models = ModelConfig(
            weak_model_name=os.getenv('BASE_MODEL_NAME', '/home/models/qwen_dpo4_lora'),
            strong_model_name=os.getenv('STRONG_MODEL_NAME', 'DeepSeek-R1'),
            strong_model_url=os.getenv('STRONG_MODEL_API_URL', 'https://llmapi.paratera.com/v1/'),
            strong_model_key=os.getenv('STRONG_MODEL_KEY', 'sk-0tKGY03c9OJPODlWGzAGPw'),
            max_model_len=int(os.getenv('MAX_MODEL_LEN', '32768')),
        )
        
        inference = InferenceConfig(
            default_temperature=float(os.getenv('DEFAULT_TEMPERATURE', '0.1')),
            default_top_p=float(os.getenv('DEFAULT_TOP_P', '0.95')),
            default_max_tokens=int(os.getenv('DEFAULT_MAX_TOKENS', '4096')),
            batch_size=int(os.getenv('BATCH_SIZE', '256')),
            max_workers=int(os.getenv('MAX_WORKERS', '20')),
        )
        
        memory = MemoryConfig(
            memory_file=paths.memory_file,
            save_frequency=int(os.getenv('SAVE_FREQUENCY', '50'))
        )
        
        logging_config = LogConfig(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=paths.log_dir / 'metaevo.log'
        )
        
        debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        return cls(
            paths=paths,
            models=models,
            inference=inference,
            memory=memory,
            logging=logging_config,
            debug_mode=debug_mode
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetaConfig':
        """Create configuration from dictionary."""
        # This is a simplified version, you may want to add more logic
        raise NotImplementedError("from_dict not yet implemented")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'paths': {
                'project_root': str(self.paths.project_root),
                'checkpoint_dir': str(self.paths.checkpoint_dir),
                'output_dir': str(self.paths.output_dir),
                'data_dir': str(self.paths.data_dir),
                'log_dir': str(self.paths.log_dir),
                'memory_file': str(self.paths.memory_file),
            },
            'models': {
                'weak_model': {
                    'type': self.models.weak_model_type,
                    'name': self.models.weak_model_name,
                },
                'strong_model': {
                    'type': self.models.strong_model_type,
                    'name': self.models.strong_model_name,
                    'url': self.models.strong_model_url,
                },
            },
            'inference': {
                'batch_size': self.inference.batch_size,
                'max_workers': self.inference.max_workers,
                'temperature': self.inference.default_temperature,
            },
            'memory': {
                'save_frequency': self.memory.save_frequency,
            },
            'debug_mode': self.debug_mode
        }
    
    def validate(self) -> None:
        """Validate configuration."""
        # Check critical paths
        if not self.paths.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.paths.project_root}")
        
        # Check model configuration
        if self.models.strong_model_type == 'api':
            if not self.models.strong_model_url:
                raise ValueError("Strong model URL is required for API type")
            if not self.models.strong_model_key:
                logger.warning("Strong model API key is not set")
        
        # Check inference parameters
        if self.inference.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.inference.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        
        logger.info("Configuration validated successfully")
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        # Ensure log directory exists
        if self.logging.log_file:
            self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            handlers = [
                logging.FileHandler(self.logging.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        else:
            handlers = [logging.StreamHandler()]
        
        logging.basicConfig(
            level=getattr(logging, self.logging.log_level),
            format=self.logging.log_format,
            handlers=handlers,
            force=True  # Override any existing configuration
        )
        
        logger.info(f"Logging configured: level={self.logging.log_level}")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.paths.checkpoint_dir,
            self.paths.output_dir,
            self.paths.data_dir,
            self.paths.log_dir,
            self.paths.memory_file.parent
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("All directories ensured")


# Global configuration instance
_config: Optional[MetaConfig] = None


def get_config() -> MetaConfig:
    """
    Get the global configuration instance.
    
    Returns:
        MetaConfig instance
    """
    global _config
    if _config is None:
        _config = MetaConfig.from_env()
        _config.validate()
    return _config


def initialize_config(config: MetaConfig) -> None:
    """
    Initialize the global configuration.
    
    Args:
        config: MetaConfig instance to use
    """
    global _config
    _config = config
    _config.validate()
    _config.setup_logging()
    _config.ensure_directories()
    logger.info("Configuration initialized")


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
