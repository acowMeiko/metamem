"""
Unified inference engine for vLLM (local) and API (remote) models.

This module abstracts away the differences between local and API inference,
providing a consistent interface for all reasoning agents.

Design Pattern: Adapter Pattern
- InferenceEngine provides unified interface
- Adapts vLLM batch inference and API concurrent calls
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from inference.local_inference import batch_inference as vllm_batch_inference
from inference.local_inference import single_inference as vllm_single_inference
from inference.api_inference import gpt_call

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference engine supporting both local (vLLM) and API models.
    
    Features:
    - Batch inference for vLLM (efficient GPU utilization)
    - Concurrent API calls (efficient I/O utilization)
    - Unified interface for weak/strong models
    - Configurable parameters (temperature, max_tokens, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the inference engine.
        
        Args:
            config: Configuration dictionary containing:
                - weak_model: Config for weak model (usually local vLLM)
                    - type: 'local' or 'api'
                    - name: Model name
                    - url: API URL (if type='api')
                    - api_key: API key (if type='api')
                - strong_model: Config for strong model (usually API)
                    - type: 'api'
                    - name: Model name
                    - url: API URL
                    - api_key: API key
                - default_temperature: Default temperature
                - default_top_p: Default top_p
                - default_max_tokens: Default max_tokens
        """
        self.config = config
        self.weak_model = config.get('weak_model', {})
        self.strong_model = config.get('strong_model', {})
        self.default_temperature = config.get('default_temperature', 0.0)
        self.default_top_p = config.get('default_top_p', 0.95)
        self.default_max_tokens = config.get('default_max_tokens', 2048)
        self.default_repetition_penalty = config.get('default_repetition_penalty', 1.05)
        self.default_frequency_penalty = config.get('default_frequency_penalty', 0.0)
        
        self._validate_config()
        logger.info("InferenceEngine initialized")
        logger.info(f"Weak model: {self.weak_model.get('type')} - {self.weak_model.get('name')}")
        logger.info(f"Strong model: {self.strong_model.get('type')} - {self.strong_model.get('name')}")
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not self.weak_model:
            raise ValueError("weak_model configuration is required")
        if not self.strong_model:
            raise ValueError("strong_model configuration is required")
        
        # Validate weak model
        if 'type' not in self.weak_model:
            raise ValueError("weak_model.type is required")
        
        # Validate strong model
        if self.strong_model.get('type') != 'api':
            logger.warning("strong_model.type is not 'api', this may cause issues")
    
    def single_inference(
        self,
        prompt: str,
        model_type: str = 'weak',
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None
    ) -> str:
        """
        Perform single inference.
        
        Args:
            prompt: Input prompt
            model_type: 'weak' or 'strong'
            temperature: Sampling temperature (default: config value)
            top_p: Top-p sampling (default: config value)
            max_tokens: Maximum tokens to generate (default: config value)
            stop: Stop sequences
            repetition_penalty: Repetition penalty (vLLM only)
            frequency_penalty: Frequency penalty (API only)
            
        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        repetition_penalty = repetition_penalty if repetition_penalty is not None else getattr(self, 'default_repetition_penalty', 1.0)
        frequency_penalty = frequency_penalty if frequency_penalty is not None else getattr(self, 'default_frequency_penalty', 0.0)
        
        model_config = self.weak_model if model_type == 'weak' else self.strong_model
        
        if model_config['type'] == 'local':
            # Use vLLM
            return vllm_single_inference(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                repetition_penalty=repetition_penalty
            )
        else:
            # Use API
            return gpt_call(
                user=prompt,
                model=model_config.get('name'),
                url=model_config.get('url'),
                api_key=model_config.get('api_key'),
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty
            )
    
    def batch_inference(
        self,
        prompts: List[str],
        model_type: str = 'weak',
        batch_size: int = 64,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None
    ) -> List[str]:
        """
        Perform batch inference (optimized for vLLM).
        
        Args:
            prompts: List of prompts
            model_type: 'weak' or 'strong'
            batch_size: Batch size for vLLM (ignored for API)
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            repetition_penalty: Repetition penalty (vLLM only)
            frequency_penalty: Frequency penalty (API only)
            
        Returns:
            List of generated texts
        """
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        repetition_penalty = repetition_penalty if repetition_penalty is not None else getattr(self, 'default_repetition_penalty', 1.0)
        frequency_penalty = frequency_penalty if frequency_penalty is not None else getattr(self, 'default_frequency_penalty', 0.0)
        
        model_config = self.weak_model if model_type == 'weak' else self.strong_model
        
        if model_config['type'] == 'local':
            # Use vLLM batch inference
            logger.info(f"vLLM batch inference: {len(prompts)} prompts")
            return vllm_batch_inference(
                prompts=prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                repetition_penalty=repetition_penalty
            )
        else:
            # For API, use concurrent calls
            logger.info(f"API batch inference (concurrent): {len(prompts)} prompts")
            return self.concurrent_api_inference(
                prompts=prompts,
                model_type=model_type,
                max_workers=20,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty
            )
    
    def concurrent_api_inference(
        self,
        prompts: List[str],
        model_type: str = 'strong',
        max_workers: int = 20,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Perform concurrent API inference for better I/O utilization.
        
        Args:
            prompts: List of prompts
            model_type: 'weak' or 'strong'
            max_workers: Number of concurrent workers
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated texts
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        model_config = self.weak_model if model_type == 'weak' else self.strong_model
        
        if model_config['type'] != 'api':
            logger.warning(f"{model_type} model is not configured for API, falling back to batch_inference")
            return self.batch_inference(
                prompts=prompts,
                model_type=model_type,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        logger.info(f"Concurrent API calls: {len(prompts)} prompts with {max_workers} workers")
        
        results = [None] * len(prompts)
        
        def call_api(idx: int, prompt: str) -> tuple:
            """Single API call."""
            try:
                response = gpt_call(
                    user=prompt,
                    model=model_config.get('name'),
                    url=model_config.get('url'),
                    api_key=model_config.get('api_key'),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return idx, response, None
            except Exception as e:
                logger.error(f"API call {idx} failed: {e}")
                return idx, None, str(e)
        
        # Execute concurrent calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(call_api, idx, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="API Calls"):
                idx, result, error = future.result()
                if error:
                    logger.warning(f"Prompt {idx} failed, using empty string")
                    results[idx] = ""
                else:
                    results[idx] = result
        
        return results
    
    def get_model_info(self, model_type: str = 'weak') -> Dict[str, Any]:
        """
        Get information about a configured model.
        
        Args:
            model_type: 'weak' or 'strong'
            
        Returns:
            Model configuration dictionary
        """
        return self.weak_model if model_type == 'weak' else self.strong_model


class InferenceEngineBuilder:
    """
    Builder class for constructing InferenceEngine instances.
    
    Usage:
        engine = (InferenceEngineBuilder()
                  .set_weak_model('local', 'qwen2.5-14b')
                  .set_strong_model('api', 'DeepSeek-R1', url='...', api_key='...')
                  .set_defaults(temperature=0.0)
                  .build())
    """
    
    def __init__(self):
        """Initialize the builder."""
        self.config = {
            'weak_model': {},
            'strong_model': {},
            'default_temperature': 0.0,
            'default_top_p': 0.95,
            'default_max_tokens': 2048,
            'default_repetition_penalty': 1.1,
            'default_frequency_penalty': 0.0
        }
    
    def set_weak_model(
        self,
        model_type: str,
        name: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> 'InferenceEngineBuilder':
        """
        Configure the weak model.
        
        Args:
            model_type: 'local' or 'api'
            name: Model name
            url: API URL (required if type='api')
            api_key: API key (required if type='api')
        """
        self.config['weak_model'] = {
            'type': model_type,
            'name': name,
            'url': url,
            'api_key': api_key
        }
        return self
    
    def set_strong_model(
        self,
        model_type: str,
        name: str,
        url: str,
        api_key: str
    ) -> 'InferenceEngineBuilder':
        """
        Configure the strong model.
        
        Args:
            model_type: Usually 'api'
            name: Model name
            url: API URL
            api_key: API key
        """
        self.config['strong_model'] = {
            'type': model_type,
            'name': name,
            'url': url,
            'api_key': api_key
        }
        return self
    
    def set_defaults(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None
    ) -> 'InferenceEngineBuilder':
        """
        Set default inference parameters.
        
        Args:
            temperature: Default temperature
            top_p: Default top_p
            max_tokens: Default max_tokens
            repetition_penalty: Default repetition penalty (vLLM)
            frequency_penalty: Default frequency penalty (API)
        """
        if temperature is not None:
            self.config['default_temperature'] = temperature
        if top_p is not None:
            self.config['default_top_p'] = top_p
        if max_tokens is not None:
            self.config['default_max_tokens'] = max_tokens
        if repetition_penalty is not None:
            self.config['default_repetition_penalty'] = repetition_penalty
        if frequency_penalty is not None:
            self.config['default_frequency_penalty'] = frequency_penalty
        return self
    
    def build(self) -> InferenceEngine:
        """Build and return the InferenceEngine instance."""
        return InferenceEngine(self.config)
