"""
Abstract base class for MetaEvo agents.

This module defines the standard interface for all meta-reasoning agents,
following the Strategy Pattern to enable multiple reasoning strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningInput:
    """Standard input format for reasoning agents."""
    question: str
    answer: str
    task_description: Optional[str] = None
    principles: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningOutput:
    """Standard output format for reasoning agents."""
    question: str
    baseline_answer: Optional[str] = None
    diff_analysis: Optional[str] = None
    principles: Optional[List[str]] = None
    chosen_answer: Optional[str] = None
    task_description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'question': self.question,
            'baseline_answer': self.baseline_answer,
            'diff_analysis': self.diff_analysis,
            'principles': self.principles,
            'chosen_answer': self.chosen_answer,
            'task_description': self.task_description,
            'metadata': self.metadata or {}
        }


class MetaAgentBase(ABC):
    """
    Abstract base class for meta-reasoning agents.
    
    All concrete agents (Stage1, Stage2, Inference) must inherit from this class
    and implement the required methods.
    
    Design Pattern: Template Method + Strategy Pattern
    - process(): Template method defining the overall flow
    - _execute_stage(): Abstract method to be implemented by subclasses
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent with configuration.
        
        Args:
            config: Configuration dictionary containing model settings,
                   inference parameters, etc.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration for this agent.
        
        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        pass
    
    @abstractmethod
    def process(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        Main processing method for the agent.
        
        This is the template method that defines the overall processing flow.
        Subclasses should implement _execute_stage() for their specific logic.
        
        Args:
            input_data: Standardized input data
            
        Returns:
            ReasoningOutput: Standardized output data
        """
        pass
    
    @abstractmethod
    def process_batch(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        Process a batch of inputs for better efficiency.
        
        Args:
            inputs: List of standardized input data
            
        Returns:
            List[ReasoningOutput]: List of standardized output data
        """
        pass
    
    def _log_processing(self, stage: str, message: str, level: str = "INFO") -> None:
        """
        Helper method for consistent logging.
        
        Args:
            stage: The processing stage name
            message: Log message
            level: Log level (INFO, WARNING, ERROR)
        """
        log_func = getattr(self.logger, level.lower())
        log_func(f"[{stage}] {message}")


class MetaAgentPipeline:
    """
    Pipeline for chaining multiple agents together.
    
    This class enables sequential execution of multiple agents,
    passing outputs from one stage as inputs to the next.
    """
    
    def __init__(self, agents: List[MetaAgentBase]):
        """
        Initialize the pipeline with a list of agents.
        
        Args:
            agents: List of agents to execute in sequence
        """
        self.agents = agents
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """
        Execute the pipeline on a batch of inputs.
        
        Args:
            inputs: List of input data
            
        Returns:
            List[ReasoningOutput]: Final outputs after all stages
        """
        current_data = inputs
        
        for i, agent in enumerate(self.agents):
            self.logger.info(f"Executing agent {i+1}/{len(self.agents)}: {agent.__class__.__name__}")
            
            # Process with current agent
            results = agent.process_batch(current_data)
            
            # Convert outputs to inputs for next stage
            if i < len(self.agents) - 1:
                current_data = [
                    ReasoningInput(
                        question=r.question,
                        answer="",  # Next stage may not need answer
                        task_description=r.task_description,
                        principles=r.principles,
                        metadata=r.metadata
                    )
                    for r in results
                ]
        
        return results
