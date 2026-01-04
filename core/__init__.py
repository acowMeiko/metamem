"""
Core module for MetaEvo framework.
Provides base classes and concrete implementations for meta-reasoning agents.
"""

from .base import MetaAgentBase
from .stages import StageOneAgent, StageTwoAgent, InferenceAgent

__all__ = [
    'MetaAgentBase',
    'StageOneAgent',
    'StageTwoAgent',
    'InferenceAgent'
]
