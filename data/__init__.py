"""
Data processing module initialization.
"""

from .processor import DatasetProcessor, DatasetRegistry, get_processor, get_registry

__all__ = [
    'DatasetProcessor',
    'DatasetRegistry',
    'get_processor',
    'get_registry'
]
