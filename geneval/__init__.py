"""
GenEval: A Unified Evaluation Framework for Generative AI Applications

This package provides a unified interface for evaluating generative AI models
across different frameworks like RAGAS and DeepEval.
"""

__version__ = "0.1.0"
__author__ = "Savitha Raghunathan"

from .exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from .framework import GenEvalFramework
from .llm_manager import LLMManager
from .profile_manager import ProfileManager
from .schemas import BatchResult, Input, MetricEvaluation, MetricResult, Output, ProfileResult

__all__ = [
    "GenEvalFramework",
    "LLMManager",
    "ProfileManager",
    "Input",
    "Output",
    "MetricResult",
    "MetricEvaluation",
    "ProfileResult",
    "BatchResult",
    "ProfileValidationError",
    "UnknownMetricError",
    "ProfileNotFoundError",
]
