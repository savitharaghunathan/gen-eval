"""
GenEval: A Unified Evaluation Framework for Generative AI Applications

This package provides a unified interface for evaluating generative AI models
across different frameworks like RAGAS and DeepEval.
"""

__version__ = "0.1.0"
__author__ = "Savitha Raghunathan"

from .framework import GenEvalFramework
from .llm_manager import LLMManager
from .schemas import Input, Output, MetricResult

__all__ = [
    "GenEvalFramework",
    "LLMManager",
    "Input", 
    "Output",
    "MetricResult",
] 