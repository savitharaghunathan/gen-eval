"""
GenEval: A Unified Evaluation Framework for Generative AI Applications

This package provides a unified interface for evaluating generative AI models
across different frameworks like RAGAS and DeepEval.
"""

__version__ = "0.1.0"
__author__ = "Savitha Raghunathan"

from .framework import GenEvalFramework
from .schemas import Input, Output, MetricResult
from .llm import LLMInitializer

__all__ = [
    "GenEvalFramework",
    "Input", 
    "Output",
    "MetricResult",
    "LLMInitializer",
] 