"""
Main framework for GenEval evaluation system.

This module provides the unified interface for evaluating generative AI models
across different frameworks like RAGAS and DeepEval.
"""

import logging
from typing import Dict
from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.llm import LLMInitializer
from geneval.schemas import Input, Output


class GenEvalFramework:
    """
    Main framework for evaluating LLMs using multiple evaluation frameworks
    """
    
    def __init__(self, llm_initializer: LLMInitializer = None):
        """
        Initialize the GenEval framework
        
        Args:
            llm_initializer: Optional LLMInitializer instance for LLM configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GenEvalFramework")
        
        # Initialize adapters
        self.adapters = {
            "ragas": RAGASAdapter(llm_initializer),
            "deepeval": DeepEvalAdapter(llm_initializer)
        }
        
        self.logger.info("GenEvalFramework initialized with adapters: %s", list(self.adapters.keys()))

    def evaluate(self, **kwargs) -> Dict:
        """
        Evaluate the model's response using the appropriate adapter
        
        Args:
            **kwargs: Arguments to create Input object (question, response, retrieval_context, reference, metrics)
            
        Returns:
            Output: Evaluation results
        """
        data = Input(**kwargs)
        self.logger.info(f"Starting evaluation for metrics: {data.metrics}")
        
        results = {}

        for metric_str in data.metrics:
            if "." in metric_str:
                adapter_name, metric_name = metric_str.split(".", 1)
                adapter = self.adapters.get(adapter_name)
                if not adapter:
                    raise ValueError(f"Unknown adapter: {adapter_name}")
                if metric_name not in adapter.supported_metrics:
                    raise ValueError(f"Adapter '{adapter_name}' does not support metric '{metric_name}'")
                
                # Create input with only this metric
                single_metric_input = Input(
                    question=data.question,
                    response=data.response,
                    retrieval_context=data.retrieval_context,
                    reference=data.reference,
                    metrics=[metric_name]
                )
                raw = adapter.evaluate(single_metric_input)
                key = metric_str
                results[key] = (adapter_name, raw)
            else:
                metric_name = metric_str
                # all adapters supporting this metric
                candidates = [name for name, ad in self.adapters.items()
                              if metric_name in ad.supported_metrics]
                if not candidates:
                    raise ValueError(f"No adapter supports metric '{metric_name}'")
                for adapter_name in candidates:
                    adapter = self.adapters[adapter_name]
                    
                    # Create input with only this metric
                    single_metric_input = Input(
                        question=data.question,
                        response=data.response,
                        retrieval_context=data.retrieval_context,
                        reference=data.reference,
                        metrics=[metric_name]
                    )
                    raw = adapter.evaluate(single_metric_input)
                    key = f"{adapter_name}.{metric_name}"
                    results[key] = (adapter_name, raw)
            self.logger.debug("Raw result %s: %s", metric_str, results[key][1])
    
        self.logger.info(f"Evaluation completed with {len(results)} results")
        return results