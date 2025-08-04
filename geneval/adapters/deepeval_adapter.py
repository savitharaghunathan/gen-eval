import logging
from typing import List, Dict, Any
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,

)
from deepeval.test_case import LLMTestCase
from geneval.schemas import Input, MetricResult, Output
from geneval.llm_manager import LLMManager


class DeepEvalAdapter:
    """
    Adapter for DeepEval metrics
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize DeepEval client and available metrics
        
        Args:
            llm_manager: LLMManager instance for LLM configuration (required)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DeepEvalAdapter")

        if not llm_manager:
            raise ValueError("LLMManager is required for DeepEvalAdapter initialization")

        # Initialize LLM
        self.llm_manager = llm_manager
        self.llm_config = self.llm_manager.configure_for_deepeval()
        self.llm_info = self.llm_manager.get_llm_info()
        
        if not self.llm_manager.get_llm():
            raise ValueError("No LLM available. Please configure an LLM provider.")
        
        self.logger.info(f"LLM configured with provider: {self.llm_info.get('provider', 'unknown')}")

        # Initialize metrics with LLM configuration
        try:
            self.available_metrics = {
                "answer_relevance": AnswerRelevancyMetric(),
                "context_relevance": ContextualRelevancyMetric(),
                "faithfulness": FaithfulnessMetric(),
                "context_recall": ContextualRecallMetric(),
                "context_precision": ContextualPrecisionMetric(),
            }
            self.logger.info(f"DeepEval metrics initialized successfully with {len(self.available_metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepEval metrics: {e}")
            raise RuntimeError(f"Failed to initialize DeepEval metrics: {e}")
        
        # Set supported metrics based on available metrics
        self.supported_metrics = list(self.available_metrics.keys())


    def _create_test_case(self, input: Input) -> LLMTestCase:
        """
        Convert input to DeepEval LLMTestCase format
        """
        # Treat context as a simple string
        context = [input.retrieval_context]
        
        return LLMTestCase(
            input=input.question,
            actual_output=input.response,
            expected_output=input.reference,
            retrieval_context=context
        )

    def _get_metrics(self, metric_names: List[str]) -> List:
        """
        Get DeepEval metric objects for the requested metrics
        """
        metrics = []
        for metric_name in metric_names:
            if metric_name in self.available_metrics:
                self.logger.info(f"Supported metric: {metric_name}")
                metrics.append(self.available_metrics[metric_name])
            else:
                self.logger.warning(f"Unsupported metric: {metric_name}")
                raise ValueError(f"Unsupported metric: {metric_name}")
        return metrics

    def evaluate(self, input: Input) -> Output:
        """
        Evaluate the model's response using DeepEval metrics
        """
        try:
            # Create test case
            self.logger.info(f"Creating test case for input: {input}")
            test_case = self._create_test_case(input)
            
            # Get requested metrics
            self.logger.info(f"Getting metrics for input: {input.metrics}")
            deepeval_metrics = self._get_metrics(input.metrics)
            
            # Run evaluation (following your notebook example)
            metric_results = []
            
            for i, metric_name in enumerate(input.metrics):
                if i < len(deepeval_metrics):
                    metric_obj = deepeval_metrics[i]
                    # Call measure() directly on each metric (following your example)
                    score = metric_obj.measure(test_case)
                    
                    # Get explanation if available
                    explanation = f"DeepEval {metric_name} evaluation"
                    if hasattr(metric_obj, 'reason') and metric_obj.reason:
                        explanation = metric_obj.reason
                    
                    metric_results.append(
                        MetricResult(
                            name=metric_name,
                            score=float(score),
                            details=explanation,
                            tool_name="deepeval"
                        )
                    )
            
            # Prepare metadata with LLM information
            self.logger.info(f"Preparing metadata")
            metadata = {
                "framework": "deepeval",
                "total_metrics": len(metric_results),
                "evaluation_successful": True,
                "test_case_count": 1
            }
            
            # Add LLM information if available
            if self.llm_info:
                metadata.update({
                    "llm_provider": self.llm_info.get("provider"),
                    "llm_model": self.llm_info.get("model")
                })
            
            self.logger.info(f"Returning output")
            return Output(
                metrics=metric_results,
                metadata=metadata
            )
            
        except Exception as e:
            # Return error result
            self.logger.error(f"Error in evaluation: {e}")
            metadata = {
                "framework": "deepeval",
                "error": str(e),
                "evaluation_successful": False
            }
            
            # Add LLM information if available
            if self.llm_info:
                metadata.update({
                    "llm_provider": self.llm_info.get("provider"),
                    "llm_model": self.llm_info.get("model")
                })

            self.logger.info(f"Returning error output: {metadata}")
            return Output(
                metrics=[],
                metadata=metadata
            )