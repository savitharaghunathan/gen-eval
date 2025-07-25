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
from geneval.llm import LLMInitializer


class DeepEvalAdapter:
    """
    Adapter for DeepEval metrics
    """

    def __init__(self, llm_initializer: LLMInitializer = None):
        """
        Initialize DeepEval client and available metrics
        
        Args:
            llm_initializer: Optional LLMInitializer instance for LLM configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DeepEvalAdapter")

        # Initialize LLM if provided
        self.llm_initializer = llm_initializer
        if self.llm_initializer:
            self.llm_config = self.llm_initializer.configure_deepeval_llm()
            self.logger.info(f"LLM configured with provider: {self.llm_initializer.get_selected_provider()}")
        else:
            self.llm_config = {}
            self.logger.warning("No LLM provided")

        # Initialize metrics with LLM configuration if available
        try:
            if self.llm_initializer and hasattr(self.llm_initializer, 'selected_provider'):
              
                self.available_metrics = {
                    "answer_relevance": AnswerRelevancyMetric(),
                    "context_relevance": ContextualRelevancyMetric(),
                    "faithfulness": FaithfulnessMetric(),
                    "context_recall": ContextualRecallMetric(),
                    "context_precision": ContextualPrecisionMetric(),
                }
            else:
                # No LLM available - initialize empty metrics
                self.logger.warning("No LLM provided, DeepEval metrics will not be available")
                self.available_metrics = {}
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepEval metrics: {e}")
            self.available_metrics = {}
        
        # Set supported metrics based on available metrics
        self.supported_metrics = list(self.available_metrics.keys())


    def _create_test_case(self, input: Input) -> LLMTestCase:
        """
        Convert input to DeepEval LLMTestCase format
        """
        # Handle retrieval_context - DeepEval expects a list of strings
        if isinstance(input.retrieval_context, str):
            # Split multiline context into separate context pieces
            context = [ctx.strip() for ctx in input.retrieval_context.split('\n\n') if ctx.strip()]
        elif isinstance(input.retrieval_context, list):
            context = input.retrieval_context
        else:
            context = [str(input.retrieval_context)]
        
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
            
            # Prepare metadata
            self.logger.info(f"Preparing metadata")
            metadata = {
                "framework": "deepeval",
                "total_metrics": len(metric_results),
                "evaluation_successful": True,
                "test_case_count": 1
            }
            
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

            self.logger.info(f"Returning error output: {metadata}")
            return Output(
                metrics=[],
                metadata=metadata
            )