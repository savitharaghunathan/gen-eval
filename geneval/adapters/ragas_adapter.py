import logging
from typing import List, Dict, Any
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    ResponseRelevancy,
    Faithfulness,
)
from ragas import evaluate
from datasets import Dataset
from geneval.schemas import Input, MetricResult, Output
from geneval.llm import LLMInitializer



class RAGASAdapter:
    """
    Adapter for RAGAs metrics
    """
 

    def __init__(self, llm_initializer: LLMInitializer = None):
        """
        Initialize the RAGASAdapter
        
        Args:
            llm_initializer: Optional LLMInitializer instance for LLM configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RAGASAdapter")

        # Initialize LLM if provided
        self.llm_initializer = llm_initializer
        if self.llm_initializer:
            self.llm_config = self.llm_initializer.configure_ragas_llm()
            self.logger.info(f"LLM configured with provider: {self.llm_initializer.get_selected_provider()}")
        else:
            self.llm_config = {}
            self.logger.warning("No LLM initializer provided")

        # Initialize metrics with LLM configuration if available
        try:
            if self.llm_initializer and self.llm_initializer.selected_provider:
                # Initialize with configured LLM
                self.available_metrics = {
                    "context_precision_without_reference": LLMContextPrecisionWithoutReference(),
                    "context_precision_with_reference": LLMContextPrecisionWithReference(),
                    "context_recall": LLMContextRecall(),
                    "context_entity_recall": ContextEntityRecall(),
                    "noise_sensitivity": NoiseSensitivity(),
                    "response_relevancy": ResponseRelevancy(),
                    "faithfulness": Faithfulness()
                }
                self.logger.info(f"RAGAS metrics initialized successfully with {len(self.available_metrics)} metrics")
            else:
                # No LLM available - initialize empty metrics
                self.logger.warning("No LLM provided, RAGAS metrics will not be available")
                self.available_metrics = {}
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGAS metrics: {e}")
            self.available_metrics = {}
        
        # Set supported metrics based on available metrics
        self.supported_metrics = list(self.available_metrics.keys())


    def _prepare_dataset(self, input: Input) -> Dataset:
        """
        Convert input to RAGAS-compatible dataset format
        """
        self.logger.info(f"Preparing dataset for input: {input}")
        
        # Handle retrieval_context - ensure it's a list of strings
        if isinstance(input.retrieval_context, str):
            # Split multiline context into separate context pieces
            contexts = [ctx.strip() for ctx in input.retrieval_context.split('\n\n') if ctx.strip()]
        elif isinstance(input.retrieval_context, list):
            contexts = input.retrieval_context
        else:
            contexts = [str(input.retrieval_context)]
        
        data = {
            "question": [input.question],
            "contexts": [contexts],
            "answer": [input.response],
            "ground_truths": [[input.reference]],
            "reference": [input.reference]
        }
        self.logger.info(f"Dataset prepared with {len(contexts)} context pieces")
        return Dataset.from_dict(data)

    def _get_metrics(self, metric_names: List[str]) -> List:
        """
        Get RAGAS metric objects for the requested metrics
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
        Evaluate the model's response using RAGAS metrics
        """
        try:
            # Prepare dataset
            dataset = self._prepare_dataset(input)
            
            # Get requested metrics
            ragas_metrics = self._get_metrics(input.metrics)
            
            # Run evaluation (following your notebook example)
            results = evaluate(
                dataset,
                ragas_metrics,
                column_map={
                    "question": "question",
                    "contexts": "contexts",
                    "answer": "answer",
                    "ground_truths": "ground_truths",
                    "reference": "reference"
                }
            )
            
            # Extract scores from results.scores[0] 
            metric_results = []
            if hasattr(results, 'scores') and len(results.scores) > 0:
                scores_dict = results.scores[0]
                for metric_name in input.metrics:
                    # Map metric names to actual RAGAS result keys
                    actual_key = None
                    for key in scores_dict.keys():
                        if metric_name.lower() in key.lower() or key.lower().endswith(metric_name.lower()):
                            actual_key = key
                            break
                    
                    if actual_key:
                        score = float(scores_dict[actual_key])
                        metric_results.append(
                            MetricResult(
                                name=metric_name,
                                score=score,
                                tool_name="ragas",
                                details=f"RAGAS {metric_name} evaluation"
                            )
                        )
                    else:
                        self.logger.error(f"Metric '{metric_name}' not found in RAGAS results. Available: {list(scores_dict.keys())}")
            else:
                self.logger.error(f"RAGAS results format unexpected: {type(results)}")
            
            # Prepare metadata
            self.logger.info(f"Preparing metadata")
            metadata = {
                "framework": "ragas",
                "total_metrics": len(metric_results),
                "evaluation_successful": True
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
                "framework": "ragas",
                "error": str(e),
                "evaluation_successful": False
            }

            self.logger.info(f"Returning error output: {metadata}")
            return Output(
                metrics=[],
                metadata=metadata
            )