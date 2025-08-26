import logging
from typing import List, Dict, Any
from deepeval import evaluate
from deepeval.models import GPTModel, AzureOpenAIModel, AnthropicModel, AmazonBedrockModel, GeminiModel, DeepSeekModel, OllamaModel
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
    Adapter for DeepEval metrics using GPTModel for OpenAI integration
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

        # Initialize LLM manager
        self.llm_manager = llm_manager
        
        # Get default provider and configuration
        default_provider = self.llm_manager.get_default_provider()
        if not default_provider:
            raise ValueError("No default LLM provider configured. Please set 'default: true' for one provider in the config.")
        
        self.llm_info = {
            "provider": default_provider,
            "model": self.llm_manager.get_provider_config(default_provider).get("model", "unknown"),
            "provider_config": self.llm_manager.get_provider_config(default_provider),
            "global_settings": self.llm_manager.get_global_settings()
        }
        
        self.logger.info(f"LLM configured with provider: {self.llm_info.get('provider', 'unknown')}")

        # Initialize GPTModel for DeepEval
        self.gpt_model = self._create_gpt_model()
        
        # Initialize metrics with GPTModel configuration
        try:
            self.available_metrics = {
                "answer_relevancy": AnswerRelevancyMetric(model=self.gpt_model),
                "context_relevance": ContextualRelevancyMetric(model=self.gpt_model),
                "faithfulness": FaithfulnessMetric(model=self.gpt_model),
                "context_recall": ContextualRecallMetric(model=self.gpt_model),
                "context_precision": ContextualPrecisionMetric(model=self.gpt_model),
            }
            self.logger.info(f"DeepEval metrics initialized successfully with {len(self.available_metrics)} metrics using GPTModel")
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepEval metrics: {e}")
            raise RuntimeError(f"Failed to initialize DeepEval metrics: {e}")
        
        # Set supported metrics based on available metrics
        self.supported_metrics = list(self.available_metrics.keys())

    def _create_gpt_model(self):
        """
        Create DeepEval model instance based on LLM configuration
        """
        provider = self.llm_info.get('provider')
        
        # Get DeepEval-compatible configuration from LLM manager
        deepeval_config = self.llm_manager.get_deepeval_config(provider)
        
        if provider == "openai":
            # Use the model from config - no fallback to hardcoded values
            if not deepeval_config.get("model"):
                raise ValueError("OpenAI model not specified in configuration. Please set a model in your llm_config.yaml")
            
            # Create GPTModel with OpenAI configuration
            gpt_model_kwargs = {
                "model": deepeval_config["model"],
                "temperature": deepeval_config["temperature"]
            }
            
            gpt_model = GPTModel(**gpt_model_kwargs)
            
            self.logger.info(f"Created GPTModel with OpenAI provider: {deepeval_config['model']}, temperature: {deepeval_config['temperature']}")
            return gpt_model
            
        elif provider == "azure_openai":
            # Handle Azure OpenAI
            if not deepeval_config.get("model"):
                raise ValueError("Azure OpenAI model not specified in configuration")
            
            # Get Azure-specific configuration
            deployment_name = deepeval_config.get('deployment_name')
            azure_openai_api_key = deepeval_config.get('azure_openai_api_key')
            openai_api_version = deepeval_config.get('openai_api_version', '2025-01-01-preview')
            azure_endpoint = deepeval_config.get('azure_endpoint')
            
            if not all([deployment_name, azure_openai_api_key, azure_endpoint]):
                raise ValueError("Azure OpenAI configuration incomplete. Required: deployment_name, azure_openai_api_key, azure_endpoint")
            
            # Create AzureOpenAIModel
            azure_model = AzureOpenAIModel(
                model_name=deepeval_config["model"],
                deployment_name=deployment_name,
                azure_openai_api_key=azure_openai_api_key,
                openai_api_version=openai_api_version,
                azure_endpoint=azure_endpoint,
                temperature=deepeval_config["temperature"]
            )
            
            self.logger.info(f"Created AzureOpenAIModel: {deepeval_config['model']}, deployment: {deployment_name}, endpoint: {azure_endpoint}")
            return azure_model
            
        elif provider == "anthropic":
            if not deepeval_config.get("model"):
                raise ValueError("Anthropic model not specified in configuration")
            
            # Create AnthropicModel
            anthropic_model = AnthropicModel(
                model=deepeval_config["model"],
                temperature=deepeval_config["temperature"]
            )
            
            self.logger.info(f"Created AnthropicModel: {deepeval_config['model']}, temperature: {deepeval_config['temperature']}")
            return anthropic_model
            
        elif provider == "amazon_bedrock":
            if not deepeval_config.get("model"):
                raise ValueError("Amazon Bedrock model not specified in configuration")
            
            # Create AmazonBedrockModel
            amazon_bedrock_model = AmazonBedrockModel(
                model_id=deepeval_config["model"],
                temperature=deepeval_config["temperature"]
            )
            
            self.logger.info(f"Created AmazonBedrockModel: {deepeval_config['model']}, temperature: {deepeval_config['temperature']}")
            return amazon_bedrock_model
            
        elif provider == "gemini":
            if not deepeval_config.get("model"):
                raise ValueError("Gemini model not specified in configuration")
            
            # Get API key from config
            api_key = deepeval_config.get('api_key')
            if not api_key:
                raise ValueError("Google API key not found in configuration")
            
            # Create GeminiModel
            gemini_model = GeminiModel(
                model_name=deepeval_config["model"],
                api_key=api_key,
                temperature=deepeval_config["temperature"]
            )
            
            self.logger.info(f"Created GeminiModel: {deepeval_config['model']}, temperature: {deepeval_config['temperature']}")
            return gemini_model
            
        elif provider == "deepseek":
            if not deepeval_config.get("model"):
                raise ValueError("DeepSeek model not specified in configuration")
            
            # Get API key from config
            api_key = deepeval_config.get('api_key')
            if not api_key:
                raise ValueError("DeepSeek API key not found in configuration")
            
            # Create DeepSeekModel
            deepseek_model = DeepSeekModel(
                model=deepeval_config["model"],
                api_key=api_key,
                temperature=deepeval_config["temperature"]
            )
            
            self.logger.info(f"Created DeepSeekModel: {deepeval_config['model']}, temperature: {deepeval_config['temperature']}")
            return deepseek_model
            
        elif provider == "ollama":
            if not deepeval_config.get("model"):
                raise ValueError("Ollama model not specified in configuration")
            
            # Get base URL from config
            base_url = deepeval_config.get('base_url', 'http://localhost:11434')
            
            # Create OllamaModel
            ollama_model = OllamaModel(
                model=deepeval_config["model"],
                base_url=base_url,
                temperature=deepeval_config["temperature"]
            )
            
            self.logger.info(f"Created OllamaModel: {deepeval_config['model']}, temperature: {deepeval_config['temperature']}")
            return ollama_model
            
        else:
            # For other providers, raise an error
            raise ValueError(f"Provider {provider} not fully supported for DeepEval. Only OpenAI, Azure OpenAI, Anthropic, Amazon Bedrock, Gemini, DeepSeek, and Ollama providers are currently supported.")

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