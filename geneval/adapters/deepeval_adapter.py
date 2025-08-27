import logging
from typing import List, Dict, Any, Optional
from deepeval import evaluate
from deepeval.models import GPTModel, AzureOpenAIModel, AnthropicModel, AmazonBedrockModel, GeminiModel, DeepSeekModel, OllamaModel, DeepEvalBaseLLM
import httpx
import asyncio
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


class VLLMModel(DeepEvalBaseLLM):
    """Custom vLLM model for DeepEval that supports any model via OpenAI-compatible API"""
    
    def __init__(self, base_url: str, model_name: str, api_path: str = "/v1", 
                 api_key: str = "dummy-key", temperature: float = 0.1, ssl_verify: bool = True):
        self.base_url = base_url
        self.model_name = model_name
        self.api_path = api_path
        self.api_key = api_key
        self.temperature = temperature
        self.ssl_verify = ssl_verify
        
        # Create HTTP clients for both sync and async operations
        timeout = httpx.Timeout(30.0)
        
        self.sync_client = httpx.Client(
            verify=self.ssl_verify,
            timeout=timeout
        )
        
        self.async_client = httpx.AsyncClient(
            verify=self.ssl_verify,
            timeout=timeout
        )
        
    def get_model_name(self) -> str:
        return self.model_name
        
    def load_model(self):
        # No local model to load; interactions are via API
        pass
        
    def generate(self, prompt: str) -> str:
        """Generate text using vLLM's OpenAI-compatible API"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key != "dummy-key":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],  # Chat format
            "temperature": self.temperature
        }
        
        try:
            # Use httpx
            endpoint_url = f"{self.base_url}{self.api_path}/chat/completions"
            
            response = self.sync_client.post(
                endpoint_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP error calling vLLM API: {e.response.status_code} - {e.response.text}")
        except httpx.ConnectError as e:
            raise RuntimeError(f"Connection error connecting to vLLM server: {e}")
        except Exception as e:
            raise RuntimeError(f"Error calling vLLM API: {e}")
    
    async def a_generate(self, prompt: str) -> str:
        """Async version of generate using httpx async client"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key != "dummy-key":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        
        try:
            endpoint_url = f"{self.base_url}{self.api_path}/chat/completions"
            
            response = await self.async_client.post(
                endpoint_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP error calling vLLM API: {e.response.status_code} - {e.response.text}")
        except httpx.ConnectError as e:
            raise RuntimeError(f"Connection error connecting to vLLM server: {e}")
        except Exception as e:
            raise RuntimeError(f"Error calling vLLM API: {e}")
    
    def __del__(self):
        """Clean up HTTP clients"""
        if hasattr(self, 'sync_client'):
            self.sync_client.close()
        if hasattr(self, 'async_client'):
            # For async client cleanup in destructor, we need to handle it carefully
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule the cleanup
                    loop.create_task(self.async_client.aclose())
                else:
                    # If no loop is running, run it synchronously
                    asyncio.run(self.async_client.aclose())
            except:
                # If there are issues with cleanup, just pass
                pass

class DeepEvalAdapter:
    """
    Adapter for DeepEval metrics with support for multiple LLM providers
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

        # Initialize model for DeepEval
        self.model = self._create_model()
        
        # Initialize metrics
        try:
            self.available_metrics = {
                "answer_relevancy": AnswerRelevancyMetric(model=self.model),
                "context_relevance": ContextualRelevancyMetric(model=self.model),
                "faithfulness": FaithfulnessMetric(model=self.model),
                "context_recall": ContextualRecallMetric(model=self.model),
                "context_precision": ContextualPrecisionMetric(model=self.model),
            }
            self.logger.info(f"DeepEval metrics initialized successfully with {len(self.available_metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepEval metrics: {e}")
            raise RuntimeError(f"Failed to initialize DeepEval metrics: {e}")
        
        # Set supported metrics based on available metrics
        self.supported_metrics = list(self.available_metrics.keys())

    def _create_model(self):
        """Create DeepEval model instance based on LLM configuration"""
        provider = self.llm_info.get('provider')
        
        # Get DeepEval-compatible configuration from LLM manager
        deepeval_config = self.llm_manager.get_deepeval_config(provider)
        
        if provider == "vllm":
            if not deepeval_config.get("model"):
                raise ValueError("vLLM model not specified in configuration")
            
            # Create custom VLLMModel
            vllm_model = VLLMModel(
                base_url=deepeval_config.get("base_url", "http://localhost:8000"),
                model_name=deepeval_config["model"],
                api_path=deepeval_config.get("api_path", "/v1"),
                api_key=deepeval_config.get("api_key", "dummy-key"),
                temperature=deepeval_config["temperature"],
                ssl_verify=deepeval_config.get("ssl_verify", True)
            )
            
            self.logger.info(f"Created custom vLLM model: {deepeval_config['model']}, endpoint: {deepeval_config.get('base_url')}{deepeval_config.get('api_path', '/v1')}")
            return vllm_model
        
        elif provider == "openai":
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
            raise ValueError(f"Provider {provider} not fully supported for DeepEval. Only OpenAI, Azure OpenAI, Anthropic, Amazon Bedrock, Gemini, DeepSeek, Ollama and VLLM providers are currently supported.")

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