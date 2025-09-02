import logging
import os
import httpx
from typing import List, Dict, Any, Optional
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    AnswerRelevancy,
    Faithfulness,
)
from ragas import evaluate
from datasets import Dataset
from geneval.schemas import Input, MetricResult, Output
from geneval.llm_manager import LLMManager

# LangChain imports for LLM creation
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

# RAGAS LangChain integration
from ragas.integrations.langchain import LangchainLLMWrapper


class RAGASAdapter:
    """
    Adapter for RAGAs metrics with integrated LangChain LLM management
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the RAGASAdapter
        
        Args:
            llm_manager: LLMManager instance for LLM configuration (required)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RAGASAdapter")

        if not llm_manager:
            raise ValueError("LLMManager is required for RAGASAdapter initialization")

        # Initialize LLM manager
        self.llm_manager = llm_manager
        
        # Get default provider and create RAGAS-compatible LLM
        default_provider = self.llm_manager.get_default_provider()
        if not default_provider:
            raise ValueError("No default LLM provider configured. Please set 'default: true' for one provider in the config.")
        
        self.llm = self._create_langchain_llm(default_provider)
        
        if not self.llm:
            raise ValueError("No LLM available. Please configure an LLM provider.")
        
        self.llm_info = {
            "provider": default_provider,
            "model": self.llm_manager.get_provider_config(default_provider).get("model", "unknown"),
            "provider_config": self.llm_manager.get_provider_config(default_provider),
            "global_settings": self.llm_manager.get_global_settings()
        }
        
        self.logger.info(f"LLM configured with provider: {self.llm_info.get('provider', 'unknown')}")

        # Initialize metrics with LLM configuration
        try:
            # For LLM-dependent metrics, we need to pass the LLM instance
            # RAGAS expects LLM instances to have certain methods
            self.available_metrics = {
                "context_precision_without_reference": LLMContextPrecisionWithoutReference(llm=self.llm),
                "context_precision_with_reference": LLMContextPrecisionWithReference(llm=self.llm),
                "context_recall": LLMContextRecall(llm=self.llm),
                "context_entity_recall": ContextEntityRecall(),
                "noise_sensitivity": NoiseSensitivity(),
                "answer_relevancy": AnswerRelevancy(),
                "faithfulness": Faithfulness()
            }
            self.logger.info(f"RAGAS metrics initialized successfully with {len(self.available_metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGAS metrics: {e}")
            raise RuntimeError(f"Failed to initialize RAGAS metrics: {e}")
        
        # Set supported metrics based on available metrics
        self.supported_metrics = list(self.available_metrics.keys())

    def _create_langchain_llm(self, provider_name: str) -> Optional[LLM]:
        """Create LangChain LLM instance and wrap it for RAGAS compatibility"""
        try:
            langchain_llm = None
            if provider_name == "openai":
                langchain_llm = self._create_openai_provider(provider_name)
            elif provider_name == "azure_openai":
                langchain_llm = self._create_azure_openai_provider(provider_name)
            elif provider_name == "anthropic":
                langchain_llm = self._create_anthropic_provider(provider_name)
            elif provider_name == "amazon_bedrock":
                langchain_llm = self._create_amazon_bedrock_provider(provider_name)
            elif provider_name == "gemini":
                langchain_llm = self._create_gemini_provider(provider_name)
            elif provider_name == "deepseek":
                langchain_llm = self._create_deepseek_provider(provider_name)
            elif provider_name == "ollama":
                langchain_llm = self._create_ollama_provider(provider_name)
            elif provider_name == "vllm":
                langchain_llm = self._create_vllm_provider(provider_name)
            else:
                self.logger.warning(f"Unknown provider: {provider_name}")
                return None
            
            # Wrap the LangChain LLM with RAGAS's official wrapper
            if langchain_llm:
                return LangchainLLMWrapper(langchain_llm)
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating {provider_name} provider: {e}")
            return None



    def _create_openai_provider(self, provider_name: str) -> Optional[ChatOpenAI]:
        """Create OpenAI provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        global_settings = self.llm_manager.get_global_settings()
        
        # Only read from environment variable, never hardcode
        api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        
        # Get model from config 
        model = provider_config.get("model")
        if not model:
            self.logger.error("OpenAI model not specified in configuration")
            return None
        
        return ChatOpenAI(
            model=model,
            temperature=global_settings.get("temperature", 0.1),
            max_tokens=global_settings.get("max_tokens", 1000),
            timeout=global_settings.get("timeout", 30)
        )

    def _create_azure_openai_provider(self, provider_name: str) -> Optional[ChatOpenAI]:
        """Create Azure OpenAI provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        global_settings = self.llm_manager.get_global_settings()
        
        # Get Azure-specific configuration
        api_key = provider_config.get("azure_openai_api_key")
        if not api_key:
            self.logger.warning("Azure OpenAI API key not found in configuration")
            return None
        
        # Get model from config
        model = provider_config.get("model")
        if not model:
            self.logger.error("Azure OpenAI model not specified in configuration")
            return None
        
        # Get deployment name
        deployment_name = provider_config.get("deployment_name")
        if not deployment_name:
            self.logger.error("Azure OpenAI deployment name not specified in configuration")
            return None
        
        # Get Azure endpoint
        azure_endpoint = provider_config.get("azure_endpoint")
        if not azure_endpoint:
            self.logger.error("Azure OpenAI endpoint not specified in configuration")
            return None
        
        # Get API version
        openai_api_version = provider_config.get("openai_api_version", "2025-01-01-preview")
        
        return ChatOpenAI(
            model=model,
            temperature=global_settings.get("temperature", 0.1),
            max_tokens=global_settings.get("max_tokens", 1000),
            timeout=global_settings.get("timeout", 30),
            openai_api_version=openai_api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
            openai_api_key=api_key
        )

    def _create_anthropic_provider(self, provider_name: str) -> Optional[ChatAnthropic]:
        """Create Anthropic provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        global_settings = self.llm_manager.get_global_settings()
        
        # Only read from environment variable, never hardcode
        api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        
        # Get model from config
        model = provider_config.get("model")
        if not model:
            self.logger.error("Anthropic model not specified in configuration")
            return None
        
        return ChatAnthropic(
            model=model,
            temperature=global_settings.get("temperature", 0.1),
            max_tokens=global_settings.get("max_tokens", 1000)
        )

    def _create_amazon_bedrock_provider(self, provider_name: str) -> Optional[LLM]:
        """Create Amazon Bedrock provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        global_settings = self.llm_manager.get_global_settings()
        
        # Get AWS credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key_id or not aws_secret_access_key:
            self.logger.warning("AWS credentials not found in environment variables")
            return None
        
        # Get model from config
        model = provider_config.get("model")
        if not model:
            self.logger.error("Amazon Bedrock model not specified in configuration")
            return None
        
        # Get region from config
        region_name = provider_config.get("region_name", "us-east-1")
        
        # For now, return None as we'll use DeepEval's AmazonBedrockModel directly
        # This method is kept for consistency with the provider structure
        self.logger.info(f"Amazon Bedrock provider configured: {model}, region: {region_name}")
        return None

    def _create_gemini_provider(self, provider_name: str) -> Optional[ChatGoogleGenerativeAI]:
        """Create Google Gemini provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        global_settings = self.llm_manager.get_global_settings()
        
        # Only read from environment variable, never hardcode
        api_key_env = provider_config.get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        
        # Get model from config
        model = provider_config.get("model")
        if not model:
            self.logger.error("Gemini model not specified in configuration")
            return None
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=global_settings.get("temperature", 0.1),
            max_output_tokens=global_settings.get("max_tokens", 1000)
        )

    def _create_deepseek_provider(self, provider_name: str) -> Optional[LLM]:
        """Create DeepSeek provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        
        # Only read from environment variable, never hardcode
        api_key_env = provider_config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        
        # Get model from config
        model = provider_config.get("model")
        if not model:
            self.logger.error("DeepSeek model not specified in configuration")
            return None
        
        # For now, return None as we'll use DeepEval's DeepSeekModel directly
        # This method is kept for consistency with the provider structure
        self.logger.info(f"DeepSeek provider configured: {model}")
        return None

    def _create_ollama_provider(self, provider_name: str) -> Optional[Ollama]:
        """Create Ollama provider"""
        provider_config = self.llm_manager.get_provider_config(provider_name)
        global_settings = self.llm_manager.get_global_settings()
        
        base_url = provider_config.get("base_url", "http://localhost:11434")
        
        # Get model from config
        model = provider_config.get("model")
        if not model:
            self.logger.error("Ollama model not specified in configuration")
            return None
        
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=global_settings.get("temperature", 0.1)
        )

    def _create_vllm_provider(self, provider_name: str) -> Optional[ChatOpenAI]:
        """Create vLLM provider using OpenAI-compatible interface"""
        try:
            provider_config = self.llm_manager.get_provider_config(provider_name)
            global_settings = self.llm_manager.get_global_settings()
            
            # Get configuration parameters
            base_url = self.llm_manager.get_base_url(provider_name) or "http://localhost:8000"
            model = provider_config.get("model")
            ssl_verify = provider_config.get("ssl_verify", True)
            api_path = self.llm_manager.get_api_path(provider_name) or "/v1"
            
            # Read API key from environment variable
            api_key_env = provider_config.get("api_key_env")
            api_key = "dummy-key"  # Default for vLLM when no auth is needed
            
            if api_key_env:
                env_api_key = os.getenv(api_key_env)
                if env_api_key:
                    api_key = env_api_key
                else:
                    self.logger.warning(f"{api_key_env} not found in environment variables, using dummy key")
            
            # Also check for direct api_key in config as fallback
            elif provider_config.get("api_key"):
                api_key = provider_config.get("api_key")
            
            if not model:
                raise ValueError("vLLM model not specified in configuration")
            
            # Create HTTP client with SSL settings
            http_client_kwargs = {}
            if not ssl_verify:
                # Create custom HTTP client with SSL verification disabled
                sync_http_client = httpx.Client(
                verify=False,
                timeout=global_settings.get("timeout", 30)
                )
                async_http_client = httpx.AsyncClient(
                    verify=False,
                    timeout=global_settings.get("timeout", 30)
                )
                
                http_client_kwargs["http_client"] = sync_http_client
                http_client_kwargs["http_async_client"] = async_http_client
                
                self.logger.warning(f"SSL verification disabled for vLLM provider: {base_url}")

            # Construct the full endpoint URL
            full_base_url = f"{base_url}{api_path}"
            
            # Create ChatOpenAI instance pointing to vLLM server
            vllm_llm = ChatOpenAI(
                model=model,
                temperature=global_settings.get("temperature", 0.1),
                max_tokens=global_settings.get("max_tokens", 1000),
                timeout=global_settings.get("timeout", 30),
                base_url=full_base_url,
                api_key=api_key,
                **http_client_kwargs
            )
            
            self.logger.info(f"Created vLLM provider: {model} at {full_base_url} (SSL verify: {ssl_verify})")
            return vllm_llm
            
        except Exception as e:
            self.logger.error(f"Error creating vLLM provider: {e}")
            return None

    def _prepare_dataset(self, input: Input) -> Dataset:
        """
        Convert input to RAGAS-compatible dataset format
        """
        self.logger.info(f"Preparing dataset for input: {input}")
        
        # Treat context as a simple string
        contexts = [input.retrieval_context]
        
        data = {
            "question": [input.question],
            "contexts": [contexts],
            "answer": [input.response],
            "ground_truths": [[input.reference]],
            "reference": [input.reference]
        }
        self.logger.info(f"Dataset prepared with context")
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
            
            # Prepare metadata with LLM information
            self.logger.info(f"Preparing metadata")
            metadata = {
                "framework": "ragas",
                "total_metrics": len(metric_results),
                "evaluation_successful": True
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
                "framework": "ragas",
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