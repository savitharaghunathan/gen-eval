import logging
import os

import httpx
from anthropic import Anthropic
from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
)

from geneval.llm_manager import LLMManager
from geneval.schemas import Input, MetricResult, Output

_METRIC_KWARGS_BUILDERS: dict[str, callable] = {
    "faithfulness": lambda inp: {
        "user_input": inp.question,
        "response": inp.response,
        "retrieved_contexts": [inp.retrieval_context],
    },
    "answer_relevancy": lambda inp: {
        "user_input": inp.question,
        "response": inp.response,
    },
    "context_precision_without_reference": lambda inp: {
        "user_input": inp.question,
        "response": inp.response,
        "retrieved_contexts": [inp.retrieval_context],
    },
    "context_precision_with_reference": lambda inp: {
        "user_input": inp.question,
        "reference": inp.reference,
        "retrieved_contexts": [inp.retrieval_context],
    },
    "context_recall": lambda inp: {
        "user_input": inp.question,
        "retrieved_contexts": [inp.retrieval_context],
        "reference": inp.reference,
    },
    "context_entity_recall": lambda inp: {
        "reference": inp.reference,
        "retrieved_contexts": [inp.retrieval_context],
    },
    "noise_sensitivity": lambda inp: {
        "user_input": inp.question,
        "response": inp.response,
        "reference": inp.reference,
        "retrieved_contexts": [inp.retrieval_context],
    },
}


class RAGASAdapter:
    def __init__(self, llm_manager: LLMManager):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RAGASAdapter")

        if not llm_manager:
            raise ValueError("LLMManager is required for RAGASAdapter initialization")

        self.llm_manager = llm_manager

        default_provider = self.llm_manager.get_default_provider()
        if not default_provider:
            raise ValueError("No default LLM provider configured. Please set 'default: true' for one provider in the config.")

        self.llm = self._create_ragas_llm(default_provider)

        if not self.llm:
            raise ValueError("No LLM available. Please configure an LLM provider.")

        self.llm_info = {
            "provider": default_provider,
            "model": self.llm_manager.get_provider_config(default_provider).get("model", "unknown"),
            "provider_config": self.llm_manager.get_provider_config(default_provider),
            "global_settings": self.llm_manager.get_global_settings(),
        }

        self.logger.info(f"LLM configured with provider: {self.llm_info.get('provider', 'unknown')}")

        try:
            ragas_embeddings = self._create_ragas_embeddings()

            self.available_metrics = {
                "context_precision_without_reference": ContextPrecisionWithoutReference(llm=self.llm),
                "context_precision_with_reference": ContextPrecisionWithReference(llm=self.llm),
                "context_recall": ContextRecall(llm=self.llm),
                "context_entity_recall": ContextEntityRecall(llm=self.llm),
                "noise_sensitivity": NoiseSensitivity(llm=self.llm),
                "faithfulness": Faithfulness(llm=self.llm),
            }

            if ragas_embeddings:
                self.available_metrics["answer_relevancy"] = AnswerRelevancy(llm=self.llm, embeddings=ragas_embeddings)
            else:
                self.logger.warning("answer_relevancy metric unavailable: no embeddings configured")
            self.logger.info(f"RAGAS metrics initialized successfully with {len(self.available_metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGAS metrics: {e}")
            raise RuntimeError(f"Failed to initialize RAGAS metrics: {e}") from e

        self.supported_metrics = list(self.available_metrics.keys())

    def _create_ragas_llm(self, provider_name: str):
        try:
            result = self._create_native_client(provider_name)
            if result is None:
                return None
            client, llm_provider = result
            model = self.llm_manager.get_provider_config(provider_name).get("model")
            global_settings = self.llm_manager.get_global_settings()
            return llm_factory(
                model,
                provider=llm_provider,
                client=client,
                temperature=global_settings.get("temperature", 0.1),
                max_tokens=global_settings.get("max_tokens", 1000),
            )
        except Exception as e:
            self.logger.error(f"Error creating {provider_name} provider: {e}")
            return None

    def _create_native_client(self, provider_name: str) -> tuple | None:
        if provider_name == "openai":
            return self._create_openai_client()
        elif provider_name == "azure_openai":
            return self._create_azure_openai_client()
        elif provider_name == "anthropic":
            return self._create_anthropic_client()
        elif provider_name == "gemini":
            return self._create_gemini_client()
        elif provider_name == "ollama":
            return self._create_ollama_client()
        elif provider_name == "vllm":
            return self._create_vllm_client()
        elif provider_name == "deepseek":
            return self._create_deepseek_client()
        else:
            self.logger.warning(f"Unsupported RAGAS provider: {provider_name}")
            return None

    def _create_openai_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("openai")
        api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        model = provider_config.get("model")
        if not model:
            self.logger.error("OpenAI model not specified in configuration")
            return None
        return AsyncOpenAI(api_key=api_key), "openai"

    def _create_azure_openai_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("azure_openai")
        api_key = provider_config.get("azure_openai_api_key")
        if not api_key:
            self.logger.warning("Azure OpenAI API key not found in configuration")
            return None
        model = provider_config.get("model")
        if not model:
            self.logger.error("Azure OpenAI model not specified in configuration")
            return None
        deployment_name = provider_config.get("deployment_name")
        if not deployment_name:
            self.logger.error("Azure OpenAI deployment name not specified in configuration")
            return None
        azure_endpoint = provider_config.get("azure_endpoint")
        if not azure_endpoint:
            self.logger.error("Azure OpenAI endpoint not specified in configuration")
            return None
        openai_api_version = provider_config.get("openai_api_version", "2025-01-01-preview")
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
            api_version=openai_api_version,
        )
        return client, "openai"

    def _create_anthropic_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("anthropic")
        api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        model = provider_config.get("model")
        if not model:
            self.logger.error("Anthropic model not specified in configuration")
            return None
        return Anthropic(api_key=api_key), "anthropic"

    def _create_gemini_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("gemini")
        api_key_env = provider_config.get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        model = provider_config.get("model")
        if not model:
            self.logger.error("Gemini model not specified in configuration")
            return None
        client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )
        return client, "openai"

    def _create_ollama_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("ollama")
        base_url = provider_config.get("base_url", "http://localhost:11434")
        model = provider_config.get("model")
        if not model:
            self.logger.error("Ollama model not specified in configuration")
            return None
        client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        return client, "openai"

    def _create_vllm_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("vllm")
        base_url = self.llm_manager.get_base_url("vllm") or "http://localhost:8000"
        model = provider_config.get("model")
        ssl_verify = provider_config.get("ssl_verify", True)
        api_path = self.llm_manager.get_api_path("vllm") or "/v1"

        api_key_env = provider_config.get("api_key_env")
        api_key = "dummy-key"
        if api_key_env:
            env_api_key = os.getenv(api_key_env)
            if env_api_key:
                api_key = env_api_key
            else:
                self.logger.warning(f"{api_key_env} not found in environment variables, using dummy key")
        elif provider_config.get("api_key"):
            api_key = provider_config.get("api_key")

        if not model:
            raise ValueError("vLLM model not specified in configuration")

        full_base_url = f"{base_url}{api_path}"
        client_kwargs = {"base_url": full_base_url, "api_key": api_key}

        if not ssl_verify:
            client_kwargs["http_client"] = httpx.AsyncClient(verify=False, timeout=30)
            self.logger.warning(f"SSL verification disabled for vLLM provider: {base_url}")

        client = AsyncOpenAI(**client_kwargs)
        self.logger.info(f"Created vLLM provider: {model} at {full_base_url} (SSL verify: {ssl_verify})")
        return client, "openai"

    def _create_deepseek_client(self) -> tuple | None:
        provider_config = self.llm_manager.get_provider_config("deepseek")
        api_key_env = provider_config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self.logger.warning(f"{api_key_env} not found in environment variables")
            return None
        model = provider_config.get("model")
        if not model:
            self.logger.error("DeepSeek model not specified in configuration")
            return None
        client = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)
        return client, "openai"

    def _create_ragas_embeddings(self):
        try:
            embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
            embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

            if not embedding_api_key:
                openai_config = self.llm_manager.get_provider_config("openai")
                if openai_config:
                    api_key_env = openai_config.get("api_key_env", "OPENAI_API_KEY")
                    embedding_api_key = os.getenv(api_key_env)
                    if not embedding_base_url:
                        embedding_base_url = openai_config.get("base_url")

            if not embedding_api_key:
                self.logger.warning("No embedding API key found. Answer relevancy may not work optimally.")
                return None

            client_kwargs = {"api_key": embedding_api_key}
            if embedding_base_url:
                client_kwargs["base_url"] = embedding_base_url
                self.logger.info(f"Using custom embeddings base URL: {embedding_base_url}")

            openai_ignore_ssl = os.getenv("OPENAI_IGNORE_SSL", "false").lower() == "true"
            if openai_ignore_ssl:
                client_kwargs["http_client"] = httpx.Client(verify=False, timeout=30)
                self.logger.info("Created RAGAS embeddings with SSL verification disabled")
            else:
                self.logger.info("Created RAGAS embeddings with standard SSL")

            client = OpenAI(**client_kwargs)
            embeddings = RagasOpenAIEmbeddings(client=client, model=embedding_model)

            self.logger.info(f"Embeddings configured successfully with model: {embedding_model}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to create RAGAS embeddings: {e}")
            self.logger.warning("Answer relevancy metric will work with LLM only (may be less optimal)")
            return None

    def _get_metrics(self, metric_names: list[str]) -> list:
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
        try:
            self._get_metrics(input.metrics)

            metric_results = []
            for metric_name in input.metrics:
                metric_obj = self.available_metrics[metric_name]
                kwargs_builder = _METRIC_KWARGS_BUILDERS.get(metric_name)
                if not kwargs_builder:
                    self.logger.error(f"No kwargs mapping for metric: {metric_name}")
                    continue

                kwargs = kwargs_builder(input)
                result = metric_obj.score(**kwargs)

                score = float(result.value)
                details = result.reason if result.reason else f"RAGAS {metric_name} evaluation"

                metric_results.append(
                    MetricResult(
                        name=metric_name,
                        score=score,
                        tool_name="ragas",
                        details=details,
                    )
                )

            metadata = {
                "framework": "ragas",
                "total_metrics": len(metric_results),
                "evaluation_successful": True,
            }

            if self.llm_info:
                metadata.update(
                    {
                        "llm_provider": self.llm_info.get("provider"),
                        "llm_model": self.llm_info.get("model"),
                    }
                )

            return Output(metrics=metric_results, metadata=metadata)

        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            metadata = {
                "framework": "ragas",
                "error": str(e),
                "evaluation_successful": False,
            }

            if self.llm_info:
                metadata.update(
                    {
                        "llm_provider": self.llm_info.get("provider"),
                        "llm_model": self.llm_info.get("model"),
                    }
                )

            return Output(metrics=[], metadata=metadata)
