import os
from unittest.mock import Mock, patch

import pytest

from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.llm_manager import LLMManager
from geneval.schemas import Input, Output


def _mock_llm_manager(provider="openai", model="gpt-4o-mini", provider_config=None, global_settings=None):
    mgr = Mock(spec=LLMManager)
    mgr.get_default_provider.return_value = provider
    mgr.get_provider_config.return_value = provider_config if provider_config is not None else {"model": model}
    mgr.get_global_settings.return_value = global_settings or {"temperature": 0.1, "max_tokens": 1000}
    mgr.get_base_url.return_value = None
    mgr.get_api_path.return_value = None
    return mgr


_ALL_METRIC_PATCHES = [
    "geneval.adapters.ragas_adapter.Faithfulness",
    "geneval.adapters.ragas_adapter.AnswerRelevancy",
    "geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference",
    "geneval.adapters.ragas_adapter.ContextPrecisionWithReference",
    "geneval.adapters.ragas_adapter.ContextRecall",
    "geneval.adapters.ragas_adapter.ContextEntityRecall",
    "geneval.adapters.ragas_adapter.NoiseSensitivity",
]


def _patch_all_metrics():
    """Return a list of patch objects for all metric classes."""
    return [patch(p) for p in _ALL_METRIC_PATCHES]


def _apply_metric_patches(func):
    """Decorator that patches all metric classes."""
    for p in reversed(_ALL_METRIC_PATCHES):
        func = patch(p)(func)
    return func


def _build_adapter(provider="openai", model="gpt-4o-mini", provider_config=None, global_settings=None, env=None):
    """Helper to build a RAGASAdapter with all necessary mocks.

    Returns (adapter, mock_llm_factory, mock_client_class).
    """
    mgr = _mock_llm_manager(provider, model, provider_config, global_settings)
    env = env or {}

    with (
        patch.dict(os.environ, env, clear=True),
        patch("geneval.adapters.ragas_adapter.llm_factory") as mock_factory,
        patch("geneval.adapters.ragas_adapter.AsyncOpenAI") as mock_async_openai,
        patch("geneval.adapters.ragas_adapter.OpenAI") as mock_openai,
        patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI") as mock_async_azure,
        patch("geneval.adapters.ragas_adapter.Anthropic") as mock_anthropic,
        patch("geneval.adapters.ragas_adapter.RagasOpenAIEmbeddings"),
        patch("geneval.adapters.ragas_adapter.Faithfulness"),
        patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
        patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
        patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
        patch("geneval.adapters.ragas_adapter.ContextRecall"),
        patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
        patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
    ):
        mock_factory.return_value = Mock()
        adapter = RAGASAdapter(mgr)
        return (
            adapter,
            mock_factory,
            {
                "async_openai": mock_async_openai,
                "openai": mock_openai,
                "async_azure": mock_async_azure,
                "anthropic": mock_anthropic,
            },
        )


class TestRAGASAdapterInit:
    def test_initialization_success(self):
        adapter, mock_factory, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})

        assert adapter.llm is not None
        assert adapter.llm_info["provider"] == "openai"
        assert adapter.llm_info["model"] == "gpt-4o-mini"
        assert len(adapter.supported_metrics) == 7
        assert "faithfulness" in adapter.supported_metrics

    def test_initialization_missing_llm_manager(self):
        with pytest.raises(ValueError, match="LLMManager is required"):
            RAGASAdapter(None)

    def test_initialization_no_default_provider(self):
        mgr = Mock(spec=LLMManager)
        mgr.get_default_provider.return_value = None
        with pytest.raises(ValueError, match="No default LLM provider configured"):
            RAGASAdapter(mgr)

    def test_initialization_llm_creation_returns_none(self):
        mgr = _mock_llm_manager()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_supported_metrics_list(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        expected = [
            "context_precision_without_reference",
            "context_precision_with_reference",
            "context_recall",
            "context_entity_recall",
            "noise_sensitivity",
            "answer_relevancy",
            "faithfulness",
        ]
        for metric in expected:
            assert metric in adapter.supported_metrics

    def test_initialization_without_embeddings_excludes_answer_relevancy(self):
        """When no embeddings are available, answer_relevancy should not be in available_metrics."""
        mgr = _mock_llm_manager()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory") as mock_factory,
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.RagasOpenAIEmbeddings"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            mock_factory.return_value = Mock()
            with patch.object(RAGASAdapter, "_create_ragas_embeddings", return_value=None):
                adapter = RAGASAdapter(mgr)

            assert "answer_relevancy" not in adapter.supported_metrics
            assert "answer_relevancy" not in adapter.available_metrics
            assert len(adapter.supported_metrics) == 6

    def test_initialization_with_embeddings_includes_answer_relevancy(self):
        """When embeddings are available, answer_relevancy should be in available_metrics."""
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})

        assert "answer_relevancy" in adapter.supported_metrics
        assert "answer_relevancy" in adapter.available_metrics
        assert len(adapter.supported_metrics) == 7

    def test_llm_info_structure(self):
        adapter, _, _ = _build_adapter(
            provider_config={"model": "gpt-4o-mini", "enabled": True},
            env={"OPENAI_API_KEY": "test-key"},
        )
        assert adapter.llm_info["provider"] == "openai"
        assert adapter.llm_info["model"] == "gpt-4o-mini"
        assert "provider_config" in adapter.llm_info
        assert "global_settings" in adapter.llm_info


class TestOpenAIProvider:
    def test_success(self):
        adapter, mock_factory, clients = _build_adapter(env={"OPENAI_API_KEY": "test-key"})

        clients["async_openai"].assert_called_once_with(api_key="test-key")
        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args
        assert call_kwargs[0][0] == "gpt-4o-mini"
        assert call_kwargs[1]["provider"] == "openai"

    def test_missing_api_key(self):
        mgr = _mock_llm_manager()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_model(self):
        mgr = _mock_llm_manager(provider_config={})
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_custom_env_var(self):
        adapter, _, clients = _build_adapter(
            provider_config={"model": "gpt-4o-mini", "api_key_env": "CUSTOM_OPENAI_KEY"},
            env={"CUSTOM_OPENAI_KEY": "custom-key"},
        )
        clients["async_openai"].assert_called_once_with(api_key="custom-key")

    def test_llm_creation_exception(self):
        mgr = _mock_llm_manager()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI") as mock_async_openai,
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            mock_async_openai.side_effect = Exception("Connection failed")
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)


class TestAnthropicProvider:
    def test_success(self):
        adapter, mock_factory, clients = _build_adapter(
            provider="anthropic",
            model="claude-3-5-haiku",
            provider_config={"model": "claude-3-5-haiku"},
            env={"ANTHROPIC_API_KEY": "test-key"},
        )
        clients["anthropic"].assert_called_once_with(api_key="test-key")
        assert mock_factory.call_args[1]["provider"] == "anthropic"

    def test_missing_api_key(self):
        mgr = _mock_llm_manager(provider="anthropic", provider_config={"model": "claude-3-5-haiku"})
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)


class TestGeminiProvider:
    def test_success(self):
        adapter, mock_factory, clients = _build_adapter(
            provider="gemini",
            model="gemini-1.5-flash",
            provider_config={"model": "gemini-1.5-flash"},
            env={"GOOGLE_API_KEY": "test-key"},
        )
        clients["async_openai"].assert_called_once_with(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="test-key",
        )
        assert mock_factory.call_args[1]["provider"] == "openai"


class TestOllamaProvider:
    def test_success(self):
        adapter, mock_factory, clients = _build_adapter(
            provider="ollama",
            model="llama3.2",
            provider_config={"model": "llama3.2", "base_url": "http://localhost:11434"},
            env={},
        )
        clients["async_openai"].assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def test_default_url(self):
        adapter, _, clients = _build_adapter(
            provider="ollama",
            model="llama3.2",
            provider_config={"model": "llama3.2"},
            env={},
        )
        clients["async_openai"].assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )


class TestVLLMProvider:
    def test_success(self):
        mgr = _mock_llm_manager(
            provider="vllm",
            provider_config={"model": "gemini-2.0-flash", "api_key_env": "OPENAI_API_KEY", "ssl_verify": True},
        )
        mgr.get_base_url.return_value = "https://vllm-server.com"
        mgr.get_api_path.return_value = "/v1"

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory") as mock_factory,
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI") as mock_async_openai,
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.RagasOpenAIEmbeddings"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            from unittest.mock import call

            mock_factory.return_value = Mock()
            RAGASAdapter(mgr)
            assert call(base_url="https://vllm-server.com/v1", api_key="test-key") in mock_async_openai.call_args_list

    def test_ssl_disabled(self):
        mgr = _mock_llm_manager(
            provider="vllm",
            provider_config={"model": "gemini-2.0-flash", "ssl_verify": False},
        )
        mgr.get_base_url.return_value = "https://vllm-server.com"
        mgr.get_api_path.return_value = "/v1"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory") as mock_factory,
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.RagasOpenAIEmbeddings"),
            patch("geneval.adapters.ragas_adapter.httpx.AsyncClient") as mock_httpx,
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            mock_factory.return_value = Mock()
            RAGASAdapter(mgr)
            mock_httpx.assert_called_once_with(verify=False, timeout=30)

    def test_missing_model(self):
        mgr = _mock_llm_manager(provider="vllm", provider_config={})
        mgr.get_base_url.return_value = "https://vllm-server.com"
        mgr.get_api_path.return_value = "/v1"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)


class TestAzureOpenAIProvider:
    def _azure_config(self):
        return {
            "model": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
            "openai_api_version": "2025-01-01-preview",
            "azure_endpoint": "https://test.openai.azure.com/",
        }

    def test_success(self):
        config = self._azure_config()
        adapter, mock_factory, clients = _build_adapter(
            provider="azure_openai",
            model="gpt-4",
            provider_config=config,
            env={},
        )
        clients["async_azure"].assert_called_once_with(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            api_version="2025-01-01-preview",
        )

    def test_missing_api_key(self):
        config = self._azure_config()
        del config["azure_openai_api_key"]
        mgr = _mock_llm_manager(provider="azure_openai", provider_config=config)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_model(self):
        config = self._azure_config()
        del config["model"]
        mgr = _mock_llm_manager(provider="azure_openai", provider_config=config)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_deployment_name(self):
        config = self._azure_config()
        del config["deployment_name"]
        mgr = _mock_llm_manager(provider="azure_openai", provider_config=config)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_endpoint(self):
        config = self._azure_config()
        del config["azure_endpoint"]
        mgr = _mock_llm_manager(provider="azure_openai", provider_config=config)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)


class TestDeepSeekProvider:
    def test_success(self):
        adapter, mock_factory, clients = _build_adapter(
            provider="deepseek",
            model="deepseek-chat",
            provider_config={"model": "deepseek-chat"},
            env={"DEEPSEEK_API_KEY": "test-key"},
        )
        clients["async_openai"].assert_called_once_with(
            base_url="https://api.deepseek.com/v1",
            api_key="test-key",
        )
        assert adapter.llm is not None

    def test_missing_api_key(self):
        mgr = _mock_llm_manager(provider="deepseek", provider_config={"model": "deepseek-chat"})
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_model(self):
        mgr = _mock_llm_manager(provider="deepseek", provider_config={})
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_custom_env_var(self):
        adapter, _, clients = _build_adapter(
            provider="deepseek",
            model="deepseek-chat",
            provider_config={"model": "deepseek-chat", "api_key_env": "CUSTOM_DEEPSEEK_KEY"},
            env={"CUSTOM_DEEPSEEK_KEY": "custom-key"},
        )
        clients["async_openai"].assert_called_once_with(
            base_url="https://api.deepseek.com/v1",
            api_key="custom-key",
        )


class TestAmazonBedrockProvider:
    def test_returns_none(self):
        mgr = _mock_llm_manager(
            provider="amazon_bedrock",
            provider_config={"model": "anthropic.claude-3-sonnet-20240229-v1:0", "region_name": "us-west-2"},
        )
        with (
            patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_credentials(self):
        mgr = _mock_llm_manager(
            provider="amazon_bedrock",
            provider_config={"model": "anthropic.claude-3-sonnet-20240229-v1:0"},
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)

    def test_missing_model(self):
        mgr = _mock_llm_manager(provider="amazon_bedrock", provider_config={})
        with (
            patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)


class TestUnknownProvider:
    def test_unknown_provider(self):
        mgr = _mock_llm_manager(provider="unknown_provider", provider_config={"model": "x"})
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("geneval.adapters.ragas_adapter.llm_factory"),
            patch("geneval.adapters.ragas_adapter.AsyncOpenAI"),
            patch("geneval.adapters.ragas_adapter.OpenAI"),
            patch("geneval.adapters.ragas_adapter.AsyncAzureOpenAI"),
            patch("geneval.adapters.ragas_adapter.Anthropic"),
            patch("geneval.adapters.ragas_adapter.Faithfulness"),
            patch("geneval.adapters.ragas_adapter.AnswerRelevancy"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithoutReference"),
            patch("geneval.adapters.ragas_adapter.ContextPrecisionWithReference"),
            patch("geneval.adapters.ragas_adapter.ContextRecall"),
            patch("geneval.adapters.ragas_adapter.ContextEntityRecall"),
            patch("geneval.adapters.ragas_adapter.NoiseSensitivity"),
        ):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mgr)


class TestGetMetrics:
    def test_supported(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        metrics = adapter._get_metrics(["faithfulness", "answer_relevancy"])
        assert len(metrics) == 2

    def test_unsupported(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        with pytest.raises(ValueError, match="Unsupported metric: unsupported_metric"):
            adapter._get_metrics(["unsupported_metric"])


class TestEvaluate:
    def _make_input(self, metrics=None):
        return Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=metrics if metrics is not None else ["faithfulness", "answer_relevancy"],
        )

    def _mock_score_result(self, value=0.85, reason="good"):
        result = Mock()
        result.value = value
        result.reason = reason
        return result

    def test_success(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})

        adapter.available_metrics["faithfulness"].score = Mock(return_value=self._mock_score_result(0.85, "High faithfulness"))
        adapter.available_metrics["answer_relevancy"].score = Mock(return_value=self._mock_score_result(0.92, "Relevant answer"))

        output = adapter.evaluate(self._make_input())

        assert isinstance(output, Output)
        assert len(output.metrics) == 2
        assert output.metadata["framework"] == "ragas"
        assert output.metadata["evaluation_successful"] is True
        assert output.metadata["llm_provider"] == "openai"
        assert output.metadata["llm_model"] == "gpt-4o-mini"

        faithfulness_metric = next(m for m in output.metrics if m.name == "faithfulness")
        answer_metric = next(m for m in output.metrics if m.name == "answer_relevancy")

        assert faithfulness_metric.score == 0.85
        assert faithfulness_metric.tool_name == "ragas"
        assert faithfulness_metric.details == "High faithfulness"

        assert answer_metric.score == 0.92
        assert answer_metric.details == "Relevant answer"

    def test_score_kwargs_faithfulness(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["faithfulness"].score = Mock(return_value=self._mock_score_result())

        adapter.evaluate(self._make_input(metrics=["faithfulness"]))

        call_kwargs = adapter.available_metrics["faithfulness"].score.call_args[1]
        assert call_kwargs["user_input"] == "What is the capital of France?"
        assert call_kwargs["response"] == "The capital of France is Paris."
        assert call_kwargs["retrieved_contexts"] == ["Paris is the capital and largest city of France."]

    def test_score_kwargs_answer_relevancy(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["answer_relevancy"].score = Mock(return_value=self._mock_score_result())

        adapter.evaluate(self._make_input(metrics=["answer_relevancy"]))

        call_kwargs = adapter.available_metrics["answer_relevancy"].score.call_args[1]
        assert call_kwargs["user_input"] == "What is the capital of France?"
        assert call_kwargs["response"] == "The capital of France is Paris."
        assert "retrieved_contexts" not in call_kwargs

    def test_score_kwargs_context_recall(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["context_recall"].score = Mock(return_value=self._mock_score_result())

        adapter.evaluate(self._make_input(metrics=["context_recall"]))

        call_kwargs = adapter.available_metrics["context_recall"].score.call_args[1]
        assert call_kwargs["user_input"] == "What is the capital of France?"
        assert call_kwargs["retrieved_contexts"] == ["Paris is the capital and largest city of France."]
        assert call_kwargs["reference"] == "Paris is the capital of France."

    def test_score_kwargs_context_precision_with_reference(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["context_precision_with_reference"].score = Mock(return_value=self._mock_score_result())

        adapter.evaluate(self._make_input(metrics=["context_precision_with_reference"]))

        call_kwargs = adapter.available_metrics["context_precision_with_reference"].score.call_args[1]
        assert call_kwargs["user_input"] == "What is the capital of France?"
        assert call_kwargs["reference"] == "Paris is the capital of France."
        assert call_kwargs["retrieved_contexts"] == ["Paris is the capital and largest city of France."]

    def test_score_kwargs_noise_sensitivity(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["noise_sensitivity"].score = Mock(return_value=self._mock_score_result())

        adapter.evaluate(self._make_input(metrics=["noise_sensitivity"]))

        call_kwargs = adapter.available_metrics["noise_sensitivity"].score.call_args[1]
        assert "user_input" in call_kwargs
        assert "response" in call_kwargs
        assert "reference" in call_kwargs
        assert "retrieved_contexts" in call_kwargs

    def test_failure(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["faithfulness"].score = Mock(side_effect=Exception("RAGAS evaluation failed"))

        output = adapter.evaluate(self._make_input(metrics=["faithfulness"]))

        assert isinstance(output, Output)
        assert len(output.metrics) == 0
        assert output.metadata["evaluation_successful"] is False
        assert "RAGAS evaluation failed" in output.metadata["error"]

    def test_empty_metrics(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        output = adapter.evaluate(self._make_input(metrics=[]))

        assert isinstance(output, Output)
        assert len(output.metrics) == 0
        assert output.metadata["evaluation_successful"] is True
        assert output.metadata["total_metrics"] == 0

    def test_no_reason_uses_default(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.available_metrics["faithfulness"].score = Mock(return_value=self._mock_score_result(0.9, None))

        output = adapter.evaluate(self._make_input(metrics=["faithfulness"]))

        metric = output.metrics[0]
        assert "RAGAS faithfulness evaluation" in metric.details

    def test_llm_info_none(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.llm_info = None
        adapter.available_metrics["faithfulness"].score = Mock(return_value=self._mock_score_result())

        output = adapter.evaluate(self._make_input(metrics=["faithfulness"]))

        assert isinstance(output, Output)
        assert len(output.metrics) == 1
        assert "llm_provider" not in output.metadata
        assert "llm_model" not in output.metadata

    def test_error_with_llm_info_none(self):
        adapter, _, _ = _build_adapter(env={"OPENAI_API_KEY": "test-key"})
        adapter.llm_info = None

        with patch.object(adapter, "_get_metrics", side_effect=Exception("Test error")):
            output = adapter.evaluate(self._make_input(metrics=["faithfulness"]))

        assert output.metadata["evaluation_successful"] is False
        assert "llm_provider" not in output.metadata
