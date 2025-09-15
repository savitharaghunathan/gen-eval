import os
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.llm_manager import LLMManager
from geneval.schemas import Input, Output


class TestRAGASAdapter:
    """Test cases for RAGASAdapter"""

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_initialization_success(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful RAGASAdapter initialization"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        # Mock ChatOpenAI
        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        # Mock LangChain wrapper
        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            assert adapter.llm_manager == mock_llm_manager
            assert adapter.llm == mock_wrapped_llm
            assert "openai" in adapter.llm_info["provider"]
            assert adapter.llm_info["model"] == "gpt-4o-mini"
            assert len(adapter.supported_metrics) > 0
            assert "faithfulness" in adapter.supported_metrics

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    def test_initialization_missing_llm_manager(self, mock_llm_manager_class):
        """Test RAGASAdapter initialization with missing LLM manager"""
        with pytest.raises(ValueError, match="LLMManager is required"):
            RAGASAdapter(None)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    def test_initialization_no_default_provider(self, mock_llm_manager_class):
        """Test RAGASAdapter initialization with no default provider"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = None

        with pytest.raises(ValueError, match="No default LLM provider configured"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_initialization_llm_creation_failure(self, mock_chat_openai, mock_llm_manager_class):
        """Test RAGASAdapter initialization when LLM creation fails"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        # Mock ChatOpenAI to return None
        mock_chat_openai.return_value = None

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_openai_provider_success(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful OpenAI provider creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1, "max_tokens": 1000, "timeout": 30}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            RAGASAdapter(mock_llm_manager)

            # Verify ChatOpenAI was called with correct parameters
            mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0.1, max_tokens=1000, timeout=30)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_openai_provider_missing_api_key(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test OpenAI provider creation with missing API key"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_openai_provider_missing_model(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test OpenAI provider creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {}  # No model
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_openai_provider_custom_env_var(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test OpenAI provider creation with custom environment variable"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini", "api_key_env": "CUSTOM_OPENAI_KEY"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"CUSTOM_OPENAI_KEY": "custom-key"}):
            adapter = RAGASAdapter(mock_llm_manager)
            assert adapter.llm == mock_wrapped_llm

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatAnthropic")
    def test_create_anthropic_provider_success(self, mock_chat_anthropic, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful Anthropic provider creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "anthropic"
        mock_llm_manager.get_provider_config.return_value = {"model": "claude-3-5-haiku"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1, "max_tokens": 1000}

        mock_anthropic_llm = Mock()
        mock_chat_anthropic.return_value = mock_anthropic_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            RAGASAdapter(mock_llm_manager)

            mock_chat_anthropic.assert_called_once_with(model="claude-3-5-haiku", temperature=0.1, max_tokens=1000)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatGoogleGenerativeAI")
    def test_create_gemini_provider_success(self, mock_chat_gemini, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful Gemini provider creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "gemini"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-1.5-flash"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1, "max_tokens": 1000}

        mock_gemini_llm = Mock()
        mock_chat_gemini.return_value = mock_gemini_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            RAGASAdapter(mock_llm_manager)

            mock_chat_gemini.assert_called_once_with(model="gemini-1.5-flash", temperature=0.1, max_output_tokens=1000)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.Ollama")
    def test_create_ollama_provider_success(self, mock_ollama, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful Ollama provider creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "ollama"
        mock_llm_manager.get_provider_config.return_value = {"model": "llama3.2", "base_url": "http://localhost:11434"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_ollama_llm = Mock()
        mock_ollama.return_value = mock_ollama_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        RAGASAdapter(mock_llm_manager)

        mock_ollama.assert_called_once_with(model="llama3.2", base_url="http://localhost:11434", temperature=0.1)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.Ollama")
    def test_create_ollama_provider_default_url(self, mock_ollama, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Ollama provider creation with default URL"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "ollama"
        mock_llm_manager.get_provider_config.return_value = {"model": "llama3.2"}  # No base_url
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_ollama_llm = Mock()
        mock_ollama.return_value = mock_ollama_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        RAGASAdapter(mock_llm_manager)

        mock_ollama.assert_called_once_with(model="llama3.2", base_url="http://localhost:11434", temperature=0.1)  # Default URL

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_vllm_provider_success(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful vLLM provider creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "vllm"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-2.0-flash", "api_key_env": "OPENAI_API_KEY", "ssl_verify": True}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1, "max_tokens": 1000, "timeout": 30}
        mock_llm_manager.get_base_url.return_value = "https://vllm-server.com"
        mock_llm_manager.get_api_path.return_value = "/v1"

        mock_vllm_llm = Mock()
        mock_chat_openai.return_value = mock_vllm_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            RAGASAdapter(mock_llm_manager)

            mock_chat_openai.assert_called_once_with(
                model="gemini-2.0-flash", temperature=0.1, max_tokens=1000, timeout=30, base_url="https://vllm-server.com/v1", api_key="test-key"
            )

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_vllm_provider_ssl_disabled(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test vLLM provider creation with SSL verification disabled"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "vllm"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-2.0-flash", "ssl_verify": False}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1, "timeout": 30}
        mock_llm_manager.get_base_url.return_value = "https://vllm-server.com"
        mock_llm_manager.get_api_path.return_value = "/custom"

        mock_vllm_llm = Mock()
        mock_chat_openai.return_value = mock_vllm_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock httpx clients
        with (
            patch("geneval.adapters.ragas_adapter.httpx.Client") as mock_sync_client,
            patch("geneval.adapters.ragas_adapter.httpx.AsyncClient") as mock_async_client,
        ):
            RAGASAdapter(mock_llm_manager)

            # Verify httpx clients were created with SSL verification disabled
            mock_sync_client.assert_called_once()
            sync_call_kwargs = mock_sync_client.call_args[1]
            assert sync_call_kwargs["verify"] is False

            mock_async_client.assert_called_once()
            async_call_kwargs = mock_async_client.call_args[1]
            assert async_call_kwargs["verify"] is False

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_vllm_provider_missing_model(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test vLLM provider creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "vllm"
        mock_llm_manager.get_provider_config.return_value = {}  # No model
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_base_url.return_value = "https://vllm-server.com"
        mock_llm_manager.get_api_path.return_value = "/v1"

        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_prepare_dataset(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test dataset preparation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness", "answer_relevancy"],
            )

            dataset = adapter._prepare_dataset(input_data)

            assert isinstance(dataset, Dataset)
            assert dataset["question"] == ["What is the capital of France?"]
            assert dataset["contexts"] == [["Paris is the capital and largest city of France."]]
            assert dataset["answer"] == ["The capital of France is Paris."]
            assert dataset["ground_truths"] == [["Paris is the capital of France."]]
            assert dataset["reference"] == ["Paris is the capital of France."]

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_get_metrics_supported(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test getting supported metrics"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            metrics = adapter._get_metrics(["faithfulness", "answer_relevancy"])
            assert len(metrics) == 2

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_get_metrics_unsupported(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test getting unsupported metrics"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            with pytest.raises(ValueError, match="Unsupported metric: unsupported_metric"):
                adapter._get_metrics(["unsupported_metric"])

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    @patch("geneval.adapters.ragas_adapter.evaluate")
    def test_evaluate_success(self, mock_evaluate, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful evaluation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock RAGAS evaluation results
        mock_results = Mock()
        mock_results.scores = [{"faithfulness": 0.85, "answer_relevancy": 0.92}]
        mock_evaluate.return_value = mock_results

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness", "answer_relevancy"],
            )

            output = adapter.evaluate(input_data)

            assert isinstance(output, Output)
            assert len(output.metrics) == 2
            assert output.metadata["framework"] == "ragas"
            assert output.metadata["evaluation_successful"] is True
            assert output.metadata["llm_provider"] == "openai"
            assert output.metadata["llm_model"] == "gpt-4o-mini"

            # Check metric results
            faithfulness_metric = next(m for m in output.metrics if m.name == "faithfulness")
            answer_relevancy_metric = next(m for m in output.metrics if m.name == "answer_relevancy")

            assert faithfulness_metric.score == 0.85
            assert faithfulness_metric.tool_name == "ragas"
            assert "faithfulness" in faithfulness_metric.details

            assert answer_relevancy_metric.score == 0.92
            assert answer_relevancy_metric.tool_name == "ragas"
            assert "answer_relevancy" in answer_relevancy_metric.details

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    @patch("geneval.adapters.ragas_adapter.evaluate")
    def test_evaluate_failure(self, mock_evaluate, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test evaluation failure handling"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock RAGAS evaluation to raise an exception
        mock_evaluate.side_effect = Exception("RAGAS evaluation failed")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"],
            )

            output = adapter.evaluate(input_data)

            assert isinstance(output, Output)
            assert len(output.metrics) == 0
            assert output.metadata["framework"] == "ragas"
            assert output.metadata["evaluation_successful"] is False
            assert "RAGAS evaluation failed" in output.metadata["error"]

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    @patch("geneval.adapters.ragas_adapter.evaluate")
    def test_evaluate_unexpected_results_format(self, mock_evaluate, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test evaluation with unexpected results format"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock RAGAS evaluation with unexpected format
        mock_results = Mock()
        mock_results.scores = []  # Empty scores
        mock_evaluate.return_value = mock_results

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"],
            )

            output = adapter.evaluate(input_data)

            assert isinstance(output, Output)
            assert len(output.metrics) == 0
            # The adapter logs an error but still sets evaluation_successful to True
            # because it's not in the exception handling block
            assert output.metadata["evaluation_successful"] is True

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_supported_metrics_list(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test that all expected metrics are supported"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            expected_metrics = [
                "context_precision_without_reference",
                "context_precision_with_reference",
                "context_recall",
                "context_entity_recall",
                "noise_sensitivity",
                "answer_relevancy",
                "faithfulness",
            ]

            for metric in expected_metrics:
                assert metric in adapter.supported_metrics, f"Metric {metric} should be supported"

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_llm_info_structure(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test LLM info structure and content"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini", "enabled": True}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            assert "provider" in adapter.llm_info
            assert "model" in adapter.llm_info
            assert "provider_config" in adapter.llm_info
            assert "global_settings" in adapter.llm_info

            assert adapter.llm_info["provider"] == "openai"
            assert adapter.llm_info["model"] == "gpt-4o-mini"
            assert adapter.llm_info["provider_config"]["enabled"] is True
            assert adapter.llm_info["global_settings"]["temperature"] == 0.1

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_azure_openai_provider_success(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test successful Azure OpenAI provider creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {
            "model": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
            "openai_api_version": "2025-01-01-preview",
            "azure_endpoint": "https://test.openai.azure.com/",
        }
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1, "max_tokens": 1000, "timeout": 30}

        mock_azure_llm = Mock()
        mock_chat_openai.return_value = mock_azure_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        RAGASAdapter(mock_llm_manager)

        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.1,
            max_tokens=1000,
            timeout=30,
            openai_api_version="2025-01-01-preview",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4-deployment",
            openai_api_key="test-key",
        )

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_azure_openai_provider_missing_api_key(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Azure OpenAI provider creation with missing API key"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {
            "model": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "azure_endpoint": "https://test.openai.azure.com/",
        }
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_azure_openai_provider_missing_model(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Azure OpenAI provider creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com/",
        }
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_azure_openai_provider_missing_deployment_name(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Azure OpenAI provider creation with missing deployment name"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {
            "model": "gpt-4",
            "azure_openai_api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com/",
        }
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_azure_openai_provider_missing_endpoint(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Azure OpenAI provider creation with missing endpoint"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {
            "model": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
        }
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_amazon_bedrock_provider_success(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Amazon Bedrock provider creation (returns None but logs info)"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "amazon_bedrock"
        mock_llm_manager.get_provider_config.return_value = {"model": "anthropic.claude-3-sonnet-20240229-v1:0", "region_name": "us-west-2"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_amazon_bedrock_provider_missing_aws_credentials(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Amazon Bedrock provider creation with missing AWS credentials"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "amazon_bedrock"
        mock_llm_manager.get_provider_config.return_value = {"model": "anthropic.claude-3-sonnet-20240229-v1:0"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_amazon_bedrock_provider_missing_model(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Amazon Bedrock provider creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "amazon_bedrock"
        mock_llm_manager.get_provider_config.return_value = {}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_amazon_bedrock_provider_default_region(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test Amazon Bedrock provider creation with default region"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "amazon_bedrock"
        mock_llm_manager.get_provider_config.return_value = {
            "model": "anthropic.claude-3-sonnet-20240229-v1:0"
            # No region_name, should use default
        }
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_deepseek_provider_success(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test DeepSeek provider creation (returns None but logs info)"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {"model": "deepseek-chat"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_deepseek_provider_missing_api_key(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test DeepSeek provider creation with missing API key"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {"model": "deepseek-chat"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_deepseek_provider_missing_model(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test DeepSeek provider creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_deepseek_provider_custom_env_var(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test DeepSeek provider creation with custom environment variable"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {"model": "deepseek-chat", "api_key_env": "CUSTOM_DEEPSEEK_KEY"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with patch.dict(os.environ, {"CUSTOM_DEEPSEEK_KEY": "custom-key"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_unknown_provider(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test creation of unknown provider"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "unknown_provider"
        mock_llm_manager.get_provider_config.return_value = {"model": "unknown-model"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_create_langchain_llm_exception_handling(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test exception handling during LLM creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        # Mock ChatOpenAI to raise an exception
        mock_chat_openai.side_effect = Exception("LLM creation failed")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No LLM available"):
                RAGASAdapter(mock_llm_manager)

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_evaluate_metric_not_found_in_results(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test evaluation when metric is not found in RAGAS results"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock RAGAS evaluation with results that don't contain the requested metric
        mock_results = Mock()
        mock_results.scores = [{"different_metric": 0.85}]

        with patch("geneval.adapters.ragas_adapter.evaluate", return_value=mock_results), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"],
            )

            output = adapter.evaluate(input_data)

            assert isinstance(output, Output)
            assert len(output.metrics) == 0  # No metrics found
            assert output.metadata["evaluation_successful"] is True
            assert output.metadata["total_metrics"] == 0

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_evaluate_with_empty_metrics_list(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test evaluation with empty metrics list"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=[],  # Empty metrics list
            )

            # Mock both _get_metrics and evaluate to handle empty metrics
            mock_results = Mock()
            mock_results.scores = []  # Empty scores for empty metrics

            with patch.object(adapter, "_get_metrics", return_value=[]), patch("geneval.adapters.ragas_adapter.evaluate", return_value=mock_results):
                output = adapter.evaluate(input_data)

                assert isinstance(output, Output)
                assert len(output.metrics) == 0
                assert output.metadata["evaluation_successful"] is True
                assert output.metadata["total_metrics"] == 0

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_evaluate_with_llm_info_none(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test evaluation when llm_info is None"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        # Mock RAGAS evaluation results
        mock_results = Mock()
        mock_results.scores = [{"faithfulness": 0.85}]

        with patch("geneval.adapters.ragas_adapter.evaluate", return_value=mock_results), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            # Manually set llm_info to None to test this path
            adapter.llm_info = None

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"],
            )

            output = adapter.evaluate(input_data)

            assert isinstance(output, Output)
            assert len(output.metrics) == 1
            assert output.metadata["evaluation_successful"] is True
            # Should not have LLM-specific metadata
            assert "llm_provider" not in output.metadata
            assert "llm_model" not in output.metadata

    @patch("geneval.adapters.ragas_adapter.LLMManager")
    @patch("geneval.adapters.ragas_adapter.LangchainLLMWrapper")
    @patch("geneval.adapters.ragas_adapter.ChatOpenAI")
    def test_evaluate_error_with_llm_info_none(self, mock_chat_openai, mock_langchain_wrapper, mock_llm_manager_class):
        """Test error handling during evaluation when llm_info is None"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}

        mock_openai_llm = Mock()
        mock_chat_openai.return_value = mock_openai_llm

        mock_wrapped_llm = Mock()
        mock_langchain_wrapper.return_value = mock_wrapped_llm

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = RAGASAdapter(mock_llm_manager)

            # Manually set llm_info to None to test this path
            adapter.llm_info = None

            input_data = Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"],
            )

            # Mock _get_metrics to raise an exception
            with patch.object(adapter, "_get_metrics", side_effect=Exception("Test error")):
                output = adapter.evaluate(input_data)

                assert isinstance(output, Output)
                assert len(output.metrics) == 0
                assert output.metadata["evaluation_successful"] is False
                assert "Test error" in output.metadata["error"]
                # Should not have LLM-specific metadata
                assert "llm_provider" not in output.metadata
                assert "llm_model" not in output.metadata
