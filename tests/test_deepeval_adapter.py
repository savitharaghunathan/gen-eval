import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.llm_manager import LLMManager
from geneval.schemas import Input, MetricResult, Output


# Mock the DeepEval metrics to avoid initialization issues
@pytest.fixture(autouse=True)
def mock_deepeval_metrics():
    """Mock all DeepEval metrics to avoid initialization issues"""
    with patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric') as mock_answer_relevancy, \
         patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric') as mock_context_relevance, \
         patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric') as mock_faithfulness, \
         patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric') as mock_context_recall, \
         patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric') as mock_context_precision:
        
        # Create mock metric instances
        mock_answer_relevancy.return_value = Mock()
        mock_context_relevance.return_value = Mock()
        mock_faithfulness.return_value = Mock()
        mock_context_recall.return_value = Mock()
        mock_context_precision.return_value = Mock()
        
        yield {
            'answer_relevancy': mock_answer_relevancy,
            'context_relevance': mock_context_relevance,
            'faithfulness': mock_faithfulness,
            'context_recall': mock_context_recall,
            'context_precision': mock_context_precision
        }


class TestDeepEvalAdapter:
    """Test cases for DeepEvalAdapter"""
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_initialization_success_openai(self, mock_gpt_model):
        """Test successful DeepEvalAdapter initialization with OpenAI"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        # Mock GPTModel
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_manager == mock_llm_manager
        assert adapter.llm_info["provider"] == "openai"
        assert adapter.llm_info["model"] == "gpt-4o-mini"
        assert len(adapter.supported_metrics) > 0
        assert "faithfulness" in adapter.supported_metrics
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.AzureOpenAIModel')
    def test_initialization_success_azure_openai(self, mock_azure_model):
        """Test successful DeepEvalAdapter initialization with Azure OpenAI"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4",
            "temperature": 0.1,
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
            "openai_api_version": "2025-01-01-preview",
            "azure_endpoint": "https://test.openai.azure.com/"
        }
        
        # Mock AzureOpenAIModel
        mock_model = Mock()
        mock_azure_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "azure_openai"
        assert adapter.llm_info["model"] == "gpt-4"
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.AnthropicModel')
    def test_initialization_success_anthropic(self, mock_anthropic_model):
        """Test successful DeepEvalAdapter initialization with Anthropic"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "anthropic"
        mock_llm_manager.get_provider_config.return_value = {"model": "claude-3-5-haiku"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "claude-3-5-haiku",
            "temperature": 0.1
        }
        
        # Mock AnthropicModel
        mock_model = Mock()
        mock_anthropic_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "anthropic"
        assert adapter.llm_info["model"] == "claude-3-5-haiku"
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.AmazonBedrockModel')
    def test_initialization_success_amazon_bedrock(self, mock_bedrock_model):
        """Test successful DeepEvalAdapter initialization with Amazon Bedrock"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "amazon_bedrock"
        mock_llm_manager.get_provider_config.return_value = {"model": "anthropic.claude-3-sonnet-20240229-v1:0"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "temperature": 0.1
        }
        
        # Mock AmazonBedrockModel
        mock_model = Mock()
        mock_bedrock_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "amazon_bedrock"
        assert adapter.llm_info["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.GeminiModel')
    def test_initialization_success_gemini(self, mock_gemini_model):
        """Test successful DeepEvalAdapter initialization with Gemini"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "gemini"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-1.5-flash"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gemini-1.5-flash",
            "temperature": 0.1,
            "api_key": "test-key"
        }
        
        # Mock GeminiModel
        mock_model = Mock()
        mock_gemini_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "gemini"
        assert adapter.llm_info["model"] == "gemini-1.5-flash"
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.DeepSeekModel')
    def test_initialization_success_deepseek(self, mock_deepseek_model):
        """Test successful DeepEvalAdapter initialization with DeepSeek"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {"model": "deepseek-chat"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "deepseek-chat",
            "temperature": 0.1,
            "api_key": "test-key"
        }
        
        # Mock DeepSeekModel
        mock_model = Mock()
        mock_deepseek_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "deepseek"
        assert adapter.llm_info["model"] == "deepseek-chat"
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.OllamaModel')
    def test_initialization_success_ollama(self, mock_ollama_model):
        """Test successful DeepEvalAdapter initialization with Ollama"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "ollama"
        mock_llm_manager.get_provider_config.return_value = {"model": "llama3.2"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "llama3.2",
            "temperature": 0.1,
            "base_url": "http://localhost:11434"
        }
        
        # Mock OllamaModel
        mock_model = Mock()
        mock_ollama_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "ollama"
        assert adapter.llm_info["model"] == "llama3.2"
        assert adapter.gpt_model == mock_model
    
    @patch('geneval.adapters.deepeval_adapter.OllamaModel')
    def test_initialization_success_ollama_default_url(self, mock_ollama_model):
        """Test successful DeepEvalAdapter initialization with Ollama using default URL"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "ollama"
        mock_llm_manager.get_provider_config.return_value = {"model": "llama3.2"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "llama3.2",
            "temperature": 0.1
            # No base_url, should use default
        }
        
        # Mock OllamaModel
        mock_model = Mock()
        mock_ollama_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert adapter.llm_info["provider"] == "ollama"
        assert adapter.llm_info["model"] == "llama3.2"
        assert adapter.gpt_model == mock_model
    
    def test_initialization_missing_llm_manager(self):
        """Test DeepEvalAdapter initialization with missing LLM manager"""
        with pytest.raises(ValueError, match="LLMManager is required"):
            DeepEvalAdapter(None)
    
    def test_initialization_no_default_provider(self):
        """Test DeepEvalAdapter initialization with no default provider"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = None
        
        with pytest.raises(ValueError, match="No default LLM provider configured"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_create_gpt_model_openai_missing_model(self, mock_gpt_model):
        """Test OpenAI model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="OpenAI model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AzureOpenAIModel')
    def test_create_gpt_model_azure_openai_missing_model(self, mock_azure_model):
        """Test Azure OpenAI model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="Azure OpenAI model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AzureOpenAIModel')
    def test_create_gpt_model_azure_openai_missing_deployment_name(self, mock_azure_model):
        """Test Azure OpenAI model creation with missing deployment name"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4",
            "temperature": 0.1,
            "azure_openai_api_key": "test-key",
            "openai_api_version": "2025-01-01-preview",
            "azure_endpoint": "https://test.openai.azure.com/"
            # Missing deployment_name
        }
        
        with pytest.raises(ValueError, match="Azure OpenAI configuration incomplete"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AzureOpenAIModel')
    def test_create_gpt_model_azure_openai_missing_api_key(self, mock_azure_model):
        """Test Azure OpenAI model creation with missing API key"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4",
            "temperature": 0.1,
            "deployment_name": "gpt-4-deployment",
            "openai_api_version": "2025-01-01-preview",
            "azure_endpoint": "https://test.openai.azure.com/"
            # Missing azure_openai_api_key
        }
        
        with pytest.raises(ValueError, match="Azure OpenAI configuration incomplete"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AzureOpenAIModel')
    def test_create_gpt_model_azure_openai_missing_endpoint(self, mock_azure_model):
        """Test Azure OpenAI model creation with missing endpoint"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4",
            "temperature": 0.1,
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
            "openai_api_version": "2025-01-01-preview"
            # Missing azure_endpoint
        }
        
        with pytest.raises(ValueError, match="Azure OpenAI configuration incomplete"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AzureOpenAIModel')
    def test_create_gpt_model_azure_openai_default_api_version(self, mock_azure_model):
        """Test Azure OpenAI model creation with default API version"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "azure_openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4",
            "temperature": 0.1,
            "deployment_name": "gpt-4-deployment",
            "azure_openai_api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com/"
            # No openai_api_version, should use default
        }
        
        # Mock AzureOpenAIModel
        mock_model = Mock()
        mock_azure_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        # Verify AzureOpenAIModel was called with default API version
        mock_azure_model.assert_called_once_with(
            model_name="gpt-4",
            deployment_name="gpt-4-deployment",
            azure_openai_api_key="test-key",
            openai_api_version="2025-01-01-preview",  # Default value
            azure_endpoint="https://test.openai.azure.com/",
            temperature=0.1
        )
    
    @patch('geneval.adapters.deepeval_adapter.AnthropicModel')
    def test_create_gpt_model_anthropic_missing_model(self, mock_anthropic_model):
        """Test Anthropic model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "anthropic"
        mock_llm_manager.get_provider_config.return_value = {"model": "claude-3-5-haiku"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="Anthropic model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AmazonBedrockModel')
    def test_create_gpt_model_amazon_bedrock_missing_model(self, mock_bedrock_model):
        """Test Amazon Bedrock model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "amazon_bedrock"
        mock_llm_manager.get_provider_config.return_value = {"model": "anthropic.claude-3-sonnet-20240229-v1:0"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="Amazon Bedrock model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.GeminiModel')
    def test_create_gpt_model_gemini_missing_model(self, mock_gemini_model):
        """Test Gemini model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "gemini"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-1.5-flash"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="Gemini model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.GeminiModel')
    def test_create_gpt_model_gemini_missing_api_key(self, mock_gemini_model):
        """Test Gemini model creation with missing API key"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "gemini"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-1.5-flash"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gemini-1.5-flash",
            "temperature": 0.1
            # Missing api_key
        }
        
        with pytest.raises(ValueError, match="Google API key not found in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.DeepSeekModel')
    def test_create_gpt_model_deepseek_missing_model(self, mock_deepseek_model):
        """Test DeepSeek model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {"model": "deepseek-chat"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="DeepSeek model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.DeepSeekModel')
    def test_create_gpt_model_deepseek_missing_api_key(self, mock_deepseek_model):
        """Test DeepSeek model creation with missing API key"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "deepseek"
        mock_llm_manager.get_provider_config.return_value = {"model": "deepseek-chat"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "deepseek-chat",
            "temperature": 0.1
            # Missing api_key
        }
        
        with pytest.raises(ValueError, match="DeepSeek API key not found in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.OllamaModel')
    def test_create_gpt_model_ollama_missing_model(self, mock_ollama_model):
        """Test Ollama model creation with missing model"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "ollama"
        mock_llm_manager.get_provider_config.return_value = {"model": "llama3.2"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "temperature": 0.1
            # Missing model
        }
        
        with pytest.raises(ValueError, match="Ollama model not specified in configuration"):
            DeepEvalAdapter(mock_llm_manager)
    
    def test_create_gpt_model_unknown_provider(self):
        """Test model creation with unknown provider"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "unknown_provider"
        mock_llm_manager.get_provider_config.return_value = {"model": "unknown-model"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "unknown-model",
            "temperature": 0.1
        }
        
        with pytest.raises(ValueError, match="Provider unknown_provider not fully supported"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_create_test_case(self, mock_gpt_model):
        """Test test case creation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness", "answer_relevancy"]
        )
        
        test_case = adapter._create_test_case(input_data)
        
        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output == "The capital of France is Paris."
        assert test_case.expected_output == "Paris is the capital of France."
        assert test_case.retrieval_context == ["Paris is the capital and largest city of France."]
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_get_metrics_supported(self, mock_gpt_model):
        """Test getting supported metrics"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        metrics = adapter._get_metrics(["faithfulness", "answer_relevancy"])
        assert len(metrics) == 2
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_get_metrics_unsupported(self, mock_gpt_model):
        """Test getting unsupported metrics"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        with pytest.raises(ValueError, match="Unsupported metric: unsupported_metric"):
            adapter._get_metrics(["unsupported_metric"])
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_evaluate_success(self, mock_gpt_model):
        """Test successful evaluation"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        # Mock the metrics to return scores
        mock_faithfulness_metric = Mock()
        mock_faithfulness_metric.measure.return_value = 0.85
        mock_faithfulness_metric.reason = "Faithfulness evaluation completed"
        
        mock_answer_relevancy_metric = Mock()
        mock_answer_relevancy_metric.measure.return_value = 0.92
        mock_answer_relevancy_metric.reason = "Answer relevancy evaluation completed"
        
        adapter.available_metrics = {
            "faithfulness": mock_faithfulness_metric,
            "answer_relevancy": mock_answer_relevancy_metric
        }
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness", "answer_relevancy"]
        )
        
        output = adapter.evaluate(input_data)
        
        assert isinstance(output, Output)
        assert len(output.metrics) == 2
        assert output.metadata["framework"] == "deepeval"
        assert output.metadata["evaluation_successful"] is True
        assert output.metadata["llm_provider"] == "openai"
        assert output.metadata["llm_model"] == "gpt-4o-mini"
        assert output.metadata["test_case_count"] == 1
        
        # Check metric results
        faithfulness_metric = next(m for m in output.metrics if m.name == "faithfulness")
        answer_relevancy_metric = next(m for m in output.metrics if m.name == "answer_relevancy")
        
        assert faithfulness_metric.score == 0.85
        assert faithfulness_metric.tool_name == "deepeval"
        assert faithfulness_metric.details == "Faithfulness evaluation completed"
        
        assert answer_relevancy_metric.score == 0.92
        assert answer_relevancy_metric.tool_name == "deepeval"
        assert answer_relevancy_metric.details == "Answer relevancy evaluation completed"
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_evaluate_success_no_reason(self, mock_gpt_model):
        """Test successful evaluation when metrics don't have reason attribute"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        # Mock the metrics to return scores without reason
        mock_faithfulness_metric = Mock()
        mock_faithfulness_metric.measure.return_value = 0.85
        # Remove the reason attribute entirely to test the fallback
        del mock_faithfulness_metric.reason
        
        adapter.available_metrics = {
            "faithfulness": mock_faithfulness_metric
        }
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        output = adapter.evaluate(input_data)
        
        assert isinstance(output, Output)
        assert len(output.metrics) == 1
        assert output.metadata["evaluation_successful"] is True
        
        faithfulness_metric = output.metrics[0]
        assert faithfulness_metric.score == 0.85
        assert faithfulness_metric.tool_name == "deepeval"
        assert faithfulness_metric.details == "DeepEval faithfulness evaluation"  # Default explanation
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_evaluate_failure(self, mock_gpt_model):
        """Test evaluation failure handling"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        # Mock _get_metrics to raise an exception
        with patch.object(adapter, '_get_metrics', side_effect=Exception("Test error")):
            output = adapter.evaluate(input_data)
            
            assert isinstance(output, Output)
            assert len(output.metrics) == 0
            assert output.metadata["framework"] == "deepeval"
            assert output.metadata["evaluation_successful"] is False
            assert "Test error" in output.metadata["error"]
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_evaluate_with_llm_info_none(self, mock_gpt_model):
        """Test evaluation when llm_info is None"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        # Mock the metrics to return scores
        mock_faithfulness_metric = Mock()
        mock_faithfulness_metric.measure.return_value = 0.85
        # Remove the reason attribute to avoid validation issues
        del mock_faithfulness_metric.reason
        
        adapter.available_metrics = {
            "faithfulness": mock_faithfulness_metric
        }
        
        # Manually set llm_info to None to test this path
        adapter.llm_info = None
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        output = adapter.evaluate(input_data)
        
        assert isinstance(output, Output)
        assert len(output.metrics) == 1
        assert output.metadata["evaluation_successful"] is True
        # Should not have LLM-specific metadata
        assert "llm_provider" not in output.metadata
        assert "llm_model" not in output.metadata
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_evaluate_error_with_llm_info_none(self, mock_gpt_model):
        """Test error handling during evaluation when llm_info is None"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        # Manually set llm_info to None to test this path
        adapter.llm_info = None
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        # Mock _get_metrics to raise an exception
        with patch.object(adapter, '_get_metrics', side_effect=Exception("Test error")):
            output = adapter.evaluate(input_data)
            
            assert isinstance(output, Output)
            assert len(output.metrics) == 0
            assert output.metadata["evaluation_successful"] is False
            assert "Test error" in output.metadata["error"]
            # Should not have LLM-specific metadata
            assert "llm_provider" not in output.metadata
            assert "llm_model" not in output.metadata
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_supported_metrics_list(self, mock_gpt_model):
        """Test that all expected metrics are supported"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        expected_metrics = [
            "answer_relevancy",
            "context_relevance",
            "faithfulness",
            "context_recall",
            "context_precision"
        ]
        
        for metric in expected_metrics:
            assert metric in adapter.supported_metrics, f"Metric {metric} should be supported"
    
    @patch('geneval.adapters.deepeval_adapter.GPTModel')
    def test_llm_info_structure(self, mock_gpt_model):
        """Test LLM info structure and content"""
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini", "enabled": True}
        mock_llm_manager.get_global_settings.return_value = {"temperature": 0.1}
        mock_llm_manager.get_deepeval_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
        
        mock_model = Mock()
        mock_gpt_model.return_value = mock_model
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        assert "provider" in adapter.llm_info
        assert "model" in adapter.llm_info
        assert "provider_config" in adapter.llm_info
        assert "global_settings" in adapter.llm_info
        
        assert adapter.llm_info["provider"] == "openai"
        assert adapter.llm_info["model"] == "gpt-4o-mini"
        assert adapter.llm_info["provider_config"]["enabled"] is True
        assert adapter.llm_info["global_settings"]["temperature"] == 0.1
