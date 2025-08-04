"""
Comprehensive test suite for the GenEval framework.

This module contains tests for all major components:
- Schemas (Input, Output, MetricResult)
- LLMInitializer
- RAGASAdapter
- DeepEvalAdapter
- GenEvalFramework
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

from geneval.schemas import Input, Output, MetricResult
from geneval.llm_manager import LLMManager
from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.framework import GenEvalFramework


class TestSchemas:
    """Test cases for Pydantic schemas"""
    
    def test_input_schema_valid(self):
        """Test Input schema with valid data"""
        input_data = {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieval_context": "Paris is the capital and largest city of France.",
            "reference": "Paris is the capital of France.",
            "metrics": ["faithfulness", "answer_relevance"]
        }
        
        input_obj = Input(**input_data)
        assert input_obj.question == "What is the capital of France?"
        assert input_obj.response == "The capital of France is Paris."
        assert input_obj.retrieval_context == "Paris is the capital and largest city of France."
        assert input_obj.reference == "Paris is the capital of France."
        assert input_obj.metrics == ["faithfulness", "answer_relevance"]
    
    def test_input_schema_missing_fields(self):
        """Test Input schema validation with missing required fields"""
        with pytest.raises(ValueError):
            Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                # Missing required fields
            )
    
    def test_metric_result_schema(self):
        """Test MetricResult schema"""
        metric_result = MetricResult(
            name="faithfulness",
            score=0.85,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation"
        )
        
        assert metric_result.name == "faithfulness"
        assert metric_result.score == 0.85
        assert metric_result.tool_name == "ragas"
        assert metric_result.details == "RAGAS faithfulness evaluation"
    
    def test_output_schema(self):
        """Test Output schema"""
        metric_results = [
            MetricResult(
                name="faithfulness",
                score=0.85,
                tool_name="ragas",
                details="RAGAS faithfulness evaluation"
            ),
            MetricResult(
                name="answer_relevance",
                score=0.92,
                tool_name="deepeval",
                details="DeepEval answer relevance evaluation"
            )
        ]
        
        metadata = {
            "framework": "ragas",
            "total_metrics": 2,
            "evaluation_successful": True
        }
        
        output = Output(metrics=metric_results, metadata=metadata)
        assert len(output.metrics) == 2
        assert output.metadata["framework"] == "ragas"
        assert output.metadata["total_metrics"] == 2


class TestLLMManager:
    """Test cases for LLMManager"""
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialization_with_config(self, mock_load_config):
        """Test LLMManager initialization with config"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": True,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            },
            "settings": {
                "timeout": 30,
                "max_retries": 3,
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        assert manager.config == mock_config
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_missing_config_file(self, mock_load_config):
        """Test LLMManager initialization with missing config file"""
        mock_load_config.side_effect = FileNotFoundError("Config file not found")
        
        with pytest.raises(FileNotFoundError):
            LLMManager()
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_invalid_config_file(self, mock_load_config):
        """Test LLMManager initialization with invalid config file"""
        mock_load_config.side_effect = RuntimeError("Invalid YAML")
        
        with pytest.raises(RuntimeError):
            LLMManager()
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_select_provider_with_default(self, mock_get_or_create, mock_load_config):
        """Test provider selection with default provider"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": True,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        mock_get_or_create.return_value = Mock()
        
        result = manager.select_provider()
        assert result is True
        assert manager.selected_provider == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_select_provider_no_default(self, mock_load_config):
        """Test provider selection with no default provider"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": False,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        
        result = manager.select_provider()
        assert result is False
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_llm_info(self, mock_load_config):
        """Test getting LLM info"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": True,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.selected_provider = "openai"
        manager.selected_model = "gpt-4o-mini"
        
        llm_info = manager.get_llm_info()
        assert llm_info["provider"] == "openai"
        assert llm_info["model"] == "gpt-4o-mini"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_llm_info_no_provider(self, mock_load_config):
        """Test getting LLM info when no provider is selected"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.selected_provider = None
        
        llm_info = manager.get_llm_info()
        assert llm_info == {}
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_llm(self, mock_get_or_create, mock_load_config):
        """Test getting LLM instance"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": True,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        manager = LLMManager()
        manager.providers = {"openai": mock_llm}
        manager.selected_provider = "openai"
        mock_get_or_create.return_value = mock_llm
        
        llm = manager.get_llm()
        assert llm == mock_llm
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_llm_auto_select(self, mock_get_or_create, mock_load_config):
        """Test getting LLM with auto provider selection"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": True,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        manager = LLMManager()
        manager.providers = {"openai": mock_llm}
        manager.selected_provider = None  # No provider selected initially
        
        llm = manager.get_llm()
        assert llm == mock_llm
        assert manager.selected_provider == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_llm_no_provider_available(self, mock_get_or_create, mock_load_config):
        """Test getting LLM when no provider is available"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": False,  # No default
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {"openai": Mock()}
        manager.selected_provider = None
        
        llm = manager.get_llm()
        assert llm is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_provider_name(self, mock_get_or_create, mock_load_config):
        """Test getting provider name"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.selected_provider = "openai"
        
        provider_name = manager.get_provider_name()
        assert provider_name == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_model_name(self, mock_get_or_create, mock_load_config):
        """Test getting model name"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.selected_model = "gpt-4o-mini"
        
        model_name = manager.get_model_name()
        assert model_name == "gpt-4o-mini"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_available_providers(self, mock_load_config):
        """Test getting available providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True},
                "anthropic": {"enabled": True, "default": False}
            }
        }
        mock_load_config.return_value = mock_config

        manager = LLMManager()

        providers = manager.get_available_providers()
        assert set(providers) == {"openai", "anthropic"}
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_is_provider_available(self, mock_get_or_create, mock_load_config):
        """Test checking if provider is available"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True}
            }
        }
        mock_load_config.return_value = mock_config

        manager = LLMManager()
        mock_get_or_create.return_value = Mock()

        assert manager.is_provider_available("openai") is True
        assert manager.is_provider_available("anthropic") is False
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_configure_for_ragas(self, mock_get_or_create, mock_load_config):
        """Test configuring LLM for RAGAS"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        manager = LLMManager()
        manager.providers = {"openai": mock_llm}
        manager.selected_provider = "openai"
        manager.selected_model = "gpt-4o-mini"
        
        config = manager.configure_for_ragas()
        assert config["llm"] == mock_llm
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o-mini"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_configure_for_ragas_no_llm(self, mock_get_or_create, mock_load_config):
        """Test configuring LLM for RAGAS when no LLM is available"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {}
        manager.selected_provider = None
        
        config = manager.configure_for_ragas()
        assert config == {}
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_configure_for_deepeval(self, mock_get_or_create, mock_load_config):
        """Test configuring LLM for DeepEval"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        manager = LLMManager()
        manager.providers = {"openai": mock_llm}
        manager.selected_provider = "openai"
        manager.selected_model = "gpt-4o-mini"
        
        config = manager.configure_for_deepeval()
        assert config["llm"] == mock_llm
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o-mini"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_configure_for_deepeval_no_llm(self, mock_get_or_create, mock_load_config):
        """Test configuring LLM for DeepEval when no LLM is available"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {}
        manager.selected_provider = None
        
        config = manager.configure_for_deepeval()
        assert config == {}
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_validate_config_multiple_defaults(self, mock_get_or_create, mock_load_config):
        """Test config validation with multiple default providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True},
                "anthropic": {"enabled": True, "default": True}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        # Should not raise an exception, just log a warning
        assert manager.config == mock_config
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_validate_config_no_defaults(self, mock_get_or_create, mock_load_config):
        """Test config validation with no default providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": False},
                "anthropic": {"enabled": True, "default": False}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        # Should not raise an exception, just log a warning
        assert manager.config == mock_config
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialize_providers_success(self, mock_load_config):
        """Test successful provider configuration registration"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True},
                "anthropic": {"enabled": True, "default": False}
            }
        }
        mock_load_config.return_value = mock_config

        manager = LLMManager()
        # With lazy initialization, providers are only registered, not created
        assert "openai" in manager.provider_configs
        assert "anthropic" in manager.provider_configs
        assert len(manager.providers) == 0  # No providers created yet
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialize_providers_disabled(self, mock_load_config):
        """Test provider configuration with disabled providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": False, "default": True},
                "anthropic": {"enabled": False, "default": False}
            }
        }
        mock_load_config.return_value = mock_config

        manager = LLMManager()
        # With lazy initialization, disabled providers are not registered
        assert len(manager.provider_configs) == 0
        assert len(manager.providers) == 0
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_create_llm_provider_unknown(self, mock_get_or_create, mock_load_config):
        """Test creating unknown provider"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        result = manager._create_llm_provider("unknown", {})
        assert result is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch('geneval.llm_manager.ChatOpenAI')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_create_openai_provider_success(self, mock_openai, mock_get_or_create, mock_load_config):
        """Test successful OpenAI provider creation"""
        mock_config = {
            "providers": {"openai": {"enabled": True}},
            "settings": {
                "temperature": 0.1,
                "max_tokens": 1000,
                "timeout": 30
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        manager = LLMManager()
        result = manager._create_openai_provider({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"})
        
        assert result == mock_llm
        mock_openai.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000,
            timeout=30
        )
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch.dict(os.environ, {}, clear=True)
    def test_create_openai_provider_no_api_key(self, mock_get_or_create, mock_load_config):
        """Test OpenAI provider creation without API key"""
        mock_config = {"providers": {}, "settings": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        result = manager._create_openai_provider({"api_key_env": "OPENAI_API_KEY"})
        
        assert result is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch('geneval.llm_manager.ChatAnthropic')
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_create_anthropic_provider_success(self, mock_anthropic, mock_get_or_create, mock_load_config):
        """Test successful Anthropic provider creation"""
        mock_config = {
            "providers": {"anthropic": {"enabled": True}},
            "settings": {
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_anthropic.return_value = mock_llm
        
        manager = LLMManager()
        result = manager._create_anthropic_provider({"api_key_env": "ANTHROPIC_API_KEY", "model": "claude-3-5-haiku"})
        
        assert result == mock_llm
        mock_anthropic.assert_called_once_with(
            model="claude-3-5-haiku",
            temperature=0.1,
            max_tokens=1000
        )
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch.dict(os.environ, {}, clear=True)
    def test_create_anthropic_provider_no_api_key(self, mock_get_or_create, mock_load_config):
        """Test Anthropic provider creation without API key"""
        mock_config = {"providers": {}, "settings": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        result = manager._create_anthropic_provider({"api_key_env": "ANTHROPIC_API_KEY"})
        
        assert result is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch('geneval.llm_manager.ChatGoogleGenerativeAI')
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
    def test_create_gemini_provider_success(self, mock_gemini, mock_get_or_create, mock_load_config):
        """Test successful Gemini provider creation"""
        mock_config = {
            "providers": {"gemini": {"enabled": True}},
            "settings": {
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        manager = LLMManager()
        result = manager._create_gemini_provider({"api_key_env": "GOOGLE_API_KEY", "model": "gemini-1.5-flash"})
        
        assert result == mock_llm
        mock_gemini.assert_called_once_with(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=1000
        )
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch.dict(os.environ, {}, clear=True)
    def test_create_gemini_provider_no_api_key(self, mock_get_or_create, mock_load_config):
        """Test Gemini provider creation without API key"""
        mock_config = {"providers": {}, "settings": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        result = manager._create_gemini_provider({"api_key_env": "GOOGLE_API_KEY"})
        
        assert result is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch('geneval.llm_manager.Ollama')
    def test_create_ollama_provider_success(self, mock_ollama, mock_get_or_create, mock_load_config):
        """Test successful Ollama provider creation"""
        mock_config = {
            "providers": {"ollama": {"enabled": True}},
            "settings": {
                "temperature": 0.1
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        manager = LLMManager()
        result = manager._create_ollama_provider({"base_url": "http://localhost:11434", "model": "llama3.2"})
        
        assert result == mock_llm
        mock_ollama.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.1
        )
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    @patch('geneval.llm_manager.Ollama')
    def test_create_ollama_provider_default_url(self, mock_ollama, mock_get_or_create, mock_load_config):
        """Test Ollama provider creation with default URL"""
        mock_config = {
            "providers": {"ollama": {"enabled": True}},
            "settings": {
                "temperature": 0.1
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        manager = LLMManager()
        result = manager._create_ollama_provider({"model": "llama3.2"})
        
        assert result == mock_llm
        mock_ollama.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.1
        )
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_select_provider_specific(self, mock_get_or_create, mock_load_config):
        """Test selecting a specific provider"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {"openai": Mock(), "anthropic": Mock()}
        
        result = manager.select_provider("anthropic")
        assert result is True
        assert manager.selected_provider == "anthropic"
        assert manager.selected_model == "claude-3-5-haiku"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_select_provider_not_available(self, mock_get_or_create, mock_load_config):
        """Test selecting a provider that's not available"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config

        manager = LLMManager()
        mock_get_or_create.return_value = None  # Provider creation fails

        result = manager.select_provider("anthropic")
        assert result is False
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_select_provider_auto_no_providers(self, mock_get_or_create, mock_load_config):
        """Test auto provider selection with no providers available"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config

        manager = LLMManager()
        mock_get_or_create.return_value = None  # Provider creation fails

        result = manager.select_provider()
        assert result is False
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_select_provider_auto_no_default(self, mock_get_or_create, mock_load_config):
        """Test auto provider selection with no default provider"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": False, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {"openai": Mock(), "anthropic": Mock()}
        
        result = manager.select_provider()
        assert result is False
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_select_provider_auto_multiple_defaults(self, mock_get_or_create, mock_load_config):
        """Test auto provider selection with multiple default providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": True, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {"openai": Mock(), "anthropic": Mock()}
        
        result = manager.select_provider()
        assert result is True
        assert manager.selected_provider == "openai"  # First one should be selected
        assert manager.selected_model == "gpt-4o-mini"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_llm_info_with_provider_config(self, mock_get_or_create, mock_load_config):
        """Test getting LLM info with provider config"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "default": True,
                    "model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.selected_provider = "openai"
        manager.selected_model = "gpt-4o-mini"
        manager.config = mock_config
        
        llm_info = manager.get_llm_info()
        assert llm_info["provider"] == "openai"
        assert llm_info["model"] == "gpt-4o-mini"
        assert llm_info["provider_config"]["api_key_env"] == "OPENAI_API_KEY"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_llm_auto_select_success(self, mock_get_or_create, mock_load_config):
        """Test getting LLM with successful auto provider selection"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_llm = Mock()
        manager = LLMManager()
        manager.providers = {"openai": mock_llm}
        manager.selected_provider = None
        
        llm = manager.get_llm()
        assert llm == mock_llm
        assert manager.selected_provider == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_get_llm_auto_select_failure(self, mock_get_or_create, mock_load_config):
        """Test getting LLM with failed auto provider selection"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": False, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager()
        manager.providers = {"openai": Mock()}
        manager.selected_provider = None
        
        llm = manager.get_llm()
        assert llm is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_validate_config_multiple_defaults_warning(self, mock_get_or_create, mock_load_config):
        """Test config validation with multiple defaults (should log warning)"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True},
                "anthropic": {"enabled": True, "default": True}
            }
        }
        mock_load_config.return_value = mock_config
        
        # Should not raise exception, just log warning
        manager = LLMManager()
        assert manager.config == mock_config
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_validate_config_no_defaults_warning(self, mock_get_or_create, mock_load_config):
        """Test config validation with no defaults (should log warning)"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": False},
                "anthropic": {"enabled": True, "default": False}
            }
        }
        mock_load_config.return_value = mock_config
        
        # Should not raise exception, just log warning
        manager = LLMManager()
        assert manager.config == mock_config
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    @patch('geneval.llm_manager.LLMManager._get_or_create_provider')
    def test_validate_config_single_default(self, mock_get_or_create, mock_load_config):
        """Test config validation with single default (should log info)"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True}
            }
        }
        mock_load_config.return_value = mock_config
        
        # Should not raise exception, just log info
        manager = LLMManager()
        assert manager.config == mock_config




    def test_load_config_yaml_error(self):
        """Test _load_config with YAML parsing error"""
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")), \
             patch('geneval.llm_manager.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.side_effect = Exception("YAML parsing error")
            
            with pytest.raises(RuntimeError, match="Error loading LLM configuration"):
                LLMManager()
    
    def test_initialize_providers_exception_handling(self):
        """Test lazy provider initialization exception handling"""
        with patch('geneval.llm_manager.LLMManager._load_config') as mock_load_config, \
             patch('geneval.llm_manager.LLMManager._create_llm_provider') as mock_create_provider:

            mock_config = {
                "providers": {
                    "openai": {"enabled": True, "default": True},
                    "anthropic": {"enabled": True, "default": False}
                }
            }
            mock_load_config.return_value = mock_config

            # Mock _create_llm_provider to raise exception for one provider
            def create_provider_side_effect(provider_name, config):
                if provider_name == "anthropic":
                    raise Exception("Provider creation failed")
                return Mock()

            mock_create_provider.side_effect = create_provider_side_effect

            # With lazy initialization, providers are only registered, not created
            manager = LLMManager()
            assert "openai" in manager.provider_configs
            assert "anthropic" in manager.provider_configs
            assert len(manager.providers) == 0  # No providers created yet
            assert "anthropic" not in manager.providers
    
    def test_create_llm_provider_exception_handling(self):
        """Test _create_llm_provider exception handling"""
        with patch('geneval.llm_manager.LLMManager._load_config') as mock_load_config, \
             patch('geneval.llm_manager.LLMManager._create_openai_provider') as mock_create_openai:
            
            mock_config = {
                "providers": {
                    "openai": {"enabled": True, "default": True}
                }
            }
            mock_load_config.return_value = mock_config
            mock_create_openai.side_effect = Exception("OpenAI creation failed")
            
            # Should not raise exception, just log error and return None
            manager = LLMManager()
            assert len(manager.providers) == 0
    
    def test_create_openai_provider_exception_handling(self):
        """Test _create_openai_provider exception handling through _create_llm_provider"""
        with patch('geneval.llm_manager.LLMManager._load_config') as mock_load_config, \
             patch('geneval.llm_manager.ChatOpenAI') as mock_openai:
            
            mock_config = {
                "providers": {
                    "openai": {"enabled": True, "default": True}
                },
                "settings": {
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "timeout": 30
                }
            }
            mock_load_config.return_value = mock_config
            mock_openai.side_effect = Exception("OpenAI initialization failed")
            
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                # Create manager instance first
                manager = LLMManager()
                # Should not raise exception, just return None
                result = manager._create_llm_provider("openai", {"api_key_env": "OPENAI_API_KEY"})
                assert result is None
    
    def test_create_anthropic_provider_exception_handling(self):
        """Test _create_anthropic_provider exception handling through _create_llm_provider"""
        with patch('geneval.llm_manager.LLMManager._load_config') as mock_load_config, \
             patch('geneval.llm_manager.ChatAnthropic') as mock_anthropic:
            
            mock_config = {
                "providers": {
                    "anthropic": {"enabled": True, "default": True}
                },
                "settings": {
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            }
            mock_load_config.return_value = mock_config
            mock_anthropic.side_effect = Exception("Anthropic initialization failed")
            
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
                # Create manager instance first
                manager = LLMManager()
                # Should not raise exception, just return None
                result = manager._create_llm_provider("anthropic", {"api_key_env": "ANTHROPIC_API_KEY"})
                assert result is None
    
    def test_create_gemini_provider_exception_handling(self):
        """Test _create_gemini_provider exception handling through _create_llm_provider"""
        with patch('geneval.llm_manager.LLMManager._load_config') as mock_load_config, \
             patch('geneval.llm_manager.ChatGoogleGenerativeAI') as mock_gemini:
            
            mock_config = {
                "providers": {
                    "gemini": {"enabled": True, "default": True}
                },
                "settings": {
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            }
            mock_load_config.return_value = mock_config
            mock_gemini.side_effect = Exception("Gemini initialization failed")
            
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
                # Create manager instance first
                manager = LLMManager()
                # Should not raise exception, just return None
                result = manager._create_llm_provider("gemini", {"api_key_env": "GOOGLE_API_KEY"})
                assert result is None
    
    def test_create_ollama_provider_exception_handling(self):
        """Test _create_ollama_provider exception handling through _create_llm_provider"""
        with patch('geneval.llm_manager.LLMManager._load_config') as mock_load_config, \
             patch('geneval.llm_manager.Ollama') as mock_ollama:
            
            mock_config = {
                "providers": {
                    "ollama": {"enabled": True, "default": True}
                },
                "settings": {
                    "temperature": 0.1
                }
            }
            mock_load_config.return_value = mock_config
            mock_ollama.side_effect = Exception("Ollama initialization failed")
            
            # Create manager instance first
            manager = LLMManager()
            # Should not raise exception, just return None
            result = manager._create_llm_provider("ollama", {"model": "llama3.2"})
            assert result is None


class TestRAGASAdapter:
    """Test cases for RAGASAdapter"""
    
    def test_initialization_without_llm(self):
        """Test RAGASAdapter initialization without LLM - should fail"""
        with pytest.raises(ValueError, match="LLMManager is required"):
            RAGASAdapter(None)
    
    def test_initialization_no_llm_available(self):
        """Test RAGASAdapter initialization when no LLM is available"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = None  # No LLM available
        
        with pytest.raises(ValueError, match="No LLM available"):
            RAGASAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.ragas_adapter.LLMContextPrecisionWithoutReference')
    @patch('geneval.adapters.ragas_adapter.LLMContextPrecisionWithReference')
    @patch('geneval.adapters.ragas_adapter.LLMContextRecall')
    @patch('geneval.adapters.ragas_adapter.ContextEntityRecall')
    @patch('geneval.adapters.ragas_adapter.NoiseSensitivity')
    @patch('geneval.adapters.ragas_adapter.ResponseRelevancy')
    @patch('geneval.adapters.ragas_adapter.Faithfulness')
    def test_initialization_with_llm(self, mock_faithfulness, mock_response_relevancy, 
                                   mock_noise_sensitivity, mock_context_entity_recall,
                                   mock_context_recall, mock_context_precision_with_ref,
                                   mock_context_precision_without_ref):
        """Test RAGASAdapter initialization with LLM"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        
        expected_metrics = [
            "context_precision_without_reference",
            "context_precision_with_reference", 
            "context_recall",
            "context_entity_recall",
            "noise_sensitivity",
            "response_relevancy",
            "faithfulness"
        ]
        
        assert set(adapter.supported_metrics) == set(expected_metrics)
    
    @patch('geneval.adapters.ragas_adapter.LLMContextPrecisionWithoutReference')
    def test_initialization_with_llm_exception(self, mock_metric):
        """Test RAGASAdapter initialization with LLM when metric initialization fails"""
        # Mock metric to raise exception
        mock_metric.side_effect = Exception("Metric initialization failed")
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        with pytest.raises(RuntimeError, match="Failed to initialize RAGAS metrics"):
            RAGASAdapter(mock_llm_manager)
    
    def test_prepare_dataset(self):
        """Test dataset preparation"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        dataset = adapter._prepare_dataset(input_data)
        
        assert dataset["question"][0] == "What is the capital of France?"
        assert dataset["answer"][0] == "The capital of France is Paris."
        assert dataset["contexts"][0] == ["Paris is the capital and largest city of France."]
        assert dataset["ground_truths"][0] == ["Paris is the capital of France."]
    
    def test_get_metrics_unsupported(self):
        """Test getting unsupported metrics"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {}
        adapter.supported_metrics = []
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            adapter._get_metrics(["unsupported_metric"])
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    @patch('geneval.adapters.ragas_adapter.Dataset')
    def test_evaluate_success(self, mock_dataset, mock_evaluate):
        """Test successful evaluation"""
        # Mock dataset
        mock_dataset.from_dict.return_value = Mock()
        
        # Mock evaluation results
        mock_results = Mock()
        mock_results.scores = [{"faithfulness": 0.85}]
        mock_evaluate.return_value = mock_results
        
        # Mock metrics
        mock_metric = Mock()
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": mock_metric}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is True
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "faithfulness"
        assert result.metrics[0].score == 0.85
        assert result.metrics[0].tool_name == "ragas"
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    def test_evaluate_exception(self, mock_evaluate):
        """Test evaluation with exception"""
        mock_evaluate.side_effect = Exception("Test error")
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is False
        assert "Test error" in result.metadata["error"]
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    def test_evaluate_metric_not_found_in_results(self, mock_evaluate):
        """Test evaluation when metric is not found in RAGAS results"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        # Mock evaluate to return results without the expected metric
        mock_results = Mock()
        mock_results.scores = [{"unexpected_metric": 0.8}]  # Different metric name
        mock_evaluate.return_value = mock_results
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        # Should still return a result, but with empty metrics due to metric not found
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is True
        assert result.metadata["total_metrics"] == 0
        assert len(result.metrics) == 0
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    def test_evaluate_exception_no_llm_info(self, mock_evaluate):
        """Test evaluation exception when llm_info is None"""
        mock_evaluate.side_effect = Exception("Test error")
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        # Set llm_info to None after initialization to test the else branch
        adapter.llm_info = None
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is False
        assert "Test error" in result.metadata["error"]
        # Should not have LLM info in metadata since llm_info is None
        assert "llm_provider" not in result.metadata
        assert "llm_model" not in result.metadata
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    def test_evaluate_success_with_llm_info(self, mock_evaluate):
        """Test successful evaluation with LLM info in metadata"""
        # Mock successful evaluation
        mock_results = Mock()
        mock_results.scores = [{"faithfulness": 0.85}]
        mock_evaluate.return_value = mock_results
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is True
        assert result.metadata["llm_provider"] == "openai"
        assert result.metadata["llm_model"] == "gpt-4o-mini"
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "faithfulness"
        assert result.metrics[0].score == 0.85
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    def test_evaluate_unexpected_results_format(self, mock_evaluate):
        """Test evaluation with unexpected RAGAS results format"""
        # Mock evaluate to return results without scores attribute
        mock_results = Mock()
        del mock_results.scores  # Remove scores attribute to trigger unexpected format
        mock_evaluate.return_value = mock_results
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        adapter = RAGASAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        # Should still return a result, but with empty metrics due to unexpected format
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is True
        assert result.metadata["total_metrics"] == 0
        assert len(result.metrics) == 0


class TestDeepEvalAdapter:
    """Test cases for DeepEvalAdapter"""
    
    def test_initialization_without_llm(self):
        """Test DeepEvalAdapter initialization without LLM - should fail"""
        with pytest.raises(ValueError, match="LLMManager is required"):
            DeepEvalAdapter(None)
    
    def test_initialization_no_llm_available(self):
        """Test DeepEvalAdapter initialization when no LLM is available"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = None  # No LLM available
        
        with pytest.raises(ValueError, match="No LLM available"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric')
    def test_initialization_with_llm(self, mock_context_precision, mock_context_recall,
                                   mock_faithfulness, mock_context_relevance, mock_answer_relevance):
        """Test DeepEvalAdapter initialization with LLM"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.configure_for_deepeval.return_value = {}
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        
        expected_metrics = [
            "answer_relevance",
            "context_relevance",
            "faithfulness",
            "context_recall",
            "context_precision"
        ]
        
        assert set(adapter.supported_metrics) == set(expected_metrics)
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    def test_initialization_metric_exception(self, mock_answer_relevance):
        """Test DeepEvalAdapter initialization when metric creation fails"""
        mock_answer_relevance.side_effect = Exception("Metric initialization failed")
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        
        with pytest.raises(RuntimeError, match="Failed to initialize DeepEval metrics"):
            DeepEvalAdapter(mock_llm_manager)
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric')
    def test_create_test_case(self, mock_context_precision, mock_context_recall,
                            mock_faithfulness, mock_context_relevance, mock_answer_relevance):
        """Test test case creation"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.configure_for_deepeval.return_value = {}
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        test_case = adapter._create_test_case(input_data)
        
        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output == "The capital of France is Paris."
        assert test_case.expected_output == "Paris is the capital of France."
        assert test_case.retrieval_context == ["Paris is the capital and largest city of France."]
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric')
    def test_get_metrics_unsupported(self, mock_context_precision, mock_context_recall,
                                   mock_faithfulness, mock_context_relevance, mock_answer_relevance):
        """Test getting unsupported metrics"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.configure_for_deepeval.return_value = {}
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        adapter.available_metrics = {}
        adapter.supported_metrics = []
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            adapter._get_metrics(["unsupported_metric"])
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric')
    def test_evaluate_success(self, mock_context_precision, mock_context_recall,
                            mock_faithfulness, mock_context_relevance, mock_answer_relevance):
        """Test successful evaluation"""
        # Mock metric
        mock_metric = Mock()
        mock_metric.measure.return_value = 0.92
        mock_metric.reason = "DeepEval faithfulness evaluation"
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.configure_for_deepeval.return_value = {}
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": mock_metric}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "deepeval"
        assert result.metadata["evaluation_successful"] is True
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "faithfulness"
        assert result.metrics[0].score == 0.92
        assert result.metrics[0].tool_name == "deepeval"
        assert "faithfulness evaluation" in result.metrics[0].details
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric')
    def test_evaluate_exception(self, mock_context_precision, mock_context_recall,
                              mock_faithfulness, mock_context_relevance, mock_answer_relevance):
        """Test evaluation with exception"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.configure_for_deepeval.return_value = {}
        
        adapter = DeepEvalAdapter(mock_llm_manager)
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        # Mock metric to raise exception
        mock_metric = Mock()
        mock_metric.measure.side_effect = Exception("Test error")
        adapter.available_metrics["faithfulness"] = mock_metric
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "deepeval"
        assert result.metadata["evaluation_successful"] is False
        assert "Test error" in result.metadata["error"]
        assert len(result.metrics) == 0


class TestGenEvalFramework:
    """Test cases for GenEvalFramework"""
    
    def test_initialization(self, mock_framework):
        """Test framework initialization with auto-initialized LLM manager"""
        framework = mock_framework
        
        assert "ragas" in framework.adapters
        assert "deepeval" in framework.adapters
        assert isinstance(framework.adapters["ragas"], Mock)
        assert isinstance(framework.adapters["deepeval"], Mock)
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    def test_initialization_with_llm(self, mock_deepeval_adapter_class, mock_ragas_adapter_class, mock_llm_manager):
        """Test framework initialization with LLM"""
        mock_ragas_adapter = Mock()
        mock_deepeval_adapter = Mock()
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        framework = GenEvalFramework(mock_llm_manager)
        
        assert "ragas" in framework.adapters
        assert "deepeval" in framework.adapters
        assert framework.adapters["ragas"] == mock_ragas_adapter
        assert framework.adapters["deepeval"] == mock_deepeval_adapter
        mock_ragas_adapter_class.assert_called_once_with(mock_llm_manager)
        mock_deepeval_adapter_class.assert_called_once_with(mock_llm_manager)
    
    @patch('geneval.framework.LLMManager')
    def test_initialization_fails_no_default_provider(self, mock_llm_manager_class):
        """Test framework initialization fails when no default provider is configured"""
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.select_provider.return_value = False  # No default provider
        mock_llm_manager_class.return_value = mock_llm_manager
        
        with pytest.raises(ValueError, match="No default LLM provider configured"):
            GenEvalFramework()
    
    def test_evaluate_single_metric_ragas(self, mock_framework):
        """Test evaluation with single RAGAS metric"""
        # Mock RAGAS evaluation result
        mock_result = Output(
            metrics=[
                MetricResult(
                    name="faithfulness",
                    score=0.85,
                    tool_name="ragas",
                    details="RAGAS faithfulness evaluation"
                )
            ],
            metadata={"framework": "ragas", "evaluation_successful": True}
        )
        
        framework = mock_framework
        
        # Set up mock adapter to return the expected result
        framework.adapters["ragas"].evaluate.return_value = mock_result
        framework.adapters["ragas"].supported_metrics = ["faithfulness"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["ragas.faithfulness"]
        )
        
        assert "ragas.faithfulness" in result
        assert result["ragas.faithfulness"][0] == "ragas"
        assert result["ragas.faithfulness"][1] == mock_result
    
    def test_evaluate_single_metric_deepeval(self, mock_framework):
        """Test evaluation with single DeepEval metric"""
        # Mock DeepEval evaluation result
        mock_result = Output(
            metrics=[
                MetricResult(
                    name="faithfulness",
                    score=0.92,
                    tool_name="deepeval",
                    details="DeepEval faithfulness evaluation"
                )
            ],
            metadata={"framework": "deepeval", "evaluation_successful": True}
        )
        
        framework = mock_framework
        
        # Set up mock adapter to return the expected result
        framework.adapters["deepeval"].evaluate.return_value = mock_result
        framework.adapters["deepeval"].supported_metrics = ["faithfulness"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["deepeval.faithfulness"]
        )
        
        assert "deepeval.faithfulness" in result
        assert result["deepeval.faithfulness"][0] == "deepeval"
        assert result["deepeval.faithfulness"][1] == mock_result
    
    def test_evaluate_unknown_adapter(self, mock_framework):
        """Test evaluation with unknown adapter"""
        framework = mock_framework
        
        with pytest.raises(ValueError, match="Unknown adapter"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["unknown.faithfulness"]
            )
    
    def test_evaluate_unsupported_metric(self, mock_framework):
        """Test evaluation with unsupported metric"""
        framework = mock_framework
        
        # Mock adapters to have no supported metrics
        framework.adapters["ragas"].supported_metrics = []
        framework.adapters["deepeval"].supported_metrics = []
        
        with pytest.raises(ValueError, match="does not support metric"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["ragas.unsupported_metric"]
            )
    
    def test_evaluate_no_adapter_supports_metric(self, mock_framework):
        """Test evaluation when no adapter supports the metric"""
        framework = mock_framework
        
        # Mock adapters to have no supported metrics
        framework.adapters["ragas"].supported_metrics = []
        framework.adapters["deepeval"].supported_metrics = []
        
        with pytest.raises(ValueError, match="No adapter supports metric"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"]
            )
    
    def test_evaluate_multiple_metrics(self, mock_framework):
        """Test evaluation with multiple metrics from different adapters"""
        # Mock RAGAS result
        mock_ragas_result = Output(
            metrics=[MetricResult(name="faithfulness", score=0.85, tool_name="ragas", details="RAGAS")],
            metadata={"framework": "ragas", "evaluation_successful": True}
        )
        
        # Mock DeepEval result
        mock_deepeval_result = Output(
            metrics=[MetricResult(name="answer_relevance", score=0.92, tool_name="deepeval", details="DeepEval")],
            metadata={"framework": "deepeval", "evaluation_successful": True}
        )
        
        framework = mock_framework
        
        # Set up mock adapters to return expected results
        framework.adapters["ragas"].evaluate.return_value = mock_ragas_result
        framework.adapters["deepeval"].evaluate.return_value = mock_deepeval_result
        framework.adapters["ragas"].supported_metrics = ["faithfulness"]
        framework.adapters["deepeval"].supported_metrics = ["answer_relevance"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["ragas.faithfulness", "deepeval.answer_relevance"]
        )
        
        assert len(result) == 2
        assert "ragas.faithfulness" in result
        assert "deepeval.answer_relevance" in result
        assert result["ragas.faithfulness"][0] == "ragas"
        assert result["deepeval.answer_relevance"][0] == "deepeval"
    
    def test_evaluate_metric_without_adapter_prefix(self, mock_framework):
        """Test evaluation with metric name without adapter prefix (uses all supporting adapters)"""
        # Mock RAGAS result
        mock_ragas_result = Output(
            metrics=[MetricResult(name="faithfulness", score=0.85, tool_name="ragas", details="RAGAS")],
            metadata={"framework": "ragas", "evaluation_successful": True}
        )
        
        # Mock DeepEval result
        mock_deepeval_result = Output(
            metrics=[MetricResult(name="faithfulness", score=0.92, tool_name="deepeval", details="DeepEval")],
            metadata={"framework": "deepeval", "evaluation_successful": True}
        )
        
        framework = mock_framework
        
        # Set up mock adapters to return expected results
        framework.adapters["ragas"].evaluate.return_value = mock_ragas_result
        framework.adapters["deepeval"].evaluate.return_value = mock_deepeval_result
        framework.adapters["ragas"].supported_metrics = ["faithfulness"]
        framework.adapters["deepeval"].supported_metrics = ["faithfulness"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]  # No adapter prefix
        )
        
        assert len(result) == 2
        assert "ragas.faithfulness" in result
        assert "deepeval.faithfulness" in result
        assert result["ragas.faithfulness"][0] == "ragas"
        assert result["deepeval.faithfulness"][0] == "deepeval"
        
        # Verify both adapters were called
        framework.adapters["ragas"].evaluate.assert_called_once()
        framework.adapters["deepeval"].evaluate.assert_called_once()





# Fixtures for common test data
@pytest.fixture
def sample_input_data():
    """Sample input data for tests"""
    return {
        "question": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "retrieval_context": "Paris is the capital and largest city of France.",
        "reference": "Paris is the capital of France.",
        "metrics": ["faithfulness", "answer_relevance"]
    }


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for tests"""
    mock_llm_manager = Mock()
    mock_llm_manager.get_llm.return_value = Mock()
    mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
    mock_llm_manager.select_provider.return_value = True
    return mock_llm_manager

@pytest.fixture
def sample_config():
    """Sample configuration fixture"""
    return {
        "providers": {
            "openai": {
                "enabled": True,
                "default": True,
                "api_key_env": "OPENAI_API_KEY",
                "model": "gpt-4o-mini"
            },
            "anthropic": {
                "enabled": True,
                "default": False,
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-3-5-haiku"
            }
        },
        "settings": {
            "temperature": 0.1,
            "max_tokens": 1000,
            "timeout": 30
        }
    }

@pytest.fixture
def mock_llm_manager_with_config(sample_config):
    """Mock LLM manager with config fixture"""
    mock_manager = Mock()
    mock_manager.get_llm.return_value = Mock()
    mock_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
    mock_manager.select_provider.return_value = True
    mock_manager.config = sample_config
    mock_manager.providers = {"openai": Mock(), "anthropic": Mock()}
    return mock_manager


@pytest.fixture
def mock_framework():
    """Mock framework with proper adapters"""
    with patch('geneval.framework.LLMManager') as mock_llm_manager_class, \
         patch('geneval.framework.RAGASAdapter') as mock_ragas_adapter_class, \
         patch('geneval.framework.DeepEvalAdapter') as mock_deepeval_adapter_class:
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_llm.return_value = Mock()
        mock_llm_manager.get_llm_info.return_value = {"provider": "openai", "model": "gpt-4o-mini"}
        mock_llm_manager.select_provider.return_value = True
        mock_llm_manager_class.return_value = mock_llm_manager
        
        mock_ragas_adapter = Mock()
        mock_deepeval_adapter = Mock()
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        framework = GenEvalFramework()
        
        # Set up mock adapters with proper methods
        framework.adapters = {
            "ragas": mock_ragas_adapter,
            "deepeval": mock_deepeval_adapter
        }
        
        # Set up mock adapter methods
        mock_ragas_adapter.evaluate.return_value = Mock()
        mock_deepeval_adapter.evaluate.return_value = Mock()
        
        yield framework


@pytest.fixture
def sample_metric_result():
    """Sample metric result for tests"""
    return MetricResult(
        name="faithfulness",
        score=0.85,
        tool_name="ragas",
        details="RAGAS faithfulness evaluation"
    )
