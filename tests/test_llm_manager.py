import pytest
import os
from unittest.mock import Mock, patch, mock_open
from geneval.llm_manager import LLMManager


class TestLLMManager:
    """Test cases for LLMManager"""
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialization_with_config(self, mock_load_config):
        """Test LLMManager initialization with config"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        assert manager.config == mock_config
        assert "openai" in manager.provider_configs
        assert manager.get_default_provider() == "openai"
    
    def test_initialization_missing_config_path(self):
        """Test LLMManager initialization with missing config_path"""
        with pytest.raises(ValueError, match="config_path is required"):
            LLMManager(config_path="")
    
    def test_initialization_empty_config_path(self):
        """Test LLMManager initialization with empty config_path"""
        with pytest.raises(ValueError, match="config_path is required"):
            LLMManager(config_path=None)
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialization_with_empty_config(self, mock_load_config):
        """Test LLMManager initialization with empty config"""
        mock_load_config.return_value = {}
        
        manager = LLMManager(config_path="test_config.yaml")
        
        assert manager.config == {}
        assert manager.provider_configs == {}
        assert manager.get_default_provider() is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialization_with_disabled_providers(self, mock_load_config):
        """Test LLMManager initialization with disabled providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": False, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": False, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        assert len(manager.provider_configs) == 0
        assert "openai" not in manager.provider_configs
        assert "anthropic" not in manager.provider_configs
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_initialization_with_enabled_providers(self, mock_load_config):
        """Test LLMManager initialization with enabled providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        assert len(manager.provider_configs) == 2
        assert "openai" in manager.provider_configs
        assert "anthropic" in manager.provider_configs
        assert manager.provider_configs["openai"]["model"] == "gpt-4o-mini"
        assert manager.provider_configs["anthropic"]["model"] == "claude-3-5-haiku"
    
    def test_load_config_file_not_found(self):
        """Test _load_config with file not found"""
        with pytest.raises(FileNotFoundError, match="LLM configuration file not found"):
            LLMManager(config_path="nonexistent_config.yaml")
    
    def test_load_config_yaml_error(self):
        """Test _load_config with YAML parsing error"""
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")), \
             patch('geneval.llm_manager.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.side_effect = Exception("YAML parsing error")
            
            with pytest.raises(RuntimeError, match="Error loading LLM configuration"):
                LLMManager(config_path="test_config.yaml")
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_validate_config_single_default(self, mock_load_config):
        """Test config validation with single default provider"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        # Should not raise an exception
        manager = LLMManager(config_path="test_config.yaml")
        assert manager.get_default_provider() == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_validate_config_multiple_defaults(self, mock_load_config):
        """Test config validation with multiple default providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": True, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        # Should not raise an exception, just log a warning
        manager = LLMManager(config_path="test_config.yaml")
        # Should return the first default provider
        assert manager.get_default_provider() == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_validate_config_no_defaults(self, mock_load_config):
        """Test config validation with no default providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": False, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        # Should not raise an exception, just log a warning
        manager = LLMManager(config_path="test_config.yaml")
        assert manager.get_default_provider() is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_provider_config_existing(self, mock_load_config):
        """Test getting configuration for existing provider"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        config = manager.get_provider_config("openai")
        assert config["model"] == "gpt-4o-mini"
        assert config["enabled"] is True
        assert config["default"] is True
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_provider_config_nonexistent(self, mock_load_config):
        """Test getting configuration for non-existent provider"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        config = manager.get_provider_config("nonexistent")
        assert config == {}  # Returns empty dict for non-existent providers
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_default_provider_single(self, mock_load_config):
        """Test getting default provider when one exists"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        default = manager.get_default_provider()
        assert default == "openai"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_default_provider_none(self, mock_load_config):
        """Test getting default provider when none exists"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": False, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        default = manager.get_default_provider()
        assert default is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_global_settings_with_settings(self, mock_load_config):
        """Test getting global settings when they exist"""
        mock_config = {
            "providers": {"openai": {"enabled": True, "default": True}},
            "settings": {"temperature": 0.1, "max_tokens": 1000}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        settings = manager.get_global_settings()
        assert settings["temperature"] == 0.1
        assert settings["max_tokens"] == 1000
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_global_settings_none(self, mock_load_config):
        """Test getting global settings when none exist"""
        mock_config = {"providers": {"openai": {"enabled": True, "default": True}}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        settings = manager.get_global_settings()
        assert settings == {}
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_api_key_openai(self, mock_load_config):
        """Test getting OpenAI API key from environment"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            api_key = manager.get_api_key("openai")
            assert api_key == "test-key"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_api_key_anthropic(self, mock_load_config):
        """Test getting Anthropic API key from environment"""
        mock_config = {
            "providers": {
                "anthropic": {"enabled": True, "default": True, "model": "claude-3-5-haiku"}
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            api_key = manager.get_api_key("anthropic")
            assert api_key == "test-key"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_api_key_gemini(self, mock_load_config):
        """Test getting Gemini API key from environment"""
        mock_config = {
            "providers": {
                "gemini": {"enabled": True, "default": True, "model": "gemini-1.5-flash"}
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            api_key = manager.get_api_key("gemini")
            assert api_key == "test-key"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_api_key_deepseek(self, mock_load_config):
        """Test getting DeepSeek API key from environment"""
        mock_config = {
            "providers": {
                "deepseek": {"enabled": True, "default": True, "model": "deepseek-chat"}
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            api_key = manager.get_api_key("deepseek")
            assert api_key == "test-key"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_api_key_not_found(self, mock_load_config):
        """Test getting API key when not found"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {}, clear=True):
            manager = LLMManager(config_path="test_config.yaml")
            
            api_key = manager.get_api_key("openai")
            assert api_key is None
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_api_key_custom_env_var(self, mock_load_config):
        """Test getting API key from custom environment variable"""
        mock_config = {
            "providers": {
                "openai": {
                    "enabled": True, 
                    "default": True, 
                    "model": "gpt-4o-mini",
                    "api_key_env": "CUSTOM_OPENAI_KEY"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"CUSTOM_OPENAI_KEY": "custom-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            api_key = manager.get_api_key("openai")
            assert api_key == "custom-key"
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_available_providers(self, mock_load_config):
        """Test getting list of available providers"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "anthropic": {"enabled": True, "default": False, "model": "claude-3-5-haiku"},
                "gemini": {"enabled": False, "default": False, "model": "gemini-1.5-flash"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        available = manager.get_available_providers()
        assert len(available) == 2
        assert "openai" in available
        assert "anthropic" in available
        assert "gemini" not in available  # Disabled provider
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_is_provider_available(self, mock_load_config):
        """Test checking if provider is available"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"},
                "gemini": {"enabled": False, "default": False, "model": "gemini-1.5-flash"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        assert manager.is_provider_available("openai") is True
        assert manager.is_provider_available("gemini") is False
        assert manager.is_provider_available("nonexistent") is False
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_openai(self, mock_load_config):
        """Test getting DeepEval config for OpenAI"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("openai")
        assert deepeval_config["model"] == "gpt-4o-mini"
        assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_azure_openai(self, mock_load_config):
        """Test getting DeepEval config for Azure OpenAI"""
        mock_config = {
            "providers": {
                "azure_openai": {
                    "enabled": True, 
                    "default": True, 
                    "model": "gpt-4",
                    "deployment_name": "gpt-4-deployment",
                    "azure_openai_api_key": "test-key",
                    "openai_api_version": "2025-01-01-preview",
                    "azure_endpoint": "https://test.openai.azure.com/"
                }
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("azure_openai")
        assert deepeval_config["model"] == "gpt-4"
        assert deepeval_config["deployment_name"] == "gpt-4-deployment"
        assert deepeval_config["azure_openai_api_key"] == "test-key"
        assert deepeval_config["openai_api_version"] == "2025-01-01-preview"
        assert deepeval_config["azure_endpoint"] == "https://test.openai.azure.com/"
        assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_azure_openai_default_version(self, mock_load_config):
        """Test getting DeepEval config for Azure OpenAI with default version"""
        mock_config = {
            "providers": {
                "azure_openai": {
                    "enabled": True, 
                    "default": True, 
                    "model": "gpt-4",
                    "deployment_name": "gpt-4-deployment",
                    "azure_openai_api_key": "test-key",
                    "azure_endpoint": "https://test.openai.azure.com/"
                }
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("azure_openai")
        assert deepeval_config["openai_api_version"] == "2025-01-01-preview"  # Default value
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_amazon_bedrock(self, mock_load_config):
        """Test getting DeepEval config for Amazon Bedrock"""
        mock_config = {
            "providers": {
                "amazon_bedrock": {
                    "enabled": True, 
                    "default": True, 
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "region_name": "us-west-2"
                }
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("amazon_bedrock")
        assert deepeval_config["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert deepeval_config["region_name"] == "us-west-2"
        assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_amazon_bedrock_default_region(self, mock_load_config):
        """Test getting DeepEval config for Amazon Bedrock with default region"""
        mock_config = {
            "providers": {
                "amazon_bedrock": {
                    "enabled": True, 
                    "default": True, 
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0"
                }
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("amazon_bedrock")
        assert deepeval_config["region_name"] == "us-east-1"  # Default value
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_gemini(self, mock_load_config):
        """Test getting DeepEval config for Gemini"""
        mock_config = {
            "providers": {
                "gemini": {"enabled": True, "default": True, "model": "gemini-1.5-flash"}
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            deepeval_config = manager.get_deepeval_config("gemini")
            assert deepeval_config["model"] == "gemini-1.5-flash"
            assert deepeval_config["api_key"] == "test-key"
            assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_deepseek(self, mock_load_config):
        """Test getting DeepEval config for DeepSeek"""
        mock_config = {
            "providers": {
                "deepseek": {"enabled": True, "default": True, "model": "deepseek-chat"}
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            manager = LLMManager(config_path="test_config.yaml")
            
            deepeval_config = manager.get_deepeval_config("deepseek")
            assert deepeval_config["model"] == "deepseek-chat"
            assert deepeval_config["api_key"] == "test-key"
            assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_ollama(self, mock_load_config):
        """Test getting DeepEval config for Ollama"""
        mock_config = {
            "providers": {
                "ollama": {
                    "enabled": True, 
                    "default": True, 
                    "model": "llama3.2",
                    "base_url": "http://localhost:11434"
                }
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("ollama")
        assert deepeval_config["model"] == "llama3.2"
        assert deepeval_config["base_url"] == "http://localhost:11434"
        assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_ollama_default_url(self, mock_load_config):
        """Test getting DeepEval config for Ollama with default URL"""
        mock_config = {
            "providers": {
                "ollama": {"enabled": True, "default": True, "model": "llama3.2"}
            },
            "settings": {"temperature": 0.1}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("ollama")
        assert deepeval_config["base_url"] == "http://localhost:11434"  # Default value
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_unknown_provider(self, mock_load_config):
        """Test getting DeepEval config for unknown provider"""
        mock_config = {"providers": {}}
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("unknown")
        assert deepeval_config["model"] is None
        assert deepeval_config["temperature"] == 0.1
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_with_settings_override(self, mock_load_config):
        """Test getting DeepEval config with settings override"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            },
            "settings": {"temperature": 0.5}
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("openai")
        assert deepeval_config["temperature"] == 0.5
    
    @patch('geneval.llm_manager.LLMManager._load_config')
    def test_get_deepeval_config_without_settings(self, mock_load_config):
        """Test getting DeepEval config without settings section"""
        mock_config = {
            "providers": {
                "openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}
            }
        }
        mock_load_config.return_value = mock_config
        
        manager = LLMManager(config_path="test_config.yaml")
        
        deepeval_config = manager.get_deepeval_config("openai")
        assert deepeval_config["temperature"] == 0.1  # Default value
