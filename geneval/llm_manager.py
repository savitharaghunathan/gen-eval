"""
LLM Manager using LangChain for GenEval framework.

This module provides a unified interface for multiple LLM providers using LangChain.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama


class LLMManager:
    """Manages LLM providers with lazy initialization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM manager
        
        Args:
            config_path: Path to YAML configuration file (default: config/llm_config.yaml)
        """
        self.config_path = config_path or "config/llm_config.yaml"
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        self._validate_config()
        
        # Store provider configurations for lazy initialization
        self.provider_configs = {}
        self.providers = {}  # Will store actual LLM instances when created
        self.selected_provider = None
        self.selected_model = None
        
        # Store enabled provider configurations
        providers_config = self.config.get("providers", {})
        for provider_name, provider_config in providers_config.items():
            if provider_config.get("enabled", False):
                self.provider_configs[provider_name] = provider_config
                self.logger.info(f"Registered {provider_name} provider configuration")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LLM configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded LLM configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            error_msg = f"LLM configuration file not found: {self.config_path}. Please create a config file with your LLM provider settings."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error loading LLM configuration from {self.config_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _get_or_create_provider(self, provider_name: str) -> Optional[LLM]:
        """Get existing provider or create it lazily"""
        # Return existing provider if already created
        if provider_name in self.providers:
            return self.providers[provider_name]
        
        # Create provider if configuration exists
        if provider_name in self.provider_configs:
            try:
                provider_config = self.provider_configs[provider_name]
                llm = self._create_llm_provider(provider_name, provider_config)
                if llm:
                    self.providers[provider_name] = llm
                    self.logger.info(f"Lazily initialized {provider_name} provider")
                    return llm
                else:
                    self.logger.warning(f"Failed to create {provider_name} provider")
                    return None
            except Exception as e:
                self.logger.error(f"Error creating {provider_name} provider: {e}")
                return None
        
        self.logger.warning(f"Provider {provider_name} not configured or not enabled")
        return None
    
    def _create_llm_provider(self, provider_name: str, config: Dict[str, Any]) -> Optional[LLM]:
        """Create LLM provider instance"""
        try:
            if provider_name == "openai":
                return self._create_openai_provider(config)
            elif provider_name == "anthropic":
                return self._create_anthropic_provider(config)
            elif provider_name == "gemini":
                return self._create_gemini_provider(config)
            elif provider_name == "ollama":
                return self._create_ollama_provider(config)
            else:
                self.logger.warning(f"Unknown provider: {provider_name}")
                return None
        except Exception as e:
            self.logger.error(f"Error creating {provider_name} provider: {e}")
            return None
    
    def _create_openai_provider(self, config: Dict[str, Any]) -> Optional[ChatOpenAI]:
        """Create OpenAI provider"""
        api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            return None
        
        return ChatOpenAI(
            model=config.get("model", "gpt-4o-mini"),
            temperature=self.config["settings"]["temperature"],
            max_tokens=self.config["settings"]["max_tokens"],
            timeout=self.config["settings"]["timeout"]
        )
    
    def _create_anthropic_provider(self, config: Dict[str, Any]) -> Optional[ChatAnthropic]:
        """Create Anthropic provider"""
        api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            return None
        
        return ChatAnthropic(
            model=config.get("model", "claude-3-5-haiku-20241022"),
            temperature=self.config["settings"]["temperature"],
            max_tokens=self.config["settings"]["max_tokens"]
        )
    
    def _create_gemini_provider(self, config: Dict[str, Any]) -> Optional[ChatGoogleGenerativeAI]:
        """Create Google Gemini provider"""
        api_key = os.getenv(config.get("api_key_env", "GOOGLE_API_KEY"))
        if not api_key:
            return None
        
        return ChatGoogleGenerativeAI(
            model=config.get("model", "gemini-1.5-flash"),
            temperature=self.config["settings"]["temperature"],
            max_output_tokens=self.config["settings"]["max_tokens"]
        )
    
    def _create_ollama_provider(self, config: Dict[str, Any]) -> Optional[Ollama]:
        """Create Ollama provider"""
        base_url = config.get("base_url", "http://localhost:11434")
        
        return Ollama(
            model=config.get("model", "llama3.2"),
            base_url=base_url,
            temperature=self.config["settings"]["temperature"]
        )
    
    def select_provider(self, provider_name: Optional[str] = None) -> bool:
        """
        Select a specific provider or auto-detect
        
        Args:
            provider_name: Name of provider to select, or None for auto-detection
            
        Returns:
            True if provider was successfully selected
        """
        if provider_name:
            # Try to get or create the specified provider
            provider = self._get_or_create_provider(provider_name)
            if provider:
                self.selected_provider = provider_name
                self.selected_model = self.provider_configs[provider_name]["model"]
                self.logger.info(f"Selected provider: {provider_name} with model: {self.selected_model}")
                return True
            else:
                self.logger.error(f"Provider not available: {provider_name}")
                return False
        else:
            # Use default provider from config
            available_providers = list(self.provider_configs.keys())
            if not available_providers:
                self.logger.error("No LLM providers configured")
                return False
            
            # Find provider marked as default
            default_providers = [
                p for p in available_providers 
                if self.provider_configs[p].get("default", False)
            ]
            
            if default_providers:
                # Try to get or create the default provider
                provider = self._get_or_create_provider(default_providers[0])
                if provider:
                    self.selected_provider = default_providers[0]
                    self.selected_model = self.provider_configs[default_providers[0]]["model"]
                    self.logger.info(f"Auto-selected default provider: {self.selected_provider} with model: {self.selected_model}")
                    return True
                else:
                    self.logger.error(f"Failed to initialize default provider: {default_providers[0]}")
                    return False
            else:
                self.logger.error("No default provider configured. Please set 'default: true' for one provider in the config.")
                return False
    
    def get_llm_info(self) -> Dict[str, str]:
        """
        Get information about the currently selected LLM
        
        Returns:
            Dictionary with provider and model information
        """
        if not self.selected_provider:
            return {}
        
        return {
            "provider": self.selected_provider,
            "model": self.selected_model,
            "provider_config": self.provider_configs.get(self.selected_provider, {})
        }
    
    def get_llm(self) -> Optional[LLM]:
        """Get the selected LLM instance"""
        if not self.selected_provider:
            if not self.select_provider():
                return None
        
        return self.providers.get(self.selected_provider)
    
    def get_provider_name(self) -> Optional[str]:
        """Get the name of the selected provider"""
        return self.selected_provider
    
    def get_model_name(self) -> Optional[str]:
        """Get the name of the selected model"""
        return self.selected_model
    
    def get_available_providers(self) -> List[str]:
        """Get list of configured providers (not necessarily initialized)"""
        return list(self.provider_configs.keys())
    
    def get_initialized_providers(self) -> List[str]:
        """Get list of providers that have been initialized"""
        return list(self.providers.keys())
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available (configured and can be initialized)"""
        if provider_name not in self.provider_configs:
            return False
        
        # Try to get or create the provider
        provider = self._get_or_create_provider(provider_name)
        return provider is not None
    
    def configure_for_ragas(self) -> Dict[str, Any]:
        """Configure LLM for RAGAS framework"""
        llm = self.get_llm()
        if not llm:
            return {}
        
        return {
            "llm": llm,
            "provider": self.selected_provider,
            "model": self.selected_model
        }
    
    def configure_for_deepeval(self) -> Dict[str, Any]:
        """Configure LLM for DeepEval framework"""
        llm = self.get_llm()
        if not llm:
            return {}
        
        return {
            "llm": llm,
            "provider": self.selected_provider,
            "model": self.selected_model
        }
    
    def _validate_config(self):
        """Validate configuration to ensure exactly one provider is marked as default"""
        providers_config = self.config.get("providers", {})
        default_providers = []
        
        for provider_name, provider_config in providers_config.items():
            if provider_config.get("default", False):
                default_providers.append(provider_name)
        
        if len(default_providers) > 1:
            self.logger.warning(f"Multiple providers marked as default: {default_providers}. Using the first one.")
        elif len(default_providers) == 0:
            self.logger.warning("No provider marked as default. Please set 'default: true' for one provider in the config.")
        else:
            self.logger.info(f"Default provider: {default_providers[0]}") 