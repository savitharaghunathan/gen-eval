"""
LLM Manager for GenEval framework.

This module provides configuration management for multiple LLM providers.
It serves as a configuration store that adapters can use to create their own LLM instances.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List


class LLMManager:
    """Manages LLM provider configurations"""
    
    def __init__(self, config_path: str):
        """
        Initialize LLM manager
        
        Args:
            config_path: Path to YAML configuration file (required)
        """
        if not config_path:
            raise ValueError("config_path is required. Please specify the path to your LLM configuration file.")
        
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        self._validate_config()
        
        # Store provider configurations
        self.provider_configs = {}
        
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
            error_msg = f"LLM configuration file not found: {self.config_path}. Please check the path and ensure the file exists."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error loading LLM configuration from {self.config_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
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
    
    def get_default_provider(self) -> Optional[str]:
        """Get the name of the default provider"""
        providers_config = self.config.get("providers", {})
        for provider_name, provider_config in providers_config.items():
            if provider_config.get("default", False):
                return provider_name
        return None
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        return self.provider_configs.get(provider_name, {})
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings"""
        return self.config.get('settings', {})
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider from environment variables"""
        provider_config = self.provider_configs.get(provider_name, {})
        api_key_env = provider_config.get("api_key_env")
        
        if api_key_env:
            return os.getenv(api_key_env)
        
        # Fallback to common environment variable names
        if provider_name == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider_name == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        elif provider_name == "gemini":
            return os.getenv("GOOGLE_API_KEY")
        elif provider_name == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY")
        
        return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of configured providers"""
        return list(self.provider_configs.keys())
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available (configured and enabled)"""
        return provider_name in self.provider_configs
    
    def get_deepeval_config(self, provider_name: str) -> Dict[str, Any]:
        """Get DeepEval-compatible configuration for a provider"""
        provider_config = self.provider_configs.get(provider_name, {})
        global_settings = self.config.get('settings', {})
        
        # Base configuration
        deepeval_config = {
            "model": provider_config.get("model"),
            "temperature": global_settings.get("temperature", 0.1),
        }
        
        # Add provider-specific configurations
        if provider_name == "openai":
            # OpenAI uses environment variables, no additional config needed
            pass
        elif provider_name == "azure_openai":
            deepeval_config.update({
                "deployment_name": provider_config.get("deployment_name"),
                "azure_openai_api_key": provider_config.get("azure_openai_api_key"),
                "openai_api_version": provider_config.get("openai_api_version", "2025-01-01-preview"),
                "azure_endpoint": provider_config.get("azure_endpoint"),
            })
        elif provider_name == "anthropic":
            # Anthropic uses environment variables, no additional config needed
            pass
        elif provider_name == "amazon_bedrock":
            deepeval_config.update({
                "region_name": provider_config.get("region_name", "us-east-1"),
            })
        elif provider_name == "gemini":
            deepeval_config.update({
                "api_key": self.get_api_key(provider_name),
            })
        elif provider_name == "deepseek":
            deepeval_config.update({
                "api_key": self.get_api_key(provider_name),
            })
        elif provider_name == "ollama":
            deepeval_config.update({
                "base_url": provider_config.get("base_url", "http://localhost:11434"),
            })
        
        return deepeval_config 