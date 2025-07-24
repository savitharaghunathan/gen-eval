"""
LLM initialization and configuration for GenEval framework.

This module handles LLM setup for evaluation frameworks that require LLM access.
"""

import logging
import os
from typing import Optional, Dict, Any
from openai import OpenAI
from anthropic import Anthropic


class LLMInitializer:
    """
    LLM initializer for evaluation frameworks
    """

    def __init__(self, provider: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing LLMInitializer")
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.selected_provider = None
        self.model_name = None
        
        # Initialize based on provider choice
        if provider:
            self.initialize_provider(provider)
        else:
            self.logger.info("No provider specified")

    def initialize_provider(self, provider: str):
        """
        Initialize a specific LLM provider
        
        Args:
            provider: Provider name ('openai' or 'anthropic')
        """
        provider = provider.lower()
        
        if provider == "openai":
            self._initialize_openai()
        elif provider == "anthropic":
            self._initialize_anthropic()
        else:
            self.logger.error(f"Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic")

    def _initialize_openai(self):
        """
        Initialize OpenAI client
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.selected_provider = "openai"
            self.model_name = "gpt-4o-mini"  # Set default model to GPT-4o mini
            self.logger.info(f"OpenAI client initialized successfully with model: {self.model_name}")
        else:
            self.logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _initialize_anthropic(self):
        """
        Initialize Anthropic client
        """
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=anthropic_api_key)
            self.selected_provider = "anthropic"
            self.model_name = "claude-3-5-haiku-20241022"  # Set default Anthropic model (comparable to GPT-4o-mini)
            self.logger.info(f"Anthropic client initialized successfully with model: {self.model_name}")
        else:
            self.logger.error("ANTHROPIC_API_KEY not found in environment variables")
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    def get_openai_client(self) -> Optional[OpenAI]:
        """
        Get OpenAI client if available
        """
        return self.openai_client

    def get_anthropic_client(self) -> Optional[Anthropic]:
        """
        Get Anthropic client if available
        """
        return self.anthropic_client

    def is_openai_available(self) -> bool:
        """
        Check if OpenAI is available
        """
        return self.openai_client is not None

    def is_anthropic_available(self) -> bool:
        """
        Check if Anthropic is available
        """
        return self.anthropic_client is not None

    def get_selected_provider(self) -> Optional[str]:
        """
        Get the currently selected provider
        """
        return self.selected_provider



    def configure_ragas_llm(self) -> Dict[str, Any]:
        """
        Configure LLM for RAGAS evaluation
        """
        config = {}
        
        if self.selected_provider == "openai" and self.is_openai_available():
            config["llm"] = self.openai_client
            config["provider"] = "openai"
            config["model"] = self.model_name
            self.logger.info(f"Configured RAGAS with OpenAI LLM using model: {self.model_name}")
        elif self.selected_provider == "anthropic" and self.is_anthropic_available():
            config["llm"] = self.anthropic_client
            config["provider"] = "anthropic"
            config["model"] = self.model_name
            self.logger.info(f"Configured RAGAS with Anthropic LLM using model: {self.model_name}")
        else:
            self.logger.error(f"No LLM provider available for RAGAS. Selected provider: {self.selected_provider}")
            
        return config

    def configure_deepeval_llm(self) -> Dict[str, Any]:
        """
        Configure LLM for DeepEval evaluation
        """
        config = {}
        
        if self.selected_provider == "openai" and self.is_openai_available():
            config["llm"] = self.openai_client
            config["provider"] = "openai"
            config["model"] = self.model_name
            self.logger.info(f"Configured DeepEval with OpenAI LLM using model: {self.model_name}")
        elif self.selected_provider == "anthropic" and self.is_anthropic_available():
            config["llm"] = self.anthropic_client
            config["provider"] = "anthropic"
            config["model"] = self.model_name
            self.logger.info(f"Configured DeepEval with Anthropic LLM using model: {self.model_name}")
        else:
            self.logger.error(f"No LLM provider available for DeepEval. Selected provider: {self.selected_provider}")
            
        return config 