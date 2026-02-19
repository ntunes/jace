"""Factory for creating LLM clients based on configuration."""

from __future__ import annotations

from jace.config.settings import LLMConfig
from jace.llm.base import LLMClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on the provider configuration."""
    if config.provider == "anthropic":
        from jace.llm.anthropic import AnthropicClient
        return AnthropicClient(model=config.model, api_key=config.api_key)
    elif config.provider == "openai":
        from jace.llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
