"""Tests for LLM client factory."""

import pytest

from jace.config.settings import LLMConfig
from jace.llm.factory import create_llm_client


def test_factory_anthropic():
    config = LLMConfig(provider="anthropic", model="test", api_key="key")
    client = create_llm_client(config)
    from jace.llm.anthropic import AnthropicClient
    assert isinstance(client, AnthropicClient)


def test_factory_openai():
    config = LLMConfig(provider="openai", model="test", api_key="key")
    client = create_llm_client(config)
    from jace.llm.openai_compat import OpenAICompatClient
    assert isinstance(client, OpenAICompatClient)


def test_factory_unknown_raises():
    config = LLMConfig(provider="unknown", model="test", api_key="key")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_client(config)
