"""Tests for LLM logging wrapper."""

from __future__ import annotations

import json

import pytest

from jace.config.settings import LLMConfig
from jace.llm.base import (
    LLMClient,
    Message,
    Response,
    Role,
    ToolCall,
    ToolDefinition,
)
from jace.llm.logging import LoggingLLMClient


class FakeLLMClient(LLMClient):
    """Fake client that returns a canned response."""

    def __init__(self, response: Response):
        self._response = response

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> Response:
        return self._response


SAMPLE_MESSAGES = [
    Message(role=Role.USER, content="Show BGP summary"),
    Message(role=Role.ASSISTANT, content="I'll check that for you."),
    Message(role=Role.TOOL, content='{"bgp": "ok"}', tool_call_id="abc123"),
]

SAMPLE_TOOLS = [
    ToolDefinition(name="run_command", description="Run a CLI command", parameters={}),
    ToolDefinition(name="get_config", description="Get device config", parameters={}),
]

SAMPLE_RESPONSE = Response(
    content="Running BGP summary now.",
    tool_calls=[
        ToolCall(id="call_1", name="run_command", arguments={"command": "show bgp summary"}),
    ],
    stop_reason="tool_use",
    usage={"input_tokens": 1234, "output_tokens": 256},
)


@pytest.fixture
def log_file(tmp_path):
    return str(tmp_path / "llm.log")


@pytest.mark.asyncio
async def test_text_format_contains_expected_sections(log_file):
    client = FakeLLMClient(SAMPLE_RESPONSE)
    wrapper = LoggingLLMClient(client, log_file, "text")

    await wrapper.chat(SAMPLE_MESSAGES, tools=SAMPLE_TOOLS, system="You are a network engineer")

    with open(log_file) as f:
        content = f.read()

    assert "REQUEST" in content
    assert "RESPONSE" in content
    assert "You are a network engineer" in content
    assert "[user]" in content
    assert "Show BGP summary" in content
    assert "[assistant]" in content
    assert "[tool]" in content
    assert "run_command, get_config" in content
    assert "Running BGP summary now." in content
    assert "--- Tool Calls ---" in content
    assert 'run_command(command="show bgp summary")' in content
    assert "--- Usage ---" in content
    assert "Input: 1234 | Output: 256" in content


@pytest.mark.asyncio
async def test_jsonl_format_produces_valid_json(log_file):
    client = FakeLLMClient(SAMPLE_RESPONSE)
    wrapper = LoggingLLMClient(client, log_file, "jsonl")

    await wrapper.chat(SAMPLE_MESSAGES, tools=SAMPLE_TOOLS, system="test system")

    with open(log_file) as f:
        lines = f.read().strip().split("\n")

    assert len(lines) == 1
    entry = json.loads(lines[0])

    assert "timestamp" in entry
    assert entry["request"]["system"] == "test system"
    assert len(entry["request"]["messages"]) == 3
    assert entry["request"]["tools"] == ["run_command", "get_config"]
    assert entry["response"]["content"] == "Running BGP summary now."
    assert entry["response"]["stop_reason"] == "tool_use"
    assert entry["response"]["tool_calls"][0]["name"] == "run_command"
    assert entry["response"]["usage"]["input_tokens"] == 1234


@pytest.mark.asyncio
async def test_wrapper_delegates_and_returns_response(log_file):
    client = FakeLLMClient(SAMPLE_RESPONSE)
    wrapper = LoggingLLMClient(client, log_file, "text")

    response = await wrapper.chat(SAMPLE_MESSAGES)

    assert response is SAMPLE_RESPONSE
    assert response.content == "Running BGP summary now."
    assert response.tool_calls[0].name == "run_command"


@pytest.mark.asyncio
async def test_appends_multiple_exchanges(log_file):
    client = FakeLLMClient(SAMPLE_RESPONSE)
    wrapper = LoggingLLMClient(client, log_file, "jsonl")

    await wrapper.chat(SAMPLE_MESSAGES)
    await wrapper.chat(SAMPLE_MESSAGES)

    with open(log_file) as f:
        lines = f.read().strip().split("\n")

    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_factory_no_wrap_without_log_file():
    config = LLMConfig(provider="anthropic", model="test", api_key="key")
    from jace.llm.factory import create_llm_client

    client = create_llm_client(config)
    assert not isinstance(client, LoggingLLMClient)


def test_factory_wraps_with_log_file(tmp_path):
    config = LLMConfig(
        provider="anthropic",
        model="test",
        api_key="key",
        log_file=str(tmp_path / "llm.log"),
    )
    from jace.llm.factory import create_llm_client

    client = create_llm_client(config)
    assert isinstance(client, LoggingLLMClient)
