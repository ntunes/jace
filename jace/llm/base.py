"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Response:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient(ABC):
    """Abstract interface for LLM backends."""

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    async def chat(self, messages: list[Message],
                   tools: list[ToolDefinition] | None = None,
                   system: str | None = None,
                   max_tokens: int = 4096) -> Response:
        """Send messages to the LLM and get a response.

        The response may contain text content, tool calls, or both.
        The caller (agent core) handles the tool execution loop.
        """
