"""Anthropic/Claude LLM implementation with native tool use."""

from __future__ import annotations

import json
import logging
from typing import Any

from jace.llm.base import (
    LLMClient,
    Message,
    Response,
    Role,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """LLM client using the Anthropic SDK."""

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key)
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, messages: list[Message],
                   tools: list[ToolDefinition] | None = None,
                   system: str | None = None,
                   max_tokens: int = 4096) -> Response:
        api_messages = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:
            logger.error("Anthropic API error: %s", exc)
            return Response(content=f"LLM Error: {exc}", stop_reason="error")

        return self._parse_response(resp)

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # system is passed separately to Anthropic

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content})

            elif msg.role == Role.TOOL:
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })

            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _parse_response(self, resp: Any) -> Response:
        content_parts = []
        tool_calls = []

        for block in resp.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input) if block.input else {},
                ))

        return Response(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
