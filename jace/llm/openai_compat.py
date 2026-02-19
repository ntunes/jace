"""OpenAI-compatible LLM implementation (works with OpenAI, Ollama, vLLM, LiteLLM)."""

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


class OpenAICompatClient(LLMClient):
    """LLM client using the OpenAI SDK (compatible with any OpenAI API endpoint)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, **kwargs: Any):
        super().__init__(model, api_key)
        import openai
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def chat(self, messages: list[Message],
                   tools: list[ToolDefinition] | None = None,
                   system: str | None = None,
                   max_tokens: int = 4096) -> Response:
        api_messages = self._convert_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            return Response(content=f"LLM Error: {exc}", stop_reason="error")

        return self._parse_response(resp)

    def _convert_messages(self, messages: list[Message],
                          system: str | None) -> list[dict]:
        result = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.content})

            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                api_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    api_msg["content"] = msg.content
                api_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append(api_msg)

            elif msg.role == Role.TOOL:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
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
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _parse_response(self, resp: Any) -> Response:
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = {}
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }

        return Response(
            content=content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage=usage,
        )
