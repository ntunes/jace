"""Logging wrapper for LLM clients."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jace.llm.base import (
    LLMClient,
    Message,
    Response,
    ToolCall,
    ToolDefinition,
)

SEPARATOR = "\u2550" * 64


class LoggingLLMClient(LLMClient):
    """Wrapper that logs all LLM requests and responses to a file."""

    def __init__(self, client: LLMClient, log_file: str, log_format: str = "text"):
        self._client = client
        self._log_path = Path(log_file).expanduser()
        self._log_format = log_format
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> Response:
        response = await self._client.chat(
            messages, tools=tools, system=system, max_tokens=max_tokens
        )
        self._log(messages, tools, system, response)
        return response

    def _log(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        system: str | None,
        response: Response,
    ) -> None:
        if self._log_format == "jsonl":
            self._log_jsonl(messages, tools, system, response)
        else:
            self._log_text(messages, tools, system, response)

    def _log_text(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        system: str | None,
        response: Response,
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = []

        # --- Request ---
        lines.append(SEPARATOR)
        lines.append(f"[{now}] REQUEST")
        lines.append(SEPARATOR)
        lines.append("")

        if system:
            lines.append(f"System: {system}")
            lines.append("")

        lines.append(f"--- Messages ({len(messages)}) ---")
        lines.append("")
        for msg in messages:
            if msg.tool_call_id:
                lines.append(f"[{msg.role.value}] (call_{msg.tool_call_id})")
            else:
                lines.append(f"[{msg.role.value}]")
            lines.append(msg.content)
            lines.append("")

        if tools:
            tool_names = ", ".join(t.name for t in tools)
            lines.append("--- Tools ---")
            lines.append(tool_names)
            lines.append("")

        # --- Response ---
        lines.append(SEPARATOR)
        lines.append(f"[{now}] RESPONSE ({response.stop_reason})")
        lines.append(SEPARATOR)
        lines.append("")

        if response.content:
            lines.append(response.content)
            lines.append("")

        if response.tool_calls:
            lines.append("--- Tool Calls ---")
            for tc in response.tool_calls:
                args = ", ".join(f'{k}="{v}"' for k, v in tc.arguments.items())
                lines.append(f"{tc.name}({args})")
            lines.append("")

        if response.usage:
            input_tokens = response.usage.get("input_tokens", 0)
            output_tokens = response.usage.get("output_tokens", 0)
            lines.append("--- Usage ---")
            lines.append(f"Input: {input_tokens} | Output: {output_tokens}")
            lines.append("")

        with open(self._log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    def _log_jsonl(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        system: str | None,
        response: Response,
    ) -> None:
        def _serialize_tool_call(tc: ToolCall) -> dict[str, Any]:
            return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}

        def _serialize_message(msg: Message) -> dict[str, Any]:
            d: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                d["tool_calls"] = [_serialize_tool_call(tc) for tc in msg.tool_calls]
            return d

        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "request": {
                "system": system,
                "messages": [_serialize_message(m) for m in messages],
                "tools": [t.name for t in tools] if tools else None,
            },
            "response": {
                "content": response.content,
                "tool_calls": [_serialize_tool_call(tc) for tc in response.tool_calls],
                "stop_reason": response.stop_reason,
                "usage": response.usage,
            },
        }

        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
