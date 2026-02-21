"""MCP (Model Context Protocol) server manager."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from jace.config.settings import MCPServerConfig
from jace.llm.base import ToolDefinition

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages connections to one or more MCP tool servers."""

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = configs
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[ToolDefinition] = []
        self._tool_map: dict[str, str] = {}  # tool_name → server_name

    @property
    def tools(self) -> list[ToolDefinition]:
        """Return cached tool definitions from all connected servers."""
        return self._tools

    async def connect_all(
        self, *, builtin_names: set[str] | None = None,
    ) -> None:
        """Connect to every configured MCP server and discover tools.

        *builtin_names* is the set of built-in tool names so we can detect
        collisions.  Failures are logged and skipped per-server.
        """
        reserved: set[str] = builtin_names or set()

        for cfg in self._configs:
            try:
                session = await self._connect_one(cfg)
                self._sessions[cfg.name] = session

                result = await session.list_tools()
                for tool in result.tools:
                    if tool.name in reserved or tool.name in self._tool_map:
                        source = "built-in" if tool.name in reserved else (
                            f"server '{self._tool_map[tool.name]}'"
                        )
                        logger.warning(
                            "MCP tool '%s' from server '%s' collides with "
                            "%s — skipping",
                            tool.name, cfg.name, source,
                        )
                        continue

                    description = f"[{cfg.name}] {tool.description or ''}"
                    self._tools.append(ToolDefinition(
                        name=tool.name,
                        description=description,
                        parameters=tool.inputSchema or {
                            "type": "object", "properties": {},
                        },
                    ))
                    self._tool_map[tool.name] = cfg.name

                logger.info(
                    "MCP server '%s': %d tool(s) registered",
                    cfg.name, sum(
                        1 for v in self._tool_map.values() if v == cfg.name
                    ),
                )
            except Exception as exc:
                logger.error(
                    "Failed to connect to MCP server '%s': %s", cfg.name, exc,
                )

    async def _connect_one(self, cfg: MCPServerConfig) -> ClientSession:
        """Establish a connection to a single MCP server."""
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(
                    f"MCP server '{cfg.name}': 'command' is required "
                    f"for stdio transport",
                )
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
            )
            transport = await self._exit_stack.enter_async_context(
                stdio_client(params),
            )
        elif cfg.transport == "sse":
            if not cfg.url:
                raise ValueError(
                    f"MCP server '{cfg.name}': 'url' is required "
                    f"for sse transport",
                )
            transport = await self._exit_stack.enter_async_context(
                sse_client(cfg.url, headers=cfg.headers or {}),
            )
        elif cfg.transport == "streamable-http":
            if not cfg.url:
                raise ValueError(
                    f"MCP server '{cfg.name}': 'url' is required "
                    f"for streamable-http transport",
                )
            transport = await self._exit_stack.enter_async_context(
                streamablehttp_client(cfg.url, headers=cfg.headers or {}),
            )
        else:
            raise ValueError(
                f"MCP server '{cfg.name}': unknown transport '{cfg.transport}'",
            )

        read_stream, write_stream = transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream),
        )
        await session.initialize()
        return session

    def has_tool(self, name: str) -> bool:
        """Check whether *name* is an MCP tool."""
        return name in self._tool_map

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the correct MCP server session."""
        server_name = self._tool_map.get(name)
        if server_name is None:
            return f"Unknown MCP tool: {name}"

        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP server '{server_name}' is not connected."

        result = await session.call_tool(name, arguments)
        # Concatenate all text content blocks
        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else "(no output)"

    async def close(self) -> None:
        """Tear down all MCP connections."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._tool_map.clear()
