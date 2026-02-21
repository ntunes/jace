"""Tests for MCPManager — tool discovery, routing, collisions, and cleanup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.config.settings import MCPServerConfig
from jace.mcp.manager import MCPManager


# --- helpers ---------------------------------------------------------------

@dataclass
class FakeTool:
    name: str
    description: str | None = "A tool"
    inputSchema: dict | None = None


@dataclass
class FakeListResult:
    tools: list[FakeTool]


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeCallResult:
    content: list[FakeTextBlock]


def _stdio_config(name: str = "test-server") -> MCPServerConfig:
    return MCPServerConfig(
        name=name, transport="stdio", command="echo", args=["hello"],
    )


def _sse_config(name: str = "sse-server") -> MCPServerConfig:
    return MCPServerConfig(
        name=name, transport="sse", url="http://localhost:8080/sse",
    )


def _http_config(name: str = "http-server") -> MCPServerConfig:
    return MCPServerConfig(
        name=name, transport="streamable-http",
        url="http://localhost:8080/mcp",
    )


def _patch_transports():
    """Patch all three MCP transport context managers."""
    mock_read = MagicMock()
    mock_write = MagicMock()
    transport_pair = (mock_read, mock_write)

    # Create an async context manager that yields the transport pair
    class FakeTransportCM:
        async def __aenter__(self):
            return transport_pair
        async def __aexit__(self, *args):
            pass

    class FakeSessionCM:
        def __init__(self, read, write):
            self.session = AsyncMock()
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, *args):
            pass

    patches = [
        patch("jace.mcp.manager.stdio_client", return_value=FakeTransportCM()),
        patch("jace.mcp.manager.sse_client", return_value=FakeTransportCM()),
        patch("jace.mcp.manager.streamablehttp_client", return_value=FakeTransportCM()),
        patch("jace.mcp.manager.ClientSession", side_effect=lambda r, w: FakeSessionCM(r, w)),
    ]
    return patches, FakeSessionCM


# --- tests -----------------------------------------------------------------

@pytest.fixture
def _mock_mcp(monkeypatch):
    """Fixture that patches transports and returns a session factory.

    The session's ``list_tools`` and ``call_tool`` can be configured per-test
    by setting attributes on the returned mock session before calling
    ``connect_all``.
    """
    mock_session = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=FakeListResult(tools=[]))
    mock_session.initialize = AsyncMock()

    class _FakeTransportCM:
        async def __aenter__(self):
            return (MagicMock(), MagicMock())
        async def __aexit__(self, *args):
            pass

    class _FakeSessionCM:
        async def __aenter__(self_inner):
            return mock_session
        async def __aexit__(self_inner, *args):
            pass

    monkeypatch.setattr("jace.mcp.manager.stdio_client",
                        lambda *a, **kw: _FakeTransportCM())
    monkeypatch.setattr("jace.mcp.manager.sse_client",
                        lambda *a, **kw: _FakeTransportCM())
    monkeypatch.setattr("jace.mcp.manager.streamablehttp_client",
                        lambda *a, **kw: _FakeTransportCM())
    monkeypatch.setattr("jace.mcp.manager.ClientSession",
                        lambda r, w: _FakeSessionCM())

    return mock_session


# -- Tool discovery ---------------------------------------------------------

async def test_discover_tools(_mock_mcp):
    """Tools from a server are converted to ToolDefinition list."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="get_weather", description="Get weather"),
        FakeTool(name="get_forecast", description="Get forecast"),
    ])

    mgr = MCPManager([_stdio_config()])
    await mgr.connect_all()

    assert len(mgr.tools) == 2
    names = {t.name for t in mgr.tools}
    assert names == {"get_weather", "get_forecast"}
    # Descriptions are prefixed with server name
    assert mgr.tools[0].description.startswith("[test-server]")


async def test_tool_description_prefix(_mock_mcp):
    """Tool description is prefixed with [server_name]."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="greet", description="Say hello"),
    ])

    mgr = MCPManager([_stdio_config("my-srv")])
    await mgr.connect_all()

    assert mgr.tools[0].description == "[my-srv] Say hello"


async def test_tool_schema_passthrough(_mock_mcp):
    """inputSchema from the MCP tool is passed through as parameters."""
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="weather", description="wx", inputSchema=schema),
    ])

    mgr = MCPManager([_stdio_config()])
    await mgr.connect_all()

    assert mgr.tools[0].parameters == schema


# -- Tool call routing ------------------------------------------------------

async def test_call_tool_routes_to_session(_mock_mcp):
    """call_tool routes the request to the correct server session."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="greet"),
    ])
    _mock_mcp.call_tool.return_value = FakeCallResult(
        content=[FakeTextBlock(text="Hello!")],
    )

    mgr = MCPManager([_stdio_config()])
    await mgr.connect_all()

    result = await mgr.call_tool("greet", {"name": "world"})
    assert result == "Hello!"
    _mock_mcp.call_tool.assert_awaited_once_with("greet", {"name": "world"})


async def test_call_unknown_tool(_mock_mcp):
    """Calling an unregistered tool returns an error string."""
    mgr = MCPManager([_stdio_config()])
    await mgr.connect_all()

    result = await mgr.call_tool("nonexistent", {})
    assert "Unknown MCP tool" in result


async def test_has_tool(_mock_mcp):
    """has_tool returns True only for registered MCP tools."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="my_tool"),
    ])

    mgr = MCPManager([_stdio_config()])
    await mgr.connect_all()

    assert mgr.has_tool("my_tool")
    assert not mgr.has_tool("other_tool")


# -- Collision detection ----------------------------------------------------

async def test_collision_with_builtin(_mock_mcp, caplog):
    """Tools that collide with built-in names are skipped with a warning."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="run_command"),  # collides with built-in
        FakeTool(name="safe_tool"),
    ])

    mgr = MCPManager([_stdio_config()])
    with caplog.at_level(logging.WARNING):
        await mgr.connect_all(builtin_names={"run_command"})

    assert len(mgr.tools) == 1
    assert mgr.tools[0].name == "safe_tool"
    assert "collides" in caplog.text


async def test_collision_between_servers(_mock_mcp, caplog):
    """When two MCP servers expose the same tool name, second is skipped."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="shared_tool"),
    ])

    mgr = MCPManager([_stdio_config("srv-a"), _stdio_config("srv-b")])
    with caplog.at_level(logging.WARNING):
        await mgr.connect_all()

    # Only the first server's tool should be registered
    assert len(mgr.tools) == 1
    assert "collides" in caplog.text


# -- Failed server connection -----------------------------------------------

async def test_failed_server_skipped(monkeypatch, caplog):
    """A server that fails to connect is logged and skipped."""
    def _raise(*args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr("jace.mcp.manager.stdio_client", _raise)

    mgr = MCPManager([_stdio_config()])
    with caplog.at_level(logging.ERROR):
        await mgr.connect_all()

    assert len(mgr.tools) == 0
    assert "Failed to connect" in caplog.text


# -- Cleanup ----------------------------------------------------------------

async def test_close_clears_state(_mock_mcp):
    """close() clears all internal state."""
    _mock_mcp.list_tools.return_value = FakeListResult(tools=[
        FakeTool(name="tool_a"),
    ])

    mgr = MCPManager([_stdio_config()])
    await mgr.connect_all()
    assert len(mgr.tools) == 1

    await mgr.close()
    assert len(mgr.tools) == 0
    assert not mgr.has_tool("tool_a")
