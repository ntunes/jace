"""JACE MCP server — exposes JACE tools to Claude Code and other MCP clients."""

from __future__ import annotations

import argparse
import os

import httpx
from mcp.server.fastmcp import FastMCP

server = FastMCP(
    name="jace",
    instructions=(
        "JACE: Autonomous Control Engine for Junos network devices. "
        "Use these tools to interact with a running JACE instance — "
        "send chat messages, inspect devices and findings, capture "
        "the TUI, read logs, switch tabs, and review conversation history."
    ),
)

_client: httpx.AsyncClient | None = None
_base_url: str = ""


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=_base_url, timeout=300.0)
    return _client


@server.tool()
async def jace_send_message(message: str) -> str:
    """Send a chat message to JACE and get the agent's response."""
    client = _get_client()
    resp = await client.post("/chat", json={"message": message})
    resp.raise_for_status()
    return resp.json()["response"]


@server.tool()
async def jace_get_devices() -> str:
    """List all managed devices and their connection status."""
    client = _get_client()
    resp = await client.get("/devices")
    resp.raise_for_status()
    import json
    return json.dumps(resp.json(), indent=2)


@server.tool()
async def jace_get_findings(
    device: str | None = None,
    severity: str | None = None,
    category: str | None = None,
    include_resolved: bool = False,
) -> str:
    """Get active findings with optional filters."""
    client = _get_client()
    params: dict[str, str] = {}
    if device:
        params["device"] = device
    if severity:
        params["severity"] = severity
    if category:
        params["category"] = category
    if include_resolved:
        params["include_resolved"] = "true"
    resp = await client.get("/findings", params=params)
    resp.raise_for_status()
    import json
    return json.dumps(resp.json(), indent=2)


@server.tool()
async def jace_get_health() -> str:
    """Get system health overview — connected devices, active/critical findings."""
    client = _get_client()
    resp = await client.get("/health")
    resp.raise_for_status()
    import json
    return json.dumps(resp.json(), indent=2)


@server.tool()
async def jace_screenshot() -> str:
    """Capture the current TUI state as SVG."""
    client = _get_client()
    resp = await client.get("/screenshot")
    resp.raise_for_status()
    return resp.json()["svg"]


@server.tool()
async def jace_get_logs(lines: int = 50) -> str:
    """Get recent application logs."""
    client = _get_client()
    resp = await client.get("/logs", params={"lines": lines})
    resp.raise_for_status()
    import json
    return json.dumps(resp.json(), indent=2)


@server.tool()
async def jace_switch_tab(tab: str) -> str:
    """Switch the active TUI tab. Valid tabs: chat, findings, logs."""
    client = _get_client()
    resp = await client.post("/tabs", json={"tab": tab})
    resp.raise_for_status()
    return f"Switched to {tab} tab"


@server.tool()
async def jace_get_chat_history(limit: int = 50) -> str:
    """Read the JACE conversation history."""
    client = _get_client()
    resp = await client.get("/chat/history", params={"limit": limit})
    resp.raise_for_status()
    import json
    return json.dumps(resp.json(), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="JACE MCP Server")
    parser.add_argument(
        "--url",
        default=os.environ.get("JACE_API_URL", "http://127.0.0.1:8080"),
        help="JACE API base URL (default: JACE_API_URL env or http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Bind port for HTTP transport (default: 9090)",
    )
    args = parser.parse_args()

    global _base_url
    _base_url = args.url.rstrip("/")

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
