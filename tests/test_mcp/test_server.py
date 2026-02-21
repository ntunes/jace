"""Tests for jace.mcp.server — MCP tool functions with mocked httpx."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from jace.mcp.server import (
    jace_get_chat_history,
    jace_get_devices,
    jace_get_findings,
    jace_get_health,
    jace_get_logs,
    jace_screenshot,
    jace_send_message,
    jace_switch_tab,
)


class FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, data: dict | list | str, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> dict | list:
        if isinstance(self._data, str):
            return json.loads(self._data)
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _mock_client() -> AsyncMock:
    client = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_jace_send_message():
    client = _mock_client()
    client.post.return_value = FakeResponse({"response": "Hello from JACE"})

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_send_message("hello")

    assert result == "Hello from JACE"
    client.post.assert_called_once_with("/chat", json={"message": "hello"})


@pytest.mark.asyncio
async def test_jace_get_devices():
    devices = [{"name": "r1", "host": "10.0.0.1", "status": "connected"}]
    client = _mock_client()
    client.get.return_value = FakeResponse(devices)

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_get_devices()

    assert json.loads(result) == devices
    client.get.assert_called_once_with("/devices")


@pytest.mark.asyncio
async def test_jace_get_findings_no_filters():
    findings = [{"title": "High temp", "severity": "warning"}]
    client = _mock_client()
    client.get.return_value = FakeResponse(findings)

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_get_findings()

    assert json.loads(result) == findings
    client.get.assert_called_once_with("/findings", params={})


@pytest.mark.asyncio
async def test_jace_get_findings_with_filters():
    client = _mock_client()
    client.get.return_value = FakeResponse([])

    with patch("jace.mcp.server._get_client", return_value=client):
        await jace_get_findings(device="r1", severity="critical", include_resolved=True)

    client.get.assert_called_once_with(
        "/findings",
        params={"device": "r1", "severity": "critical", "include_resolved": "true"},
    )


@pytest.mark.asyncio
async def test_jace_get_health():
    health = {"status": "ok", "devices_connected": 2}
    client = _mock_client()
    client.get.return_value = FakeResponse(health)

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_get_health()

    assert json.loads(result) == health
    client.get.assert_called_once_with("/health")


@pytest.mark.asyncio
async def test_jace_screenshot():
    client = _mock_client()
    client.get.return_value = FakeResponse({"svg": "<svg>...</svg>"})

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_screenshot()

    assert result == "<svg>...</svg>"
    client.get.assert_called_once_with("/screenshot")


@pytest.mark.asyncio
async def test_jace_get_logs():
    logs = [{"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "message": "ok"}]
    client = _mock_client()
    client.get.return_value = FakeResponse(logs)

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_get_logs(lines=10)

    assert json.loads(result) == logs
    client.get.assert_called_once_with("/logs", params={"lines": 10})


@pytest.mark.asyncio
async def test_jace_switch_tab():
    client = _mock_client()
    client.post.return_value = FakeResponse({"tab": "findings"})

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_switch_tab("findings")

    assert "findings" in result
    client.post.assert_called_once_with("/tabs", json={"tab": "findings"})


@pytest.mark.asyncio
async def test_jace_get_chat_history():
    history = [
        {"role": "user", "content": "show devices"},
        {"role": "assistant", "content": "Here are your devices..."},
    ]
    client = _mock_client()
    client.get.return_value = FakeResponse(history)

    with patch("jace.mcp.server._get_client", return_value=client):
        result = await jace_get_chat_history(limit=10)

    assert json.loads(result) == history
    client.get.assert_called_once_with("/chat/history", params={"limit": 10})
