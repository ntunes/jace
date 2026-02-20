"""Tests for run_shell tool — blocklist, approval, execution, and timeout."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jace.agent.core import AgentCore, SHELL_BLOCKED_PATTERNS
from jace.agent.findings import FindingsTracker
from jace.checks.registry import CheckRegistry
from jace.config.settings import LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.llm.base import ToolCall


def _make_agent(*, approval_callback=None) -> AgentCore:
    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    agent = AgentCore(
        settings=settings,
        llm=AsyncMock(),
        device_manager=MagicMock(spec=DeviceManager),
        check_registry=AsyncMock(spec=CheckRegistry),
        findings_tracker=AsyncMock(spec=FindingsTracker),
    )
    if approval_callback is not None:
        agent.set_approval_callback(approval_callback)
    return agent


def _shell_call(command: str, reason: str = "testing") -> ToolCall:
    return ToolCall(
        id="tc1", name="run_shell",
        arguments={"command": command, "reason": reason},
    )


# ---------- Blocklist tests ----------


class TestBlocklist:
    """Dangerous commands are rejected outright."""

    @pytest.mark.parametrize("cmd", [
        "sudo reboot",
        "rm -rf /",
        "reboot",
        "shutdown -h now",
        "kill -9 1234",
        "killall python",
        "pkill nginx",
        "docker rm container",
        "kubectl delete pod foo",
        "brew install something",
        "pip install malware",
        "npm install evil",
        "apt install pkg",
        "dd if=/dev/zero of=/dev/sda",
        "chmod 777 /etc/passwd",
        "chown root:root /etc/shadow",
        "mv /etc/passwd /tmp",
        "cp /dev/zero /dev/sda",
        "systemctl stop sshd",
        "service sshd stop",
        "mkfs.ext4 /dev/sda1",
        "halt",
        "poweroff",
        "yum install pkg",
    ])
    def test_blocked_commands(self, cmd: str) -> None:
        result = AgentCore._is_shell_blocked(cmd)
        assert result is not None
        assert "Blocked" in result

    @pytest.mark.parametrize("cmd", [
        "ping -c 4 10.0.0.1",
        "traceroute 10.0.0.1",
        "dig example.com",
        "nslookup example.com",
        "cat ~/.ssh/config",
        "head -20 /etc/hosts",
        "ip route show",
        "ifconfig",
        "netstat -rn",
        "ss -tlnp",
        "curl -s http://example.com",
        "nc -zv 10.0.0.1 22",
        "ssh -v user@host exit",
        "host example.com",
        "mtr --report 10.0.0.1",
        "arp -a",
        "route -n",
    ])
    def test_allowed_commands(self, cmd: str) -> None:
        result = AgentCore._is_shell_blocked(cmd)
        assert result is None

    def test_blocked_in_pipe(self) -> None:
        """Blocked patterns are detected even in piped commands."""
        result = AgentCore._is_shell_blocked("echo test | sudo tee /etc/hosts")
        assert result is not None
        assert "sudo" in result

    def test_blocked_with_path_prefix(self) -> None:
        """Blocked patterns are detected even with full paths."""
        result = AgentCore._is_shell_blocked("/usr/bin/sudo whoami")
        assert result is not None
        assert "sudo" in result


# ---------- Approval tests ----------


@pytest.mark.asyncio
async def test_denied_approval_returns_denial() -> None:
    """When user denies, return denial message."""
    callback = AsyncMock(return_value=False)
    agent = _make_agent(approval_callback=callback)

    result = await agent._execute_tool(_shell_call("ping -c 1 10.0.0.1"))

    callback.assert_awaited_once_with("ping -c 1 10.0.0.1", "testing")
    assert "denied" in result.lower()


@pytest.mark.asyncio
async def test_missing_approval_callback_returns_error() -> None:
    """Without an approval callback (headless mode), shell is unavailable."""
    agent = _make_agent()  # no callback

    result = await agent._execute_tool(_shell_call("ping -c 1 10.0.0.1"))

    assert "interactive session" in result.lower()


# ---------- Execution tests ----------


@pytest.mark.asyncio
async def test_approved_command_executes() -> None:
    """Approved command runs and returns output."""
    callback = AsyncMock(return_value=True)
    agent = _make_agent(approval_callback=callback)

    result = await agent._execute_tool(_shell_call("echo hello"))

    assert "hello" in result


@pytest.mark.asyncio
async def test_command_stderr_included() -> None:
    """stderr output is included in the result."""
    callback = AsyncMock(return_value=True)
    agent = _make_agent(approval_callback=callback)

    result = await agent._execute_tool(
        _shell_call("echo error >&2")
    )

    assert "error" in result


# ---------- Timeout test ----------


@pytest.mark.asyncio
async def test_command_timeout() -> None:
    """Commands that exceed timeout return a timeout error."""
    callback = AsyncMock(return_value=True)
    agent = _make_agent(approval_callback=callback)

    # Patch the timeout to 0.1s for fast testing
    import jace.agent.core as core_mod
    original = core_mod.SHELL_COMMAND_TIMEOUT
    core_mod.SHELL_COMMAND_TIMEOUT = 0.1
    try:
        result = await agent._execute_tool(_shell_call("sleep 10"))
    finally:
        core_mod.SHELL_COMMAND_TIMEOUT = original

    assert "timed out" in result.lower()


# ---------- Blocklist doesn't call approval ----------


@pytest.mark.asyncio
async def test_blocked_command_skips_approval() -> None:
    """Blocked commands never reach the approval callback."""
    callback = AsyncMock(return_value=True)
    agent = _make_agent(approval_callback=callback)

    result = await agent._execute_tool(_shell_call("sudo whoami"))

    callback.assert_not_awaited()
    assert "Blocked" in result
