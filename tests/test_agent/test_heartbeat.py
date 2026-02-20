"""Tests for the heartbeat system — HeartbeatManager and AgentCore integration."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.heartbeat import HeartbeatManager
from jace.agent.core import AgentCore
from jace.agent.findings import FindingsTracker
from jace.checks.registry import CheckRegistry
from jace.config.settings import (
    HeartbeatConfig,
    LLMConfig,
    ScheduleConfig,
    Settings,
)
from jace.device.manager import DeviceManager
from jace.llm.base import Response


# ── HeartbeatConfig ──────────────────────────────────────────────────────────


def test_heartbeat_config_defaults():
    cfg = HeartbeatConfig()
    assert cfg.enabled is False
    assert cfg.interval == 1800
    assert cfg.file == "heartbeat.md"


def test_heartbeat_config_custom():
    cfg = HeartbeatConfig(enabled=True, interval=600, file="/tmp/hb.md")
    assert cfg.enabled is True
    assert cfg.interval == 600
    assert cfg.file == "/tmp/hb.md"


def test_settings_includes_heartbeat():
    s = Settings()
    assert hasattr(s, "heartbeat")
    assert isinstance(s.heartbeat, HeartbeatConfig)


# ── HeartbeatManager ─────────────────────────────────────────────────────────


@pytest.fixture
def hb_file(tmp_path: Path) -> Path:
    return tmp_path / "heartbeat.md"


@pytest.fixture
def hb_manager(hb_file: Path) -> HeartbeatManager:
    return HeartbeatManager(hb_file)


def test_load_missing_file(hb_manager: HeartbeatManager):
    result = hb_manager.load()
    assert result == ""


def test_load_existing_file(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- Check BGP peers\n- Check CPU\n")
    result = hb_manager.load()
    assert "Check BGP peers" in result
    assert "Check CPU" in result


def test_has_changed_after_external_edit(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- initial\n")
    hb_manager.load()
    assert not hb_manager.has_changed()

    # Simulate external edit (touch the file with a different mtime)
    time.sleep(0.05)
    hb_file.write_text("- modified\n")
    assert hb_manager.has_changed()


def test_has_changed_detects_deletion(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- something\n")
    hb_manager.load()
    hb_file.unlink()
    assert hb_manager.has_changed()


def test_get_instructions_auto_reloads(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- original\n")
    hb_manager.load()

    time.sleep(0.05)
    hb_file.write_text("- updated\n")

    content = hb_manager.get_instructions()
    assert "updated" in content


def test_add_instruction(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("# Heartbeat\n- Existing check\n")
    hb_manager.load()

    result = hb_manager.add_instruction("Verify NTP sync")
    assert "Verify NTP sync" in result
    assert "Existing check" in result

    # Verify persisted to disk
    disk_content = hb_file.read_text()
    assert "- Verify NTP sync" in disk_content


def test_add_instruction_preserves_dash_prefix(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("")
    hb_manager.load()
    hb_manager.add_instruction("- Already has dash")
    disk_content = hb_file.read_text()
    assert "- Already has dash" in disk_content
    # Should not double the dash
    assert "- - Already has dash" not in disk_content


def test_remove_instruction(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- First\n- Second\n- Third\n")
    hb_manager.load()

    result = hb_manager.remove_instruction(2)
    assert "Second" not in result
    assert "First" in result
    assert "Third" in result


def test_remove_instruction_invalid_index(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- Only one\n")
    hb_manager.load()

    result = hb_manager.remove_instruction(5)
    assert "Invalid index" in result


def test_remove_instruction_zero_index(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- Only one\n")
    hb_manager.load()

    result = hb_manager.remove_instruction(0)
    assert "Invalid index" in result


def test_replace_instructions(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("- Old stuff\n")
    hb_manager.load()

    result = hb_manager.replace_instructions("- Brand new\n- Also new\n")
    assert "Brand new" in result
    assert "Also new" in result
    assert "Old stuff" not in result


def test_list_instructions_numbered(hb_file: Path, hb_manager: HeartbeatManager):
    hb_file.write_text("# Header\n- Alpha\n- Beta\n")
    result = hb_manager.list_instructions()
    assert "1. Alpha" in result
    assert "2. Beta" in result


def test_list_instructions_empty(hb_manager: HeartbeatManager):
    result = hb_manager.list_instructions()
    assert "No heartbeat instructions" in result


def test_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "heartbeat.md"
    mgr = HeartbeatManager(nested)
    mgr.add_instruction("test")
    assert nested.exists()


# ── AgentCore heartbeat integration ──────────────────────────────────────────


def _make_agent(
    *,
    heartbeat_manager: HeartbeatManager | None = None,
    llm: AsyncMock | None = None,
    findings_tracker: AsyncMock | None = None,
) -> AgentCore:
    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
        heartbeat=HeartbeatConfig(enabled=True),
    )
    return AgentCore(
        settings=settings,
        llm=llm or AsyncMock(),
        device_manager=MagicMock(spec=DeviceManager),
        check_registry=AsyncMock(spec=CheckRegistry),
        findings_tracker=findings_tracker or AsyncMock(spec=FindingsTracker),
        heartbeat_manager=heartbeat_manager,
    )


@pytest.mark.asyncio
async def test_run_heartbeat_with_findings(tmp_path: Path):
    """Heartbeat cycle with LLM returning findings creates findings."""
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("- Check BGP peers\n")
    mgr = HeartbeatManager(hb_file)

    findings_json = '[{"severity": "warning", "title": "BGP peer down", "detail": "Peer 10.0.0.1 not established", "recommendation": "Check peer config"}]'
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content=findings_json, stop_reason="end_turn",
    ))

    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.add_or_update = AsyncMock(
        return_value=(MagicMock(), True),
    )
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(
        heartbeat_manager=mgr, llm=llm,
        findings_tracker=findings_tracker,
    )

    await agent._run_heartbeat()

    llm.chat.assert_called()
    findings_tracker.add_or_update.assert_called_once()
    call_kwargs = findings_tracker.add_or_update.call_args
    assert call_kwargs.kwargs["category"] == "heartbeat"
    assert call_kwargs.kwargs["device"] == "*"


@pytest.mark.asyncio
async def test_run_heartbeat_silent_when_ok(tmp_path: Path):
    """Heartbeat cycle with LLM returning [] creates no findings."""
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("- Check BGP peers\n")
    mgr = HeartbeatManager(hb_file)

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="[]", stop_reason="end_turn",
    ))

    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.add_or_update = AsyncMock()
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(
        heartbeat_manager=mgr, llm=llm,
        findings_tracker=findings_tracker,
    )

    await agent._run_heartbeat()

    llm.chat.assert_called()
    findings_tracker.add_or_update.assert_not_called()


@pytest.mark.asyncio
async def test_run_heartbeat_skips_when_no_instructions(tmp_path: Path):
    """Heartbeat with empty file should not call LLM."""
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("")
    mgr = HeartbeatManager(hb_file)

    llm = AsyncMock()
    agent = _make_agent(heartbeat_manager=mgr, llm=llm)

    await agent._run_heartbeat()

    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_heartbeat_graceful_when_file_missing(tmp_path: Path):
    """Heartbeat with missing file should not crash."""
    hb_file = tmp_path / "nonexistent.md"
    mgr = HeartbeatManager(hb_file)

    llm = AsyncMock()
    agent = _make_agent(heartbeat_manager=mgr, llm=llm)

    await agent._run_heartbeat()

    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_heartbeat_no_manager():
    """Heartbeat without manager should return immediately."""
    agent = _make_agent(heartbeat_manager=None)
    # Should not raise
    await agent._run_heartbeat()


# ── manage_heartbeat tool execution ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_manage_heartbeat_list(tmp_path: Path):
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("- Check BGP\n- Check OSPF\n")
    mgr = HeartbeatManager(hb_file)
    agent = _make_agent(heartbeat_manager=mgr)

    from jace.llm.base import ToolCall
    tc = ToolCall(id="tc1", name="manage_heartbeat", arguments={"action": "list"})
    result = await agent._execute_tool(tc)
    assert "1. Check BGP" in result
    assert "2. Check OSPF" in result


@pytest.mark.asyncio
async def test_tool_manage_heartbeat_add(tmp_path: Path):
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("- Existing\n")
    mgr = HeartbeatManager(hb_file)
    agent = _make_agent(heartbeat_manager=mgr)

    from jace.llm.base import ToolCall
    tc = ToolCall(
        id="tc2", name="manage_heartbeat",
        arguments={"action": "add", "instruction": "Check NTP"},
    )
    result = await agent._execute_tool(tc)
    assert "Check NTP" in result


@pytest.mark.asyncio
async def test_tool_manage_heartbeat_remove(tmp_path: Path):
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("- First\n- Second\n")
    mgr = HeartbeatManager(hb_file)
    agent = _make_agent(heartbeat_manager=mgr)

    from jace.llm.base import ToolCall
    tc = ToolCall(
        id="tc3", name="manage_heartbeat",
        arguments={"action": "remove", "index": 1},
    )
    result = await agent._execute_tool(tc)
    assert "First" not in result
    assert "Second" in result


@pytest.mark.asyncio
async def test_tool_manage_heartbeat_replace(tmp_path: Path):
    hb_file = tmp_path / "heartbeat.md"
    hb_file.write_text("- Old\n")
    mgr = HeartbeatManager(hb_file)
    agent = _make_agent(heartbeat_manager=mgr)

    from jace.llm.base import ToolCall
    tc = ToolCall(
        id="tc4", name="manage_heartbeat",
        arguments={"action": "replace", "instruction": "- New instruction\n"},
    )
    result = await agent._execute_tool(tc)
    assert "New instruction" in result
    assert "Old" not in result


@pytest.mark.asyncio
async def test_tool_manage_heartbeat_not_configured():
    agent = _make_agent(heartbeat_manager=None)

    from jace.llm.base import ToolCall
    tc = ToolCall(id="tc5", name="manage_heartbeat", arguments={"action": "list"})
    result = await agent._execute_tool(tc)
    assert "not configured" in result.lower()
