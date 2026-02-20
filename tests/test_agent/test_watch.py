"""Tests for the WatchManager — lightweight metric watches."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from jace.agent.metrics_store import MetricPoint, MetricsStore
from jace.agent.watch import Watch, WatchManager
from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


def _make_watch(**overrides) -> Watch:
    defaults = {
        "id": "",
        "device": "r1",
        "command": "show interfaces ge-0/0/0 extensive",
        "metric_name": "ge0_crc_errors",
        "interval": 60,
        "parse_pattern": r"CRC Errors\s+(?P<value>\d+)",
        "unit": "errors",
    }
    defaults.update(overrides)
    return Watch(**defaults)


def _make_manager(
    *,
    device_manager: MagicMock | None = None,
    metrics_store: AsyncMock | None = None,
) -> WatchManager:
    return WatchManager(
        device_manager=device_manager or MagicMock(spec=DeviceManager),
        metrics_store=metrics_store or AsyncMock(spec=MetricsStore),
    )


# ---------- add / remove / list ----------


@pytest.mark.asyncio
async def test_add_returns_deterministic_id():
    mgr = _make_manager()
    watch = _make_watch()
    watch_id = mgr.add(watch)
    assert isinstance(watch_id, str)
    assert len(watch_id) == 12
    mgr.stop_all()


@pytest.mark.asyncio
async def test_add_idempotent():
    """Adding the same (device, command, metric_name) twice returns the same id."""
    mgr = _make_manager()
    id1 = mgr.add(_make_watch())
    id2 = mgr.add(_make_watch())
    assert id1 == id2
    assert len(mgr.list_watches()) == 1
    mgr.stop_all()


@pytest.mark.asyncio
async def test_add_different_metric_creates_separate_watch():
    mgr = _make_manager()
    id1 = mgr.add(_make_watch(metric_name="crc_errors"))
    id2 = mgr.add(_make_watch(metric_name="input_errors"))
    assert id1 != id2
    assert len(mgr.list_watches()) == 2
    mgr.stop_all()


@pytest.mark.asyncio
async def test_remove_existing():
    mgr = _make_manager()
    watch_id = mgr.add(_make_watch())
    assert mgr.remove(watch_id) is True
    assert len(mgr.list_watches()) == 0


def test_remove_nonexistent():
    mgr = _make_manager()
    assert mgr.remove("nonexistent") is False


def test_list_watches_empty():
    mgr = _make_manager()
    assert mgr.list_watches() == []


@pytest.mark.asyncio
async def test_list_watches_populated():
    mgr = _make_manager()
    mgr.add(_make_watch(metric_name="m1"))
    mgr.add(_make_watch(metric_name="m2"))
    watches = mgr.list_watches()
    assert len(watches) == 2
    names = {w.metric_name for w in watches}
    assert names == {"m1", "m2"}
    mgr.stop_all()


# ---------- Interval enforcement ----------


@pytest.mark.asyncio
async def test_minimum_interval_enforced():
    mgr = _make_manager()
    watch = _make_watch(interval=5)
    mgr.add(watch)
    stored = mgr.list_watches()[0]
    assert stored.interval == 30
    mgr.stop_all()


# ---------- Regex validation ----------


def test_invalid_regex_raises():
    mgr = _make_manager()
    watch = _make_watch(parse_pattern=r"[invalid(")
    with pytest.raises(ValueError, match="Invalid regex"):
        mgr.add(watch)


def test_missing_value_group_raises():
    mgr = _make_manager()
    watch = _make_watch(parse_pattern=r"\d+")
    with pytest.raises(ValueError, match="named group 'value'"):
        mgr.add(watch)


# ---------- stop_all ----------


@pytest.mark.asyncio
async def test_stop_all_clears_watches():
    mgr = _make_manager()
    mgr.add(_make_watch(metric_name="m1"))
    mgr.add(_make_watch(metric_name="m2"))
    assert len(mgr.list_watches()) == 2
    mgr.stop_all()
    assert len(mgr.list_watches()) == 0


@pytest.mark.asyncio
async def test_stop_all_cancels_tasks():
    mgr = _make_manager()
    mgr.add(_make_watch())
    assert len(mgr._tasks) == 1
    task = list(mgr._tasks.values())[0]
    mgr.stop_all()
    # Yield to let cancellation propagate
    await asyncio.sleep(0)
    assert task.cancelled()
    assert len(mgr._tasks) == 0


# ---------- Collection loop ----------


@pytest.mark.asyncio
async def test_collect_once_records_metric():
    """A successful command + regex match should record a MetricPoint."""
    dm = AsyncMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=CommandResult(
        command="show interfaces ge-0/0/0 extensive",
        output="  CRC Errors          42\n  Input Errors         0",
        driver_used="pyez",
        success=True,
    ))

    ms = AsyncMock(spec=MetricsStore)
    ms.record = AsyncMock()

    mgr = WatchManager(device_manager=dm, metrics_store=ms)
    watch = _make_watch()
    watch.id = mgr._make_id(watch.device, watch.command, watch.metric_name)
    compiled = re.compile(watch.parse_pattern)

    await mgr._collect_once(watch, compiled)

    ms.record.assert_called_once()
    point = ms.record.call_args[0][0]
    assert isinstance(point, MetricPoint)
    assert point.device == "r1"
    assert point.category == "watch"
    assert point.metric == "ge0_crc_errors"
    assert point.value == 42.0
    assert point.unit == "errors"


@pytest.mark.asyncio
async def test_collect_once_no_match_no_record():
    """When regex doesn't match, no metric is recorded."""
    dm = AsyncMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=CommandResult(
        command="show interfaces ge-0/0/0 extensive",
        output="  No relevant data here",
        driver_used="pyez",
        success=True,
    ))

    ms = AsyncMock(spec=MetricsStore)
    ms.record = AsyncMock()

    mgr = WatchManager(device_manager=dm, metrics_store=ms)
    watch = _make_watch()
    watch.id = mgr._make_id(watch.device, watch.command, watch.metric_name)
    compiled = re.compile(watch.parse_pattern)

    await mgr._collect_once(watch, compiled)

    ms.record.assert_not_called()


@pytest.mark.asyncio
async def test_collect_once_command_failure_no_record():
    """A failed command should not record a metric."""
    dm = AsyncMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=CommandResult(
        command="show interfaces ge-0/0/0 extensive",
        output="",
        driver_used="pyez",
        success=False,
        error="Connection timeout",
    ))

    ms = AsyncMock(spec=MetricsStore)
    ms.record = AsyncMock()

    mgr = WatchManager(device_manager=dm, metrics_store=ms)
    watch = _make_watch()
    watch.id = mgr._make_id(watch.device, watch.command, watch.metric_name)
    compiled = re.compile(watch.parse_pattern)

    await mgr._collect_once(watch, compiled)

    ms.record.assert_not_called()


@pytest.mark.asyncio
async def test_collect_once_non_numeric_value_no_record():
    """When the named group matches non-numeric text, no metric is recorded."""
    dm = AsyncMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=CommandResult(
        command="show interfaces ge-0/0/0 extensive",
        output="  Status: (?P<value>abc)",
        driver_used="pyez",
        success=True,
    ))

    ms = AsyncMock(spec=MetricsStore)
    ms.record = AsyncMock()

    mgr = WatchManager(device_manager=dm, metrics_store=ms)
    watch = _make_watch(parse_pattern=r"Status: (?P<value>\w+)")
    watch.id = mgr._make_id(watch.device, watch.command, watch.metric_name)
    compiled = re.compile(watch.parse_pattern)

    await mgr._collect_once(watch, compiled)

    ms.record.assert_not_called()


# ---------- Tool dispatch via AgentCore ----------


@pytest.mark.asyncio
async def test_tool_dispatch_add():
    """manage_watches add action dispatches to WatchManager.add."""
    from jace.agent.core import AgentCore
    from jace.config.settings import LLMConfig, ScheduleConfig, Settings
    from jace.llm.base import ToolCall

    dm = MagicMock(spec=DeviceManager)
    ms = AsyncMock(spec=MetricsStore)
    wm = WatchManager(device_manager=dm, metrics_store=ms)

    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    agent = AgentCore(
        settings=settings,
        llm=AsyncMock(),
        device_manager=dm,
        check_registry=AsyncMock(),
        findings_tracker=AsyncMock(),
        watch_manager=wm,
    )

    tool_call = ToolCall(
        id="tc1",
        name="manage_watches",
        arguments={
            "action": "add",
            "device": "r1",
            "command": "show interfaces ge-0/0/0 extensive",
            "metric_name": "crc_errs",
            "parse_pattern": r"CRC Errors\s+(?P<value>\d+)",
            "interval": 60,
            "unit": "errors",
        },
    )

    result = await agent._execute_tool(tool_call)
    assert "Watch created:" in result
    assert len(wm.list_watches()) == 1
    wm.stop_all()


@pytest.mark.asyncio
async def test_tool_dispatch_list_empty():
    from jace.agent.core import AgentCore
    from jace.config.settings import LLMConfig, ScheduleConfig, Settings
    from jace.llm.base import ToolCall

    wm = WatchManager(
        device_manager=MagicMock(spec=DeviceManager),
        metrics_store=AsyncMock(spec=MetricsStore),
    )

    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    agent = AgentCore(
        settings=settings,
        llm=AsyncMock(),
        device_manager=MagicMock(),
        check_registry=AsyncMock(),
        findings_tracker=AsyncMock(),
        watch_manager=wm,
    )

    tool_call = ToolCall(
        id="tc2",
        name="manage_watches",
        arguments={"action": "list"},
    )

    result = await agent._execute_tool(tool_call)
    assert result == "No active watches."


@pytest.mark.asyncio
async def test_tool_dispatch_remove():
    from jace.agent.core import AgentCore
    from jace.config.settings import LLMConfig, ScheduleConfig, Settings
    from jace.llm.base import ToolCall

    dm = MagicMock(spec=DeviceManager)
    ms = AsyncMock(spec=MetricsStore)
    wm = WatchManager(device_manager=dm, metrics_store=ms)

    # First add a watch
    watch = _make_watch()
    watch_id = wm.add(watch)

    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    agent = AgentCore(
        settings=settings,
        llm=AsyncMock(),
        device_manager=dm,
        check_registry=AsyncMock(),
        findings_tracker=AsyncMock(),
        watch_manager=wm,
    )

    tool_call = ToolCall(
        id="tc3",
        name="manage_watches",
        arguments={"action": "remove", "watch_id": watch_id},
    )

    result = await agent._execute_tool(tool_call)
    assert "removed" in result
    assert len(wm.list_watches()) == 0


@pytest.mark.asyncio
async def test_tool_dispatch_add_invalid_regex():
    from jace.agent.core import AgentCore
    from jace.config.settings import LLMConfig, ScheduleConfig, Settings
    from jace.llm.base import ToolCall

    wm = WatchManager(
        device_manager=MagicMock(spec=DeviceManager),
        metrics_store=AsyncMock(spec=MetricsStore),
    )

    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    agent = AgentCore(
        settings=settings,
        llm=AsyncMock(),
        device_manager=MagicMock(),
        check_registry=AsyncMock(),
        findings_tracker=AsyncMock(),
        watch_manager=wm,
    )

    tool_call = ToolCall(
        id="tc4",
        name="manage_watches",
        arguments={
            "action": "add",
            "device": "r1",
            "command": "show interfaces",
            "metric_name": "test",
            "parse_pattern": "[invalid(",
        },
    )

    result = await agent._execute_tool(tool_call)
    assert "Error:" in result
    assert len(wm.list_watches()) == 0


@pytest.mark.asyncio
async def test_tool_dispatch_no_watch_manager():
    """When watch_manager is None, tool returns not-configured message."""
    from jace.agent.core import AgentCore
    from jace.config.settings import LLMConfig, ScheduleConfig, Settings
    from jace.llm.base import ToolCall

    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    agent = AgentCore(
        settings=settings,
        llm=AsyncMock(),
        device_manager=MagicMock(),
        check_registry=AsyncMock(),
        findings_tracker=AsyncMock(),
        watch_manager=None,
    )

    tool_call = ToolCall(
        id="tc5",
        name="manage_watches",
        arguments={"action": "list"},
    )

    result = await agent._execute_tool(tool_call)
    assert "not configured" in result.lower()
