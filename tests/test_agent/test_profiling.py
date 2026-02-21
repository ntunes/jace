"""Tests for automatic device profiling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.core import AgentCore, PROFILE_PROMPT_TEMPLATE
from jace.agent.findings import FindingsTracker
from jace.agent.memory import MemoryStore
from jace.checks.registry import CheckRegistry
from jace.config.settings import LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.device.models import CommandResult
from jace.llm.base import Response, ToolCall


def _make_agent(
    *,
    llm: AsyncMock | None = None,
    device_manager: MagicMock | None = None,
    memory_store: MagicMock | None = None,
) -> AgentCore:
    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    return AgentCore(
        settings=settings,
        llm=llm or AsyncMock(),
        device_manager=device_manager or MagicMock(spec=DeviceManager),
        check_registry=AsyncMock(spec=CheckRegistry),
        findings_tracker=AsyncMock(spec=FindingsTracker),
        memory_store=memory_store,
    )


def _cmd_result(output: str = "sample output") -> CommandResult:
    return CommandResult(
        command="test", output=output, driver_used="pyez", success=True,
    )


# ---------- profile_device tests ----------


@pytest.mark.asyncio
async def test_profile_device_runs_expected_commands():
    """profile_device should run the three profiling commands."""
    dm = MagicMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=_cmd_result())

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="## Device Profile\n**Role:** core", stop_reason="end_turn",
    ))

    memory = MagicMock(spec=MemoryStore)
    memory.save = MagicMock(return_value="Saved.")

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=memory)
    await agent.profile_device("r1")

    assert dm.run_command.call_count == 3
    commands = [call.args[1] for call in dm.run_command.call_args_list]
    assert any("routing-instances" in c for c in commands)
    assert any("route summary" in c for c in commands)
    assert any("chassis hardware" in c for c in commands)


@pytest.mark.asyncio
async def test_profile_device_saves_to_memory():
    """profile_device should save the LLM result to memory store."""
    dm = MagicMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=_cmd_result())

    profile_text = "## Device Profile\n**Role:** PE"
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content=profile_text, stop_reason="end_turn",
    ))

    memory = MagicMock(spec=MemoryStore)
    memory.save = MagicMock(return_value="Saved.")

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=memory)
    result = await agent.profile_device("r1")

    memory.save.assert_called_once_with("device", "r1", profile_text)
    assert result == profile_text


@pytest.mark.asyncio
async def test_profile_device_sends_profile_prompt():
    """profile_device should send the PROFILE_PROMPT_TEMPLATE to the LLM."""
    dm = MagicMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=_cmd_result("config data"))

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="## Device Profile\n**Role:** edge", stop_reason="end_turn",
    ))

    memory = MagicMock(spec=MemoryStore)
    memory.save = MagicMock(return_value="Saved.")

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=memory)
    await agent.profile_device("r1")

    llm.chat.assert_called()
    call_kwargs = llm.chat.call_args
    messages = call_kwargs.kwargs["messages"]
    prompt = messages[-1].content
    assert "Device Profile" in prompt
    assert "Role" in prompt
    assert "r1" in prompt


@pytest.mark.asyncio
async def test_profile_device_works_without_memory_store():
    """profile_device should work even if memory_store is None."""
    dm = MagicMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=_cmd_result())

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="## Device Profile\n**Role:** core", stop_reason="end_turn",
    ))

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=None)
    result = await agent.profile_device("r1")

    assert "Device Profile" in result


# ---------- profile_all_devices tests ----------


@pytest.mark.asyncio
async def test_profile_all_skips_already_profiled():
    """Devices with existing '## Device Profile' in memory should be skipped."""
    dm = MagicMock(spec=DeviceManager)
    dm.get_connected_devices = MagicMock(return_value=["r1", "r2"])
    dm.run_command = AsyncMock(return_value=_cmd_result())

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="## Device Profile\n**Role:** edge", stop_reason="end_turn",
    ))

    memory = MagicMock(spec=MemoryStore)
    memory.get_device = MagicMock(side_effect=lambda name: {
        "r1": "# Device: r1\n\n## Device Profile\n**Role:** core",
        "r2": "# Device: r2\n\nSome notes",
    }[name])
    memory.save = MagicMock(return_value="Saved.")

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=memory)
    await agent.profile_all_devices()

    # Only r2 should be profiled (r1 already has a profile)
    assert dm.run_command.call_count == 3  # 3 commands for r2 only


@pytest.mark.asyncio
async def test_profile_all_profiles_new_devices():
    """Devices without a profile should get profiled."""
    dm = MagicMock(spec=DeviceManager)
    dm.get_connected_devices = MagicMock(return_value=["r1", "r2"])
    dm.run_command = AsyncMock(return_value=_cmd_result())

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="## Device Profile\n**Role:** core", stop_reason="end_turn",
    ))

    memory = MagicMock(spec=MemoryStore)
    memory.get_device = MagicMock(return_value="")
    memory.save = MagicMock(return_value="Saved.")

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=memory)
    await agent.profile_all_devices()

    # Both devices profiled: 3 commands each = 6 total
    assert dm.run_command.call_count == 6
    assert memory.save.call_count == 2


@pytest.mark.asyncio
async def test_profile_all_no_connected_devices():
    """profile_all_devices should handle no connected devices gracefully."""
    dm = MagicMock(spec=DeviceManager)
    dm.get_connected_devices = MagicMock(return_value=[])

    agent = _make_agent(device_manager=dm)
    await agent.profile_all_devices()  # Should not raise


# ---------- Tool execution tests ----------


@pytest.mark.asyncio
async def test_profile_tool_execution():
    """_execute_tool should handle 'profile_device' tool calls."""
    dm = MagicMock(spec=DeviceManager)
    dm.run_command = AsyncMock(return_value=_cmd_result())

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(
        content="## Device Profile\n**Role:** PE", stop_reason="end_turn",
    ))

    memory = MagicMock(spec=MemoryStore)
    memory.save = MagicMock(return_value="Saved.")

    agent = _make_agent(llm=llm, device_manager=dm, memory_store=memory)

    tool_call = ToolCall(
        id="call-1", name="profile_device",
        arguments={"device": "r1"},
    )
    result = await agent._execute_tool(tool_call)

    assert "Device Profile" in result
    memory.save.assert_called_once()


@pytest.mark.asyncio
async def test_profile_tool_without_memory_store():
    """profile_device tool should return error when memory store is missing."""
    agent = _make_agent(memory_store=None)

    tool_call = ToolCall(
        id="call-1", name="profile_device",
        arguments={"device": "r1"},
    )
    result = await agent._execute_tool(tool_call)

    assert "not configured" in result.lower()
