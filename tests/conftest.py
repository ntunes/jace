"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from jace.agent.findings import FindingsTracker
from jace.config.settings import (
    DeviceConfig,
    LLMConfig,
    ScheduleConfig,
    Settings,
)
from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


@pytest.fixture
def sample_settings() -> Settings:
    return Settings(
        llm=LLMConfig(provider="anthropic", model="test-model", api_key="test-key"),
        devices=[
            DeviceConfig(name="test-router", host="192.168.1.1", username="admin", password="secret"),
        ],
        schedule=ScheduleConfig(chassis=60, interfaces=60, routing=60, system=60, config=300),
    )


@pytest.fixture
def mock_device_manager() -> DeviceManager:
    mgr = DeviceManager()
    mgr.add_device(DeviceConfig(
        name="test-router", host="192.168.1.1", username="admin", password="secret",
    ))
    return mgr


@pytest.fixture
def mock_command_result() -> CommandResult:
    return CommandResult(
        command="show chassis alarms",
        output="No alarms currently active",
        driver_used="pyez",
        success=True,
    )


@pytest.fixture
async def findings_tracker(tmp_path: Path) -> FindingsTracker:
    tracker = FindingsTracker(tmp_path)
    await tracker.initialize()
    yield tracker
    await tracker.close()


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    from jace.llm.base import Response
    client = AsyncMock()
    client.chat = AsyncMock(return_value=Response(
        content="No issues found.", stop_reason="end_turn",
    ))
    return client
