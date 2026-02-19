"""Tests for the Textual TUI — uses Textual's pilot framework."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.core import AgentCore
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.config.settings import LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.device.models import DeviceInfo, DeviceStatus
from jace.ui.tui import JaceApp
from jace.ui.widgets import ChatInput, ChatView, DeviceList, FindingsList


def _make_app(
    *,
    agent: AgentCore | None = None,
    device_manager: DeviceManager | None = None,
    findings_tracker: FindingsTracker | None = None,
) -> JaceApp:
    if agent is None:
        agent = MagicMock(spec=AgentCore)
        agent.set_notify_callback = MagicMock()
        agent.handle_user_input = AsyncMock(return_value="Test response")

    if device_manager is None:
        device_manager = MagicMock(spec=DeviceManager)
        device_manager.list_devices = MagicMock(return_value=[])
        device_manager.get_connected_devices = MagicMock(return_value=[])

    if findings_tracker is None:
        findings_tracker = MagicMock(spec=FindingsTracker)
        findings_tracker.get_active = MagicMock(return_value=[])
        findings_tracker.active_count = 0
        findings_tracker.critical_count = 0

    return JaceApp(
        agent=agent,
        device_manager=device_manager,
        findings_tracker=findings_tracker,
    )


def _sample_finding() -> Finding:
    return Finding(
        id="abc123",
        device="mx-01",
        severity=Severity.WARNING,
        category="chassis",
        title="Fan tray 1 failure",
        detail="Fan tray 1 is reporting a failure.",
        recommendation="Replace fan tray 1.",
        first_seen="2025-01-01T00:00:00",
        last_seen="2025-01-01T00:00:00",
    )


@pytest.mark.asyncio
async def test_widget_tree_composition():
    """Verify the widget tree has all expected components."""
    app = _make_app()
    async with app.run_test() as pilot:
        assert app.query_one("#device-list", DeviceList)
        assert app.query_one("#findings-list", FindingsList)
        assert app.query_one("#chat-view", ChatView)
        assert app.query_one("#chat-input", ChatInput)
        assert app.query_one("#log-panel")


@pytest.mark.asyncio
async def test_input_submission_routes_to_agent():
    """Typing a query and pressing Enter should call agent.handle_user_input."""
    agent = MagicMock(spec=AgentCore)
    agent.set_notify_callback = MagicMock()
    agent.handle_user_input = AsyncMock(return_value="LLM says hello")

    app = _make_app(agent=agent)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#chat-input", ChatInput)
        input_widget.focus()
        await pilot.press(*"hello")
        await pilot.press("enter")
        await pilot.pause()

        agent.handle_user_input.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_sidebar_updates_devices():
    """Sidebar DeviceList should update when _refresh_sidebar() is called."""
    dm = MagicMock(spec=DeviceManager)
    dm.list_devices = MagicMock(return_value=[
        DeviceInfo(name="mx-01", host="10.0.0.1", status=DeviceStatus.CONNECTED),
        DeviceInfo(name="mx-02", host="10.0.0.2", status=DeviceStatus.DISCONNECTED),
    ])
    dm.get_connected_devices = MagicMock(return_value=["mx-01"])

    app = _make_app(device_manager=dm)
    async with app.run_test() as pilot:
        # on_mount already called _refresh_sidebar, wait for it to render
        await pilot.pause()
        device_list = app.query_one("#device-list", DeviceList)
        # Static stores its content — render it to string to check
        rendered = str(device_list._Static__content)
        assert "mx-01" in rendered
        assert "mx-02" in rendered


@pytest.mark.asyncio
async def test_sidebar_updates_findings():
    """Sidebar FindingsList should show active findings."""
    ft = MagicMock(spec=FindingsTracker)
    ft.get_active = MagicMock(return_value=[_sample_finding()])
    ft.active_count = 1
    ft.critical_count = 0

    app = _make_app(findings_tracker=ft)
    async with app.run_test() as pilot:
        await pilot.pause()
        findings_list = app.query_one("#findings-list", FindingsList)
        rendered = str(findings_list._Static__content)
        assert "Fan tray" in rendered


@pytest.mark.asyncio
async def test_finding_notification_appears_in_chat():
    """When _on_finding is called, the finding should appear in the chat."""
    app = _make_app()
    finding = _sample_finding()

    async with app.run_test() as pilot:
        chat: ChatView = app.query_one("#chat-view")
        lines_before = len(chat.lines)
        await app._on_finding(finding, is_new=True)
        await pilot.pause()
        # The finding panel was written to the ChatView
        assert len(chat.lines) > lines_before


@pytest.mark.asyncio
async def test_help_command():
    """/help should display the help panel in chat."""
    app = _make_app()
    async with app.run_test() as pilot:
        chat: ChatView = app.query_one("#chat-view")
        lines_before = len(chat.lines)

        input_widget = app.query_one("#chat-input", ChatInput)
        input_widget.focus()
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()

        # Help panel adds lines to chat
        assert len(chat.lines) > lines_before


@pytest.mark.asyncio
async def test_quit_command():
    """/quit should exit the app."""
    app = _make_app()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#chat-input", ChatInput)
        input_widget.focus()
        await pilot.press(*"/quit")
        await pilot.press("enter")
        await pilot.pause()

        # App should have exited (or be in process of exiting)
        # The run_test context manager handles cleanup


@pytest.mark.asyncio
async def test_unknown_command_shows_error():
    """/foobar should show an unknown command message."""
    app = _make_app()
    async with app.run_test() as pilot:
        chat: ChatView = app.query_one("#chat-view")
        lines_before = len(chat.lines)

        input_widget = app.query_one("#chat-input", ChatInput)
        input_widget.focus()
        await pilot.press(*"/foobar")
        await pilot.press("enter")
        await pilot.pause()

        assert len(chat.lines) > lines_before


@pytest.mark.asyncio
async def test_clear_command_clears_chat():
    """/clear should clear the chat log."""
    app = _make_app()
    async with app.run_test() as pilot:
        chat: ChatView = app.query_one("#chat-view")
        await pilot.pause()
        initial_lines = len(chat.lines)
        assert initial_lines > 0  # banner + system msg

        input_widget = app.query_one("#chat-input", ChatInput)
        input_widget.focus()
        await pilot.press(*"/clear")
        await pilot.press("enter")
        await pilot.pause()

        assert len(chat.lines) < initial_lines
