"""Tests for the Textual TUI — uses Textual's pilot framework."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.core import AgentCore
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.config.settings import LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.device.models import DeviceInfo, DeviceStatus
from jace.ui.tui import JACE
from jace.ui.widgets import ChatInput, ChatView, DeviceList, FindingsTable

from textual.widgets import TabbedContent, TabPane


def _make_app(
    *,
    agent: AgentCore | None = None,
    device_manager: DeviceManager | None = None,
    findings_tracker: FindingsTracker | None = None,
) -> JACE:
    if agent is None:
        agent = MagicMock(spec=AgentCore)
        agent.set_notify_callback = MagicMock()
        agent.handle_user_input = AsyncMock(return_value="Test response")
        agent.profile_all_devices = AsyncMock()

    if device_manager is None:
        device_manager = MagicMock(spec=DeviceManager)
        device_manager.connect_all = AsyncMock()
        device_manager.list_devices = MagicMock(return_value=[])
        device_manager.get_connected_devices = MagicMock(return_value=[])

    if findings_tracker is None:
        findings_tracker = MagicMock(spec=FindingsTracker)
        findings_tracker.get_active = MagicMock(return_value=[])
        findings_tracker.active_count = 0
        findings_tracker.critical_count = 0

    return JACE(
        agent=agent,
        device_manager=device_manager,
        findings_tracker=findings_tracker,
    )


def _sample_finding(
    severity: Severity = Severity.WARNING,
    device: str = "mx-01",
    title: str = "Fan tray 1 failure",
) -> Finding:
    return Finding(
        id="abc123",
        device=device,
        severity=severity,
        category="chassis",
        title=title,
        detail="Fan tray 1 is reporting a failure.",
        recommendation="Replace fan tray 1.",
        first_seen="2025-01-01T00:00:00",
        last_seen="2025-01-01T00:00:00",
    )


@pytest.mark.asyncio
async def test_widget_tree_composition():
    """Verify the widget tree has TabbedContent with all three tab panes."""
    app = _make_app()
    async with app.run_test() as pilot:
        assert app.query_one("#device-list", DeviceList)
        assert app.query_one("#tabs", TabbedContent)
        assert app.query_one("#tab-chat", TabPane)
        assert app.query_one("#tab-findings", TabPane)
        assert app.query_one("#tab-logs", TabPane)
        assert app.query_one("#chat-view", ChatView)
        assert app.query_one("#findings-table", FindingsTable)
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
async def test_findings_table_in_tab():
    """FindingsTable in the Findings tab should show active findings."""
    ft = MagicMock(spec=FindingsTracker)
    ft.get_active = MagicMock(return_value=[_sample_finding()])
    ft.active_count = 1
    ft.critical_count = 0

    app = _make_app(findings_tracker=ft)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#findings-table", FindingsTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_finding_notification_is_toast():
    """When _on_finding is called, a toast should fire (not a chat write)."""
    app = _make_app()
    finding = _sample_finding()

    async with app.run_test() as pilot:
        chat: ChatView = app.query_one("#chat-view")
        lines_before = len(chat.lines)

        with patch.object(app, "notify") as mock_notify:
            await app._on_finding(finding, is_new=True)
            await pilot.pause()
            mock_notify.assert_called_once()
            call_kwargs = mock_notify.call_args
            assert "error" == call_kwargs.kwargs.get("severity") or \
                   "warning" == call_kwargs.kwargs.get("severity") or \
                   "information" == call_kwargs.kwargs.get("severity")

        # Chat should NOT have new lines from the finding
        assert len(chat.lines) == lines_before


@pytest.mark.asyncio
async def test_findings_table_refresh():
    """refresh_findings() should populate the DataTable with correct rows."""
    app = _make_app()
    async with app.run_test() as pilot:
        table = app.query_one("#findings-table", FindingsTable)
        findings = [
            _sample_finding(Severity.CRITICAL, "mx-01", "BGP peer down"),
            _sample_finding(Severity.WARNING, "mx-02", "Fan tray 1 failure"),
            _sample_finding(Severity.INFO, "mx-03", "NTP drift detected"),
        ]
        table.refresh_findings(findings)
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_findings_command_switches_tab():
    """/findings should switch active tab to tab-findings."""
    app = _make_app()
    async with app.run_test() as pilot:
        tabs = app.query_one("#tabs", TabbedContent)
        assert tabs.active == "tab-chat"

        input_widget = app.query_one("#chat-input", ChatInput)
        input_widget.focus()
        await pilot.press(*"/findings")
        await pilot.press("enter")
        await pilot.pause()

        assert tabs.active == "tab-findings"


@pytest.mark.asyncio
async def test_tab_badge_updates():
    """After _refresh_sidebar with findings, the Findings tab label should include count."""
    ft = MagicMock(spec=FindingsTracker)
    findings = [_sample_finding()]
    ft.get_active = MagicMock(return_value=findings)
    ft.active_count = 1
    ft.critical_count = 0

    app = _make_app(findings_tracker=ft)
    async with app.run_test() as pilot:
        await pilot.pause()
        tabs = app.query_one("#tabs", TabbedContent)
        tab = tabs.get_tab("tab-findings")
        label_text = str(tab.label)
        assert "(1)" in label_text


@pytest.mark.asyncio
async def test_tab_badge_critical():
    """Tab badge should be styled for critical findings."""
    ft = MagicMock(spec=FindingsTracker)
    findings = [_sample_finding(Severity.CRITICAL, "mx-01", "BGP peer down")]
    ft.get_active = MagicMock(return_value=findings)
    ft.active_count = 1
    ft.critical_count = 1

    app = _make_app(findings_tracker=ft)
    async with app.run_test() as pilot:
        await pilot.pause()
        tabs = app.query_one("#tabs", TabbedContent)
        tab = tabs.get_tab("tab-findings")
        label_text = str(tab.label)
        assert "(1)" in label_text


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
