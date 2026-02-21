"""Textual TUI for JACE — tabbed chat interface with live sidebar."""

from __future__ import annotations

import asyncio
import logging

from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, TabbedContent, TabPane

from jace.agent.core import AgentCore
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.device.manager import DeviceManager
from jace.ui.logging_handler import TextualLogHandler
from jace.ui.notifications import finding_toast_params
from jace.ui.widgets import ChatInput, ChatView, DeviceList, FindingsTable, ThinkingIndicator

logger = logging.getLogger(__name__)

BANNER = """\
         ╦╔═╗╔═╗╔═╗
         ║╠═╣║  ║╣
        ╚╝╩ ╩╚═╝╚═╝
  JACE: Autonomous Control Engine
"""

HELP_TEXT = """\
Commands:
  /devices    — List managed devices and their status
  /findings   — Switch to the Findings tab
  /check <device> <category> — Run a health check now
  /clear      — Clear the chat log
  /help       — Show this help message
  /quit       — Exit the agent

Or just type a natural language question about your network devices.\
"""

SIDEBAR_REFRESH_SECONDS = 3


class JACE(App):
    """Main Textual application for JACE."""

    CSS = """
    #sidebar {
        width: 26;
        dock: left;
        padding: 1;
        border-right: solid $primary-background-lighten-2;
    }
    #main-area {
        width: 1fr;
    }
    #tabs {
        height: 1fr;
    }
    TabPane {
        height: 1fr;
    }
    #chat-view {
        height: 1fr;
    }
    #thinking {
        height: 1;
        display: none;
    }
    #chat-input {
        dock: bottom;
        margin-top: 1;
    }
    #log-panel {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+f", "focus_findings", "Findings", show=True),
        Binding("ctrl+g", "focus_logs", "Logs", show=True),
    ]

    def __init__(
        self,
        agent: AgentCore,
        device_manager: DeviceManager,
        findings_tracker: FindingsTracker,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._agent = agent
        self._device_manager = device_manager
        self._findings_tracker = findings_tracker
        self._log_handler: TextualLogHandler | None = None
        self._pending_approval: asyncio.Future[bool] | None = None

    # ── Layout ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield DeviceList(id="device-list")
            with Vertical(id="main-area"):
                with TabbedContent(initial="tab-chat", id="tabs"):
                    with TabPane("Chat", id="tab-chat"):
                        yield ChatView(id="chat-view", highlight=True, markup=True)
                        yield ThinkingIndicator(id="thinking")
                    with TabPane("Findings", id="tab-findings"):
                        yield FindingsTable(id="findings-table")
                    with TabPane("Logs", id="tab-logs"):
                        yield RichLog(id="log-panel", highlight=True, markup=True)
                yield ChatInput()
        yield Footer()

    # ── Lifecycle ───────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # Install log handler
        self._log_handler = TextualLogHandler(self)
        root = logging.getLogger()
        # Remove existing StreamHandlers to avoid double output
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TextualLogHandler):
                root.removeHandler(h)
        root.addHandler(self._log_handler)

        # Register notification callback
        self._agent.set_notify_callback(self._on_finding)

        # Register shell approval callback
        self._agent.set_approval_callback(self._request_shell_approval)

        # Register status callback for dynamic thinking indicator
        self._agent.set_status_callback(self._on_status)

        # Start sidebar refresh timer
        self.set_interval(SIDEBAR_REFRESH_SECONDS, self._refresh_sidebar)

        # Show banner + initial sidebar
        chat: ChatView = self.query_one("#chat-view")
        chat.write(Panel(BANNER, style="bold cyan", border_style="cyan"))
        chat.add_system_message("Type /help for commands, or ask a question.\n")

        self._refresh_sidebar()
        self.query_one("#chat-input").focus()

        # Connect devices and start monitoring in the background
        self._connect_and_start()

    def on_unmount(self) -> None:
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    @work(thread=True)
    def _connect_and_start(self) -> None:
        """Connect to devices in a background OS thread."""
        logger.info("Connecting to devices...")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._device_manager.connect_all(
                    on_connect=lambda: self.call_from_thread(self._refresh_sidebar),
                )
            )
        finally:
            loop.close()

        connected = self._device_manager.get_connected_devices()
        logger.info("Connected to %d device(s): %s", len(connected), connected)
        self.call_from_thread(self._refresh_sidebar)
        self.call_from_thread(self._post_connect)

    @work(exclusive=False)
    async def _post_connect(self) -> None:
        """Profile devices and start monitoring (main event loop)."""
        await self._agent.profile_all_devices()
        self._agent.start_monitoring()

    # ── Input handling ──────────────────────────────────────────────────

    async def on_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        # Intercept approval responses
        if self._pending_approval is not None:
            approved = text.lower().startswith("y")
            self._pending_approval.set_result(approved)
            return

        # Auto-switch to Chat tab when submitting a query
        tabs: TabbedContent = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-chat"

        if text.startswith("/"):
            await self._handle_command(text)
        else:
            self._handle_query(text)

    # ── Query handling (async worker) ───────────────────────────────────

    @work(exclusive=True)
    async def _handle_query(self, text: str) -> None:
        """Send a natural language query to the agent."""
        chat: ChatView = self.query_one("#chat-view")
        indicator: ThinkingIndicator = self.query_one("#thinking")
        chat.add_user_message(text)
        indicator.start()

        try:
            response = await self._agent.handle_user_input(text)
            indicator.stop()
            chat.add_agent_response(response)
        except Exception as exc:
            indicator.stop()
            chat.write(Text(f"Error: {exc}", style="red"))

    # ── Shell approval ────────────────────────────────────────────────

    async def _request_shell_approval(self, command: str, reason: str) -> bool:
        """Show an approval prompt and wait for the user to type y/n."""
        chat: ChatView = self.query_one("#chat-view")
        chat_input: ChatInput = self.query_one("#chat-input")

        chat.add_approval_prompt(command, reason)

        # Store original placeholder and set approval prompt
        original_placeholder = chat_input.placeholder
        chat_input.placeholder = "Type y to approve or n to deny..."

        loop = asyncio.get_running_loop()
        self._pending_approval = loop.create_future()

        try:
            result = await self._pending_approval
        finally:
            self._pending_approval = None
            chat_input.placeholder = original_placeholder

        chat.add_approval_result(result)
        return result

    # ── Status callback ─────────────────────────────────────────────────

    def _on_status(self, message: str) -> None:
        """Update the thinking indicator with a contextual status message."""
        indicator: ThinkingIndicator = self.query_one("#thinking")
        indicator.set_status(message)

    # ── Slash commands ──────────────────────────────────────────────────

    async def _handle_command(self, text: str) -> None:
        chat: ChatView = self.query_one("#chat-view")
        chat.add_user_message(text)

        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("/quit", "/exit"):
            self.exit()

        elif cmd == "/help":
            chat.write(Panel(HELP_TEXT, title="Help", border_style="cyan"))

        elif cmd == "/clear":
            chat.clear()

        elif cmd == "/devices":
            devices = self._device_manager.list_devices()
            if not devices:
                chat.write(Text("No devices configured.", style="yellow"))
                return
            for d in devices:
                status_style = {
                    "connected": "green",
                    "disconnected": "red",
                    "error": "bold red",
                    "connecting": "yellow",
                }.get(d.status.value, "white")
                line = Text()
                line.append(f"  {d.status.value:12s} ", style=status_style)
                line.append(d.device_key, style="bold")
                line.append(f" ({d.host}) ", style="")
                line.append(f"{d.model} {d.version}", style="dim")
                chat.write(line)

        elif cmd == "/findings":
            # Switch to the Findings tab
            tabs: TabbedContent = self.query_one("#tabs", TabbedContent)
            tabs.active = "tab-findings"

        elif cmd == "/check":
            if len(parts) < 3:
                chat.write(Text("Usage: /check <device> <category>", style="yellow"))
                return
            device_name, category = parts[1], parts[2]
            try:
                device_name = self._device_manager.resolve_device(device_name)
            except (KeyError, ValueError) as exc:
                chat.write(Text(f"Error: {exc}", style="red"))
                return
            chat.add_system_message(f"Running {category} check on {device_name}...")
            try:
                response = await self._agent.handle_user_input(
                    f"Run a {category} health check on {device_name} and report the results."
                )
                chat.add_agent_response(response)
            except Exception as exc:
                chat.write(Text(f"Error: {exc}", style="red"))

        else:
            chat.write(Text(f"Unknown command: {cmd}. Type /help.", style="yellow"))

    # ── Finding notification callback ───────────────────────────────────

    async def _on_finding(self, finding: Finding, is_new: bool) -> None:
        """Called from the agent when a finding is created/updated/resolved."""
        def _show() -> None:
            params = finding_toast_params(finding, is_new)
            self.notify(**params)
            self._refresh_sidebar()

        self.call_later(_show)

    # ── Sidebar & findings refresh ───────────────────────────────────────

    def _refresh_sidebar(self) -> None:
        devices = self._device_manager.list_devices()
        findings = self._findings_tracker.get_active()

        device_list: DeviceList = self.query_one("#device-list")
        device_list.update_devices(devices)

        # Update findings table
        table: FindingsTable = self.query_one("#findings-table")
        table.refresh_findings(findings)

        self._update_tab_badges(findings)

    def _update_tab_badges(self, findings: list[Finding]) -> None:
        """Update the Findings tab label with the active finding count."""
        tabs: TabbedContent = self.query_one("#tabs", TabbedContent)
        tab = tabs.get_tab("tab-findings")
        count = len(findings)
        if count == 0:
            tab.label = "Findings"
        else:
            has_critical = any(f.severity == Severity.CRITICAL for f in findings)
            if has_critical:
                tab.label = Text.assemble("Findings ", (f"({count})", "bold red"))
            else:
                tab.label = f"Findings ({count})"

    # ── Keybinding actions ──────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        chat: ChatView = self.query_one("#chat-view")
        chat.clear()

    def action_show_help(self) -> None:
        chat: ChatView = self.query_one("#chat-view")
        chat.write(Panel(HELP_TEXT, title="Help", border_style="cyan"))

    def action_focus_findings(self) -> None:
        tabs: TabbedContent = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-findings"

    def action_focus_logs(self) -> None:
        tabs: TabbedContent = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-logs"
