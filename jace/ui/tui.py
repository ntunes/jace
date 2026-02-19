"""Textual TUI for JACE — chat interface with live sidebar and log footer."""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog

from jace.agent.core import AgentCore
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.device.manager import DeviceManager
from jace.ui.logging_handler import TextualLogHandler
from jace.ui.notifications import render_finding_panel, render_findings_summary
from jace.ui.widgets import ChatInput, ChatView, DeviceList, FindingsList

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
  /findings   — Show active findings summary
  /check <device> <category> — Run a health check now
  /clear      — Clear the chat log
  /help       — Show this help message
  /quit       — Exit the agent

Or just type a natural language question about your network devices.\
"""

SIDEBAR_REFRESH_SECONDS = 3


class JaceApp(App):
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
    #chat-view {
        height: 1fr;
    }
    #chat-input {
        dock: bottom;
        margin-top: 1;
    }
    #log-panel {
        height: 5;
        dock: bottom;
        border-top: solid $primary-background-lighten-2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
        Binding("f1", "show_help", "Help", show=True),
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

    # ── Layout ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield DeviceList(id="device-list")
                yield FindingsList(id="findings-list")
            with Vertical(id="main-area"):
                yield ChatView(id="chat-view", highlight=True, markup=True)
                yield ChatInput()
        yield RichLog(id="log-panel", highlight=True, markup=True)
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

        # Start sidebar refresh timer
        self.set_interval(SIDEBAR_REFRESH_SECONDS, self._refresh_sidebar)

        # Show banner + initial sidebar
        chat: ChatView = self.query_one("#chat-view")
        chat.write(Panel(BANNER, style="bold cyan", border_style="cyan"))
        chat.add_system_message("Type /help for commands, or ask a question.\n")

        self._refresh_sidebar()
        self.query_one("#chat-input").focus()

    def on_unmount(self) -> None:
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    # ── Input handling ──────────────────────────────────────────────────

    async def on_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            await self._handle_command(text)
        else:
            self._handle_query(text)

    # ── Query handling (async worker) ───────────────────────────────────

    @work(exclusive=True)
    async def _handle_query(self, text: str) -> None:
        """Send a natural language query to the agent."""
        chat: ChatView = self.query_one("#chat-view")
        chat.add_user_message(text)
        chat.add_system_message("Thinking...")

        try:
            response = await self._agent.handle_user_input(text)
            chat.add_agent_response(response)
        except Exception as exc:
            chat.write(Text(f"Error: {exc}", style="red"))

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
                line.append(d.name, style="bold")
                line.append(f" ({d.host}) ", style="")
                line.append(f"{d.model} {d.version}", style="dim")
                chat.write(line)

        elif cmd == "/findings":
            findings = self._findings_tracker.get_active()
            if not findings:
                chat.write(Text("No active findings.", style="green"))
                return
            # Summary counts
            critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
            warning = sum(1 for f in findings if f.severity == Severity.WARNING)
            info = sum(1 for f in findings if f.severity == Severity.INFO)
            summary = Text()
            summary.append("Active Findings: ", style="bold")
            if critical:
                summary.append(f"{critical} critical ", style="bold red")
            if warning:
                summary.append(f"{warning} warning ", style="bold yellow")
            if info:
                summary.append(f"{info} info ", style="bold blue")
            chat.write(Panel(summary, border_style="dim"))
            for f in findings:
                chat.write(render_finding_panel(f, is_new=False))

        elif cmd == "/check":
            if len(parts) < 3:
                chat.write(Text("Usage: /check <device> <category>", style="yellow"))
                return
            device_name, category = parts[1], parts[2]
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
            chat: ChatView = self.query_one("#chat-view")
            chat.show_finding_alert(finding, is_new)
            self._refresh_sidebar()

        self.call_later(_show)

    # ── Sidebar refresh ─────────────────────────────────────────────────

    def _refresh_sidebar(self) -> None:
        devices = self._device_manager.list_devices()
        findings = self._findings_tracker.get_active()

        device_list: DeviceList = self.query_one("#device-list")
        device_list.update_devices(devices)

        findings_list: FindingsList = self.query_one("#findings-list")
        findings_list.update_findings(findings)

    # ── Keybinding actions ──────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        chat: ChatView = self.query_one("#chat-view")
        chat.clear()

    def action_show_help(self) -> None:
        chat: ChatView = self.query_one("#chat-view")
        chat.write(Panel(HELP_TEXT, title="Help", border_style="cyan"))
