"""Interactive terminal REPL using Rich."""

from __future__ import annotations

import asyncio
import logging
import signal

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from jace.agent.core import AgentCore
from jace.agent.findings import Finding
from jace.ui.log_panel import LogPanel
from jace.ui.notifications import (
    format_status_bar,
    render_finding,
    render_findings_summary,
)

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
  /clear      — Clear the screen
  /help       — Show this help message
  /quit       — Exit the agent

Or just type a natural language question about your network devices.\
"""


class InteractiveShell:
    """Rich-based interactive REPL for JACE."""

    def __init__(self, agent: AgentCore) -> None:
        self._agent = agent
        self._console = Console()
        self._running = False
        self._log_panel = LogPanel(self._console)

    async def run(self) -> None:
        """Main REPL loop."""
        self._running = True
        self._log_panel.setup()
        try:
            self._console.print(Panel(
                BANNER,
                style="bold cyan",
                border_style="cyan",
            ))
            self._console.print("[dim]Type /help for commands, or ask a question.[/dim]\n")

            # Set up notification callback
            self._agent.set_notify_callback(self._on_finding)

            while self._running:
                try:
                    user_input = await self._prompt()
                except (EOFError, KeyboardInterrupt):
                    self._console.print("\n[dim]Goodbye![/dim]")
                    break

                if not user_input.strip():
                    continue

                await self._handle_input(user_input.strip())
        finally:
            self._log_panel.teardown()

    async def _prompt(self) -> str:
        """Display prompt and read user input (runs in executor for async)."""
        loop = asyncio.get_running_loop()

        # Show status bar
        devices = self._agent._device_manager.get_connected_devices()
        findings_count = self._agent._findings.active_count
        critical_count = self._agent._findings.critical_count
        status = format_status_bar(len(devices), findings_count, critical_count)
        self._console.print(Text.from_markup(f"[dim]{status}[/dim]"))

        try:
            return await loop.run_in_executor(
                None,
                lambda: self._console.input("[bold cyan]jace>[/bold cyan] "),
            )
        except EOFError:
            raise

    async def _handle_input(self, text: str) -> None:
        """Handle user input — commands or natural language queries."""
        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._handle_query(text)

    async def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/exit":
            self._running = False
            self._console.print("[dim]Shutting down...[/dim]")

        elif cmd == "/help":
            self._console.print(Panel(HELP_TEXT, title="Help", border_style="cyan"))

        elif cmd == "/clear":
            self._console.clear()
            self._log_panel.restore_after_clear()

        elif cmd == "/devices":
            devices = self._agent._device_manager.list_devices()
            if not devices:
                self._console.print("[yellow]No devices configured.[/yellow]")
                return
            for d in devices:
                status_style = {
                    "connected": "green",
                    "disconnected": "red",
                    "error": "bold red",
                    "connecting": "yellow",
                }.get(d.status.value, "white")
                self._console.print(
                    f"  [{status_style}]{d.status.value:12s}[/{status_style}] "
                    f"[bold]{d.name}[/bold] ({d.host}) "
                    f"[dim]{d.model} {d.version}[/dim]"
                )

        elif cmd == "/findings":
            findings = self._agent._findings.get_active()
            render_findings_summary(self._console, findings)
            for f in findings:
                render_finding(self._console, f, is_new=False)

        elif cmd == "/check":
            if len(parts) < 3:
                self._console.print(
                    "[yellow]Usage: /check <device> <category>[/yellow]"
                )
                return
            device_name, category = parts[1], parts[2]
            self._console.print(
                f"[dim]Running {category} check on {device_name}...[/dim]"
            )
            try:
                response = await self._agent.handle_user_input(
                    f"Run a {category} health check on {device_name} and report the results."
                )
                self._console.print(Markdown(response))
            except Exception as exc:
                self._console.print(f"[red]Error: {exc}[/red]")

        else:
            self._console.print(f"[yellow]Unknown command: {cmd}. Type /help.[/yellow]")

    async def _handle_query(self, text: str) -> None:
        """Send a natural language query to the agent."""
        self._console.print("[dim]Thinking...[/dim]")
        try:
            response = await self._agent.handle_user_input(text)
            self._console.print()
            self._console.print(Markdown(response))
            self._console.print()
        except Exception as exc:
            self._console.print(f"[red]Error: {exc}[/red]")

    async def _on_finding(self, finding: Finding, is_new: bool) -> None:
        """Callback for background findings — renders alerts above the prompt."""
        render_finding(self._console, finding, is_new)
