"""Custom Textual widgets for the JACE TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.timer import Timer
from textual.widgets import DataTable, Input, RichLog, Static

if TYPE_CHECKING:
    from jace.agent.findings import Finding, Severity
    from jace.device.models import DeviceInfo

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Status dot colours keyed by DeviceStatus value
_DEVICE_DOTS: dict[str, tuple[str, str]] = {
    "connected":    ("\u25cf", "green"),       # ●
    "disconnected": ("\u25cf", "red"),         # ●
    "connecting":   ("\u25cf", "yellow"),      # ●
    "error":        ("\u25cf", "bold red"),     # ●
}

# Severity icons
_SEVERITY_ICONS: dict[str, tuple[str, str]] = {
    "critical": ("\u25c9", "red"),       # ◉
    "warning":  ("\u26a0", "yellow"),    # ⚠
    "info":     ("\u2139", "blue"),      # ℹ
}


class DeviceList(Static):
    """Renders a list of devices with coloured status dots."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)

    def update_devices(self, devices: list[DeviceInfo]) -> None:
        text = Text()
        text.append("Devices\n", style="bold underline")
        if not devices:
            text.append("  (none)", style="dim")
            self.update(text)
            return

        has_categories = any(dev.category for dev in devices)
        if has_categories:
            # Group by category
            grouped: dict[str, list[DeviceInfo]] = {}
            for dev in devices:
                key = dev.category or "_uncategorized"
                grouped.setdefault(key, []).append(dev)
            for cat in sorted(grouped):
                label = cat if cat != "_uncategorized" else "other"
                text.append(f"  {label}\n", style="bold dim")
                for dev in grouped[cat]:
                    dot, style = _DEVICE_DOTS.get(
                        dev.status.value, ("\u25cf", "white"),
                    )
                    text.append(f"    {dot} ", style=style)
                    text.append(f"{dev.name}\n")
        else:
            for dev in devices:
                dot, style = _DEVICE_DOTS.get(
                    dev.status.value, ("\u25cf", "white"),
                )
                text.append(f"  {dot} ", style=style)
                text.append(f"{dev.name}\n")
        self.update(text)


class FindingsList(Static):
    """Renders a list of active findings with severity icons."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)

    def update_findings(self, findings: list[Finding]) -> None:
        text = Text()
        text.append("Findings\n", style="bold underline")
        if not findings:
            text.append("  (none)", style="dim")
        for f in findings:
            icon, style = _SEVERITY_ICONS.get(f.severity.value, ("\u2022", "white"))
            text.append(f"  {icon} ", style=style)
            text.append(f"{f.title}\n")
        self.update(text)


class FindingsTable(DataTable):
    """Full-width DataTable showing all active findings."""

    _SEV_LABELS = {"critical": "CRIT", "warning": "WARN", "info": "INFO"}

    def on_mount(self) -> None:
        self.add_columns("Severity", "Device", "Category", "Title", "First Seen", "Status")

    def refresh_findings(self, findings: list[Finding]) -> None:
        """Clear and re-populate the table from the given findings list."""
        self.clear()
        sorted_findings = sorted(
            findings,
            key=lambda f: (_SEVERITY_ORDER.get(f.severity.value, 9), f.last_seen),
        )
        for f in sorted_findings:
            sev = self._SEV_LABELS.get(f.severity.value, f.severity.value.upper())
            status = "RESOLVED" if f.resolved else "ACTIVE"
            first = f.first_seen[:19] if len(f.first_seen) > 19 else f.first_seen
            self.add_row(sev, f.device, f.category, f.title, first, status)


class ChatView(RichLog):
    """Scrollable chat log with convenience methods for message types."""

    def add_user_message(self, text: str) -> None:
        line = Text()
        line.append("user> ", style="bold cyan")
        line.append(text)
        self.write(line)

    def add_agent_response(self, markdown_text: str) -> None:
        label = Text()
        label.append("jace> ", style="bold green")
        self.write(label)
        self.write(Markdown(markdown_text))
        self.write(Text(""))  # blank separator line

    def add_system_message(self, text: str) -> None:
        self.write(Text(text, style="dim"))

    def show_finding_alert(self, finding: Finding, is_new: bool) -> None:
        from jace.ui.notifications import render_finding_panel
        panel = render_finding_panel(finding, is_new)
        self.write(panel)

    def add_approval_prompt(self, command: str, reason: str) -> None:
        """Render a shell approval request in the chat."""
        body = Text()
        body.append("jace wants to run:\n", style="bold yellow")
        body.append(f"  $ {command}\n", style="bold white")
        if reason:
            body.append(f"  Reason: {reason}\n", style="dim")
        body.append("Type ", style="yellow")
        body.append("y", style="bold green")
        body.append(" to approve or ", style="yellow")
        body.append("n", style="bold red")
        body.append(" to deny.", style="yellow")
        panel = Panel(body, border_style="yellow", title="[shell]")
        self.write(panel)

    def add_approval_result(self, approved: bool) -> None:
        """Show the approval decision inline."""
        if approved:
            self.write(Text("  Approved", style="bold green"))
        else:
            self.write(Text("  Denied", style="bold red"))


class ThinkingIndicator(Static):
    """Animated spinner shown while waiting for the agent response."""

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer: Timer | None = None

    def start(self) -> None:
        self._frame = 0
        self._tick()
        self.display = True
        self._timer = self.set_interval(1 / 12, self._tick)

    def stop(self) -> None:
        self.display = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        char = self._FRAMES[self._frame % len(self._FRAMES)]
        self.update(Text(f"  {char} Thinking…", style="dim"))
        self._frame += 1


class ChatInput(Input):
    """Single-line input with placeholder text."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            placeholder="Ask a question or type /help...",
            id="chat-input",
            **kwargs,
        )
