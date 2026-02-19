"""Custom Textual widgets for the JACE TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Input, RichLog, Static

if TYPE_CHECKING:
    from jace.agent.findings import Finding, Severity
    from jace.device.models import DeviceInfo

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
        for dev in devices:
            dot, style = _DEVICE_DOTS.get(dev.status.value, ("\u25cf", "white"))
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


class ChatView(RichLog):
    """Scrollable chat log with convenience methods for message types."""

    def add_user_message(self, text: str) -> None:
        line = Text()
        line.append("jace> ", style="bold cyan")
        line.append(text)
        self.write(line)

    def add_agent_response(self, markdown_text: str) -> None:
        self.write(Markdown(markdown_text))
        self.write(Text(""))  # blank separator line

    def add_system_message(self, text: str) -> None:
        self.write(Text(text, style="dim"))

    def show_finding_alert(self, finding: Finding, is_new: bool) -> None:
        from jace.ui.notifications import render_finding_panel
        panel = render_finding_panel(finding, is_new)
        self.write(panel)


class ChatInput(Input):
    """Single-line input with placeholder text."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            placeholder="Ask a question or type /help...",
            id="chat-input",
            **kwargs,
        )
