"""Alert rendering and formatting for terminal output."""

from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from jace.agent.findings import Finding, Severity

SEVERITY_STYLES = {
    Severity.CRITICAL: ("bold white on red", "CRITICAL"),
    Severity.WARNING: ("bold black on yellow", "WARNING"),
    Severity.INFO: ("bold white on blue", "INFO"),
}


def _format_timestamp(iso: str) -> str:
    """Format an ISO timestamp to a human-friendly string."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return iso


def render_finding_panel(finding: Finding, is_new: bool) -> Panel:
    """Build a Rich Panel renderable for a finding."""
    style, label = SEVERITY_STYLES.get(
        finding.severity, ("", finding.severity.value.upper())
    )

    status = "NEW" if is_new else ("RESOLVED" if finding.resolved else "UPDATED")

    title = Text()
    title.append(f" {label} ", style=style)
    title.append(f" [{status}] ", style="bold")
    title.append(f" {finding.device} ", style="dim")

    body = Text()
    body.append(finding.title, style="bold")
    body.append("\n\n")
    body.append(finding.detail)
    if finding.recommendation:
        body.append("\n\n")
        body.append("Recommendation: ", style="bold cyan")
        body.append(finding.recommendation)

    # Timestamps
    first = _format_timestamp(finding.first_seen)
    last = _format_timestamp(finding.last_seen)
    body.append("\n\n")
    body.append("First detected: ", style="dim bold")
    body.append(first, style="dim")
    if first != last:
        body.append("  Last updated: ", style="dim bold")
        body.append(last, style="dim")

    border_style = {
        Severity.CRITICAL: "red",
        Severity.WARNING: "yellow",
        Severity.INFO: "blue",
    }.get(finding.severity, "white")

    return Panel(
        body,
        title=title,
        border_style=border_style,
        padding=(0, 1),
    )


def render_finding(console: Console, finding: Finding, is_new: bool) -> None:
    """Render a finding as a Rich panel (legacy console-printing wrapper)."""
    console.print()
    console.print(render_finding_panel(finding, is_new))


def render_findings_summary(console: Console, findings: list[Finding]) -> None:
    """Render a summary of all active findings."""
    if not findings:
        console.print("[green]No active findings.[/green]")
        return

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

    console.print(Panel(summary, border_style="dim"))


def finding_toast_params(finding: Finding, is_new: bool) -> dict:
    """Return kwargs suitable for ``App.notify(**params)``.

    Maps finding severity to Textual toast severity/timeout so the caller
    can simply do ``self.notify(**finding_toast_params(f, True))``.
    """
    sev = finding.severity
    if sev == Severity.CRITICAL:
        toast_severity = "error"
        timeout = 10
    elif sev == Severity.WARNING:
        toast_severity = "warning"
        timeout = 8
    else:
        toast_severity = "information"
        timeout = 5

    status = "NEW" if is_new else ("RESOLVED" if finding.resolved else "UPDATED")
    label = sev.value.upper()
    title = f"{label} on {finding.device}"
    message = f"[bold]{finding.title}[/bold] ({status})"

    return {
        "message": message,
        "title": title,
        "severity": toast_severity,
        "timeout": timeout,
    }


def format_status_bar(devices_connected: int, findings_count: int,
                      critical_count: int) -> str:
    """Format the status bar text."""
    parts = [f"Devices: {devices_connected}"]
    if critical_count > 0:
        parts.append(f"[bold red]Findings: {findings_count} ({critical_count} critical)[/bold red]")
    elif findings_count > 0:
        parts.append(f"[yellow]Findings: {findings_count}[/yellow]")
    else:
        parts.append("[green]Findings: 0[/green]")
    return " | ".join(parts)
