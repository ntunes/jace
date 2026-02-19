"""Textual-compatible logging handler — routes log records to a RichLog widget."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:
    from textual.app import App
    from textual.widgets import RichLog

# Level tag + Rich style
_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG:    ("DBG", "dim cyan"),
    logging.INFO:     ("INF", "cyan"),
    logging.WARNING:  ("WRN", "yellow"),
    logging.ERROR:    ("ERR", "red"),
    logging.CRITICAL: ("CRT", "bold red"),
}


class TextualLogHandler(logging.Handler):
    """Logging handler that writes formatted records to a Textual RichLog widget.

    Uses ``app.call_from_thread()`` so it is safe to emit from PyEZ / Netmiko
    executor threads.
    """

    def __init__(self, app: App, widget_id: str = "log-panel") -> None:
        super().__init__()
        self._app = app
        self._widget_id = widget_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self._format_record(record)
            self._app.call_from_thread(self._write, text)
        except Exception:
            self.handleError(record)

    def _write(self, text: Text) -> None:
        try:
            widget: RichLog = self._app.query_one(f"#{self._widget_id}")
            widget.write(text)
        except Exception:
            pass

    @staticmethod
    def _format_record(record: logging.LogRecord) -> Text:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        tag, style = _LEVEL_STYLES.get(record.levelno, ("???", ""))
        name = record.name.rsplit(".", 1)[-1]
        msg = record.getMessage()

        text = Text()
        text.append(ts, style="dim")
        text.append(" ")
        text.append(tag, style=style)
        text.append(" ")
        text.append(f"{name}: ", style="dim")
        text.append(msg)
        return text
