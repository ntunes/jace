"""Fixed log panel at the bottom of the terminal using ANSI scroll regions."""

from __future__ import annotations

import logging
import os
import signal
import sys
from collections import deque
from datetime import datetime

from rich.console import Console

logger = logging.getLogger(__name__)

# ANSI color codes for log levels
_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG:    ("DBG", "\033[36m"),       # cyan
    logging.INFO:     ("INF", "\033[36m"),       # cyan
    logging.WARNING:  ("WRN", "\033[33m"),       # yellow
    logging.ERROR:    ("ERR", "\033[31m"),       # red
    logging.CRITICAL: ("CRT", "\033[1;31m"),     # bold red
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_SEPARATOR_CHAR = "\u2500"  # ─


class LogPanel:
    """Persistent log panel at the bottom of the terminal.

    Uses ANSI DECSTBM escape sequences to create a fixed region at the
    bottom of the terminal where log messages are displayed, while the
    main shell output scrolls independently above.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._active = False
        self._panel_rows = 0
        self._msg_rows = 0
        self._term_height = 0
        self._term_width = 0
        self._messages: deque[tuple[str, int]] = deque()
        self._original_handlers: list[logging.Handler] = []
        self._handler: _LogPanelHandler | None = None
        self._prev_sigwinch = None

    @property
    def is_active(self) -> bool:
        return self._active

    def setup(self) -> None:
        """Install the log panel: set scroll region, add logging handler."""
        if not sys.stdout.isatty():
            return

        self._calculate_dimensions()
        self._messages = deque(maxlen=self._msg_rows)
        self._set_scroll_region()
        self._redraw()

        # Install logging handler
        self._handler = _LogPanelHandler(self)
        root = logging.getLogger()

        # Remove existing StreamHandlers on stderr so logs don't double-print
        self._original_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, _LogPanelHandler)
        ]
        for h in self._original_handlers:
            root.removeHandler(h)

        root.addHandler(self._handler)

        # Listen for terminal resize
        self._prev_sigwinch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._handle_resize)

        self._active = True

    def teardown(self) -> None:
        """Remove log panel: restore scroll region and logging handlers."""
        if not self._active:
            return
        self._active = False

        # Remove our handler, restore originals
        root = logging.getLogger()
        if self._handler:
            root.removeHandler(self._handler)
            self._handler = None
        for h in self._original_handlers:
            root.addHandler(h)
        self._original_handlers = []

        # Restore SIGWINCH
        if self._prev_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._prev_sigwinch)
            self._prev_sigwinch = None

        # Reset scroll region to full terminal and clear the panel area
        sys.stdout.write("\033[r")  # reset scroll region
        sys.stdout.write(f"\033[{self._term_height - self._panel_rows + 1};1H")
        sys.stdout.write("\033[J")  # clear from cursor to end
        sys.stdout.write(f"\033[{self._term_height - self._panel_rows};1H")
        sys.stdout.flush()

    def restore_after_clear(self) -> None:
        """Re-establish scroll region and redraw after /clear."""
        if not self._active:
            return
        self._calculate_dimensions()
        self._set_scroll_region()
        self._redraw()

    def add_message(self, formatted: str, levelno: int) -> None:
        """Add a formatted log message and redraw the panel."""
        self._messages.append((formatted, levelno))
        if self._active:
            self._redraw()

    def _calculate_dimensions(self) -> None:
        """Calculate panel size based on current terminal dimensions."""
        try:
            size = os.get_terminal_size()
            self._term_height = size.lines
            self._term_width = size.columns
        except OSError:
            self._term_height = 24
            self._term_width = 80

        self._panel_rows = min(10, max(3, self._term_height // 4))
        self._msg_rows = self._panel_rows - 1  # one row for separator

    def _set_scroll_region(self) -> None:
        """Set the DECSTBM scroll region to exclude the panel area."""
        scroll_end = self._term_height - self._panel_rows
        # DECSTBM: set scrolling region to rows 1..scroll_end
        sys.stdout.write(f"\033[1;{scroll_end}r")
        # Move cursor into scroll region
        sys.stdout.write(f"\033[{scroll_end};1H")
        sys.stdout.flush()

    def _redraw(self) -> None:
        """Redraw the fixed panel area at the bottom of the terminal."""
        if not sys.stdout.isatty():
            return

        # Synchronize with Rich output
        lock = getattr(self._console, "_lock", None)
        if lock:
            lock.acquire()
        try:
            sep_row = self._term_height - self._panel_rows + 1

            # Save cursor position (DECSC)
            buf = ["\0337"]

            # Draw separator line
            sep_line = _SEPARATOR_CHAR * self._term_width
            buf.append(f"\033[{sep_row};1H")
            buf.append(f"{_DIM}{sep_line}{_RESET}")

            # Draw message rows — show the most recent messages
            msgs = list(self._messages)
            for i in range(self._msg_rows):
                row = sep_row + 1 + i
                buf.append(f"\033[{row};1H")
                buf.append("\033[2K")  # clear entire line
                idx = len(msgs) - self._msg_rows + i
                if idx >= 0:
                    text, _levelno = msgs[idx]
                    buf.append(text[:self._term_width])

            # Restore cursor position (DECRC)
            buf.append("\0338")

            sys.stdout.write("".join(buf))
            sys.stdout.flush()
        finally:
            if lock:
                lock.release()

    def _handle_resize(self, signum: int, frame: object) -> None:
        """Handle terminal resize: recalculate and redraw."""
        old_msg_rows = self._msg_rows
        self._calculate_dimensions()

        # Rebuild deque with new maxlen
        if self._msg_rows != old_msg_rows:
            old_msgs = list(self._messages)
            self._messages = deque(old_msgs, maxlen=self._msg_rows)

        self._set_scroll_region()
        self._redraw()

        # Chain to previous handler if any
        if callable(self._prev_sigwinch):
            self._prev_sigwinch(signum, frame)


class _LogPanelHandler(logging.Handler):
    """Logging handler that routes records to the LogPanel."""

    def __init__(self, panel: LogPanel) -> None:
        super().__init__()
        self._panel = panel

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            level_tag, level_color = _LEVEL_STYLES.get(
                record.levelno, ("???", "")
            )
            name = record.name.split(".")[-1]  # short module name
            msg = record.getMessage()
            formatted = (
                f"{_DIM}{ts}{_RESET} "
                f"{level_color}{level_tag}{_RESET} "
                f"{_DIM}{name}:{_RESET} {msg}"
            )
            self._panel.add_message(formatted, record.levelno)
        except Exception:
            self.handleError(record)
