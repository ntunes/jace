"""Heartbeat file manager — reads/writes user-defined monitoring instructions."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """Manages the heartbeat instructions file with mtime-based change detection."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._cached_content: str = ""
        self._cached_mtime: float = 0.0

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> str:
        """Read file and update cached mtime. Returns content or empty string."""
        if not self._path.exists():
            self._cached_content = ""
            self._cached_mtime = 0.0
            return ""
        self._cached_content = self._path.read_text(encoding="utf-8")
        self._cached_mtime = self._path.stat().st_mtime
        return self._cached_content

    def has_changed(self) -> bool:
        """Check if the file was modified since last load."""
        if not self._path.exists():
            return self._cached_mtime != 0.0
        return self._path.stat().st_mtime != self._cached_mtime

    def get_instructions(self) -> str:
        """Return cached content, re-reading if file changed on disk."""
        if self.has_changed():
            logger.info("Heartbeat file changed on disk, reloading")
            self.load()
        elif not self._cached_content:
            self.load()
        return self._cached_content

    def _get_lines(self) -> list[str]:
        """Parse instruction lines (non-empty, non-comment lines)."""
        content = self.get_instructions()
        return [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def add_instruction(self, instruction: str) -> str:
        """Append an instruction line and write file. Returns updated content."""
        content = self.get_instructions()
        line = instruction.strip()
        if not line.startswith("- "):
            line = f"- {line}"
        if content and not content.endswith("\n"):
            content += "\n"
        content += line + "\n"
        self._write(content)
        return self.list_instructions()

    def remove_instruction(self, index: int) -> str:
        """Remove instruction by 1-based index. Returns updated content."""
        lines = self._get_lines()
        if index < 1 or index > len(lines):
            return f"Invalid index {index}. There are {len(lines)} instruction(s)."
        target = lines[index - 1]
        # Remove the first occurrence of the target line from the full content
        all_lines = self.get_instructions().splitlines(keepends=True)
        new_lines: list[str] = []
        removed = False
        for raw in all_lines:
            if not removed and raw.strip() == target.strip():
                removed = True
                continue
            new_lines.append(raw)
        self._write("".join(new_lines))
        return self.list_instructions()

    def replace_instructions(self, content: str) -> str:
        """Overwrite file with new content. Returns updated content."""
        self._write(content)
        return self.list_instructions()

    def list_instructions(self) -> str:
        """Return numbered list of current instructions."""
        lines = self._get_lines()
        if not lines:
            return "No heartbeat instructions configured."
        parts: list[str] = []
        for i, line in enumerate(lines, 1):
            # Strip leading "- " for display
            text = line.lstrip("- ").strip()
            parts.append(f"{i}. {text}")
        return "\n".join(parts)

    def _write(self, content: str) -> None:
        """Write content to file and update cache."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(content, encoding="utf-8")
        self._cached_content = content
        self._cached_mtime = self._path.stat().st_mtime
