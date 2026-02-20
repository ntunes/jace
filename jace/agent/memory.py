"""Persistent memory store — device profiles, user prefs, incident history."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryStore:
    """Manages persistent markdown memory files under a base directory.

    Directory layout::

        <base>/memory/
            user.md              — operator preferences
            devices/<name>.md    — per-device learned profiles
            incidents/<slug>.md  — past incident records
    """

    def __init__(
        self,
        base_path: Path,
        max_file_size: int = 8000,
        max_total_size: int = 24000,
    ) -> None:
        self._base = base_path / "memory"
        self._max_file_size = max_file_size
        self._max_total_size = max_total_size
        # mtime cache: path → (mtime, content)
        self._cache: dict[Path, tuple[float, str]] = {}

    def initialize(self) -> None:
        """Create directory structure."""
        (self._base / "devices").mkdir(parents=True, exist_ok=True)
        (self._base / "incidents").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_device(self, name: str, content: str) -> None:
        """Save or append to a device profile."""
        path = self._base / "devices" / f"{self._sanitize(name)}.md"
        self._append_or_create(path, content, f"# Device: {name}")

    def save_user_preferences(self, content: str) -> None:
        """Save or append to user preferences."""
        path = self._base / "user.md"
        self._append_or_create(path, content, "# User Preferences")

    def save_incident(self, slug: str, content: str) -> None:
        """Save or append to an incident record."""
        path = self._base / "incidents" / f"{self._sanitize(slug)}.md"
        self._append_or_create(path, content, f"# Incident: {slug}")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_device(self, name: str) -> str:
        """Return device profile content, or empty string."""
        path = self._base / "devices" / f"{self._sanitize(name)}.md"
        return self._read_cached(path)

    def get_user_preferences(self) -> str:
        """Return user preferences content, or empty string."""
        return self._read_cached(self._base / "user.md")

    def get_incident(self, slug: str) -> str:
        """Return incident content, or empty string."""
        path = self._base / "incidents" / f"{self._sanitize(slug)}.md"
        return self._read_cached(path)

    # ------------------------------------------------------------------
    # List operations
    # ------------------------------------------------------------------

    def get_all_device_names(self) -> list[str]:
        """Return names of all devices with memory files."""
        devices_dir = self._base / "devices"
        if not devices_dir.exists():
            return []
        return sorted(p.stem for p in devices_dir.glob("*.md"))

    def list_incidents(self, limit: int = 10) -> list[str]:
        """Return slugs of most recent incidents (by mtime)."""
        incidents_dir = self._base / "incidents"
        if not incidents_dir.exists():
            return []
        files = sorted(incidents_dir.glob("*.md"), key=lambda p: p.stat().st_mtime,
                        reverse=True)
        return [p.stem for p in files[:limit]]

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_memory_context(self, device_names: list[str] | None = None) -> str:
        """Assemble a budget-limited memory block for system prompt injection.

        Priority: user prefs → device profiles → recent incidents.
        """
        parts: list[str] = []
        budget = self._max_total_size

        # 1. User preferences
        user_prefs = self.get_user_preferences()
        if user_prefs:
            chunk = self._budget_trim(user_prefs)
            parts.append(chunk)
            budget -= len(chunk)

        # 2. Device profiles
        names = device_names or self.get_all_device_names()
        for name in names:
            if budget <= 0:
                break
            profile = self.get_device(name)
            if profile:
                chunk = self._budget_trim(profile, budget)
                parts.append(chunk)
                budget -= len(chunk)

        # 3. Recent incidents
        for slug in self.list_incidents(limit=5):
            if budget <= 0:
                break
            incident = self.get_incident(slug)
            if incident:
                chunk = self._budget_trim(incident, budget)
                parts.append(chunk)
                budget -= len(chunk)

        if not parts:
            return ""

        return (
            "\n\n--- Persistent Memory ---\n\n"
            + "\n\n".join(parts)
            + "\n\n--- End Memory ---"
        )

    # ------------------------------------------------------------------
    # Generic save/read for tool interface
    # ------------------------------------------------------------------

    def save(self, category: str, key: str, content: str) -> str:
        """Save memory entry by category. Returns confirmation message."""
        if category == "device":
            self.save_device(key, content)
        elif category == "user":
            self.save_user_preferences(content)
        elif category == "incident":
            self.save_incident(key, content)
        else:
            return f"Unknown memory category: {category}"
        return f"Saved to {category}/{key or 'preferences'}."

    def read(self, category: str, key: str | None = None) -> str:
        """Read memory entry by category. Returns content or listing."""
        if category == "device":
            if key:
                content = self.get_device(key)
                return content or f"No memory for device '{key}'."
            names = self.get_all_device_names()
            if not names:
                return "No device memories stored."
            return "Device memories: " + ", ".join(names)
        elif category == "user":
            content = self.get_user_preferences()
            return content or "No user preferences stored."
        elif category == "incident":
            if key:
                content = self.get_incident(key)
                return content or f"No incident record for '{key}'."
            slugs = self.list_incidents()
            if not slugs:
                return "No incident records stored."
            return "Incidents: " + ", ".join(slugs)
        return f"Unknown memory category: {category}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_or_create(self, path: Path, content: str, heading: str) -> None:
        """Append content to file, or create with heading if new."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            new_content = existing.rstrip("\n") + "\n\n" + content.strip() + "\n"
        else:
            new_content = heading + "\n\n" + content.strip() + "\n"

        self._truncate_and_write(path, new_content, heading)

    def _truncate_and_write(self, path: Path, content: str, heading: str) -> None:
        """Write content, truncating to max_file_size if needed."""
        if len(content) <= self._max_file_size:
            path.write_text(content, encoding="utf-8")
        else:
            # Keep heading + most recent entries that fit
            lines = content.split("\n")
            result = heading + "\n\n"
            remaining = self._max_file_size - len(result) - 40  # buffer
            # Take lines from the end until budget exhausted
            kept: list[str] = []
            for line in reversed(lines):
                if remaining - len(line) - 1 < 0:
                    break
                kept.append(line)
                remaining -= len(line) + 1
            result += "(earlier entries truncated)\n\n"
            result += "\n".join(reversed(kept))
            path.write_text(result.rstrip("\n") + "\n", encoding="utf-8")

        # Update cache
        self._cache[path] = (path.stat().st_mtime, path.read_text(encoding="utf-8"))

    def _read_cached(self, path: Path) -> str:
        """Read file with mtime caching."""
        if not path.exists():
            self._cache.pop(path, None)
            return ""

        mtime = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]

        content = path.read_text(encoding="utf-8")
        self._cache[path] = (mtime, content)
        return content

    def _budget_trim(self, text: str, budget: int | None = None) -> str:
        """Trim text to fit within budget (defaults to max_file_size)."""
        limit = min(budget, self._max_file_size) if budget is not None else self._max_file_size
        if len(text) <= limit:
            return text
        return text[:limit - 3] + "..."

    @staticmethod
    def _sanitize(name: str) -> str:
        """Sanitize a name for use as a filename."""
        return re.sub(r"[^\w\-.]", "_", name)
