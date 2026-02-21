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
    # Path helpers
    # ------------------------------------------------------------------

    def _device_path(self, name: str) -> Path:
        """Build the path for a device memory file.

        Composite keys (``"category/device"``) produce a nested path:
        ``devices/<category>/<device>.md``.  Bare names produce the flat
        path: ``devices/<name>.md``.
        """
        if "/" in name:
            category, bare = name.split("/", 1)
            return (
                self._base / "devices"
                / self._sanitize(category)
                / f"{self._sanitize(bare)}.md"
            )
        return self._base / "devices" / f"{self._sanitize(name)}.md"

    def _device_heading(self, name: str) -> str:
        """Return the heading for a device memory file."""
        if "/" in name:
            _, bare = name.split("/", 1)
            return f"# Device: {bare}"
        return f"# Device: {name}"

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def migrate_device_files(self, device_keys: list[str]) -> int:
        """Move legacy flat device files into category subdirectories.

        For each composite key (``"category/name"``), if the flat file
        ``devices/<name>.md`` exists but the nested file does not, move it.

        Returns the number of files migrated.
        """
        migrated = 0
        for key in device_keys:
            if "/" not in key:
                continue
            _, bare = key.split("/", 1)
            legacy_path = self._base / "devices" / f"{self._sanitize(bare)}.md"
            new_path = self._device_path(key)
            if legacy_path.is_file() and not new_path.is_file():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path.rename(new_path)
                self._cache.pop(legacy_path, None)
                migrated += 1
                logger.info("Migrated device memory %s → %s", legacy_path, new_path)
        return migrated

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_device(self, name: str, content: str) -> None:
        """Save or append to a device profile."""
        path = self._device_path(name)
        self._append_or_create(path, content, self._device_heading(name))

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
        path = self._device_path(name)
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
        """Return names of all devices with memory files.

        Returns composite keys (``"category/name"``) for files in
        subdirectories and bare names for files directly under
        ``devices/``.
        """
        devices_dir = self._base / "devices"
        if not devices_dir.exists():
            return []
        names: list[str] = []
        for p in devices_dir.rglob("*.md"):
            rel = p.relative_to(devices_dir)
            parts = rel.parts
            if len(parts) == 1:
                # Flat file: devices/<name>.md
                names.append(p.stem)
            elif len(parts) == 2:
                # Nested: devices/<category>/<name>.md
                names.append(f"{parts[0]}/{p.stem}")
        return sorted(names)

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
