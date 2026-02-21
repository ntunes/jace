"""Tests for MemoryStore — persistent memory files."""

from __future__ import annotations

from pathlib import Path

import pytest

from jace.agent.memory import MemoryStore
from jace.config.settings import MemoryConfig


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path, max_file_size=500, max_total_size=1200)
    s.initialize()
    return s


class TestInitialize:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        assert (tmp_path / "memory" / "devices").is_dir()
        assert (tmp_path / "memory" / "incidents").is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        store.initialize()  # should not raise


class TestDeviceMemory:
    def test_save_and_get(self, store: MemoryStore) -> None:
        store.save_device("mx-01", "Runs Junos 21.4")
        content = store.get_device("mx-01")
        assert "Runs Junos 21.4" in content
        assert "# Device: mx-01" in content

    def test_append(self, store: MemoryStore) -> None:
        store.save_device("mx-01", "First note")
        store.save_device("mx-01", "Second note")
        content = store.get_device("mx-01")
        assert "First note" in content
        assert "Second note" in content

    def test_get_missing_device(self, store: MemoryStore) -> None:
        assert store.get_device("nonexistent") == ""

    def test_get_all_device_names(self, store: MemoryStore) -> None:
        store.save_device("alpha", "note")
        store.save_device("beta", "note")
        names = store.get_all_device_names()
        assert names == ["alpha", "beta"]

    def test_get_all_device_names_empty(self, store: MemoryStore) -> None:
        assert store.get_all_device_names() == []


class TestCategorizedDeviceMemory:
    def test_save_and_get_categorized_device(self, store: MemoryStore) -> None:
        store.save_device("production/mx-01", "Runs Junos 21.4")
        content = store.get_device("production/mx-01")
        assert "Runs Junos 21.4" in content
        assert "# Device: mx-01" in content
        # File should be in nested directory
        path = store._base / "devices" / "production" / "mx-01.md"
        assert path.is_file()

    def test_get_all_device_names_with_categories(self, store: MemoryStore) -> None:
        store.save_device("alpha", "note")
        store.save_device("production/beta", "note")
        store.save_device("lab/gamma", "note")
        names = store.get_all_device_names()
        assert names == ["alpha", "lab/gamma", "production/beta"]

    def test_migrate_device_files(self, store: MemoryStore) -> None:
        """Legacy flat file is migrated to nested category path."""
        # Create a legacy flat file
        flat_path = store._base / "devices" / "mx-01.md"
        flat_path.write_text("# Device: mx-01\n\nLegacy data\n")

        migrated = store.migrate_device_files(["production/mx-01"])
        assert migrated == 1

        # Old file should be gone
        assert not flat_path.exists()
        # New file should exist with same content
        content = store.get_device("production/mx-01")
        assert "Legacy data" in content

    def test_migrate_skips_existing_nested(self, store: MemoryStore) -> None:
        """Migration should not overwrite existing nested files."""
        # Create both flat and nested
        flat_path = store._base / "devices" / "mx-01.md"
        flat_path.write_text("# Device: mx-01\n\nOld data\n")
        store.save_device("production/mx-01", "New data")

        migrated = store.migrate_device_files(["production/mx-01"])
        assert migrated == 0
        # Nested file should keep new data
        content = store.get_device("production/mx-01")
        assert "New data" in content

    def test_migrate_skips_bare_keys(self, store: MemoryStore) -> None:
        """Bare keys (uncategorized) should not be migrated."""
        migrated = store.migrate_device_files(["r1"])
        assert migrated == 0


class TestUserPreferences:
    def test_save_and_get(self, store: MemoryStore) -> None:
        store.save_user_preferences("Prefer terse output")
        content = store.get_user_preferences()
        assert "Prefer terse output" in content

    def test_append(self, store: MemoryStore) -> None:
        store.save_user_preferences("Pref A")
        store.save_user_preferences("Pref B")
        content = store.get_user_preferences()
        assert "Pref A" in content
        assert "Pref B" in content

    def test_get_empty(self, store: MemoryStore) -> None:
        assert store.get_user_preferences() == ""


class TestIncidentMemory:
    def test_save_and_get(self, store: MemoryStore) -> None:
        store.save_incident("bgp-flap-2024-01", "BGP peer went down")
        content = store.get_incident("bgp-flap-2024-01")
        assert "BGP peer went down" in content

    def test_list_incidents(self, store: MemoryStore) -> None:
        store.save_incident("inc-a", "A")
        store.save_incident("inc-b", "B")
        slugs = store.list_incidents()
        assert set(slugs) == {"inc-a", "inc-b"}

    def test_list_incidents_empty(self, store: MemoryStore) -> None:
        assert store.list_incidents() == []

    def test_list_incidents_limit(self, store: MemoryStore) -> None:
        for i in range(5):
            store.save_incident(f"inc-{i}", f"Incident {i}")
        assert len(store.list_incidents(limit=3)) == 3


class TestTruncation:
    def test_truncates_large_file(self, store: MemoryStore) -> None:
        """When content exceeds max_file_size, it should be truncated."""
        big_content = "Line of text\n" * 100  # ~1300 chars
        store.save_device("big", big_content)
        content = store.get_device("big")
        assert len(content) <= store._max_file_size + 50  # allow small buffer


class TestBuildMemoryContext:
    def test_empty_when_no_memory(self, store: MemoryStore) -> None:
        assert store.build_memory_context() == ""

    def test_includes_user_prefs(self, store: MemoryStore) -> None:
        store.save_user_preferences("Always use set format")
        ctx = store.build_memory_context()
        assert "Always use set format" in ctx
        assert "Persistent Memory" in ctx

    def test_includes_device_profiles(self, store: MemoryStore) -> None:
        store.save_device("r1", "Known BGP quirk")
        ctx = store.build_memory_context(device_names=["r1"])
        assert "Known BGP quirk" in ctx

    def test_respects_budget(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, max_file_size=200, max_total_size=300)
        store.initialize()
        store.save_user_preferences("X" * 150)
        store.save_device("r1", "Y" * 150)
        store.save_device("r2", "Z" * 150)
        ctx = store.build_memory_context()
        # Total should not exceed max_total_size + envelope overhead
        assert len(ctx) < 500

    def test_priority_order(self, store: MemoryStore) -> None:
        """User prefs should come before devices, devices before incidents."""
        store.save_user_preferences("USER_PREF")
        store.save_device("r1", "DEVICE_PROF")
        store.save_incident("inc-1", "INCIDENT_REC")
        ctx = store.build_memory_context()
        assert ctx.index("USER_PREF") < ctx.index("DEVICE_PROF")
        assert ctx.index("DEVICE_PROF") < ctx.index("INCIDENT_REC")


class TestMtimeCaching:
    def test_detects_external_edit(self, store: MemoryStore) -> None:
        store.save_device("r1", "Original")
        content1 = store.get_device("r1")
        assert "Original" in content1

        # Simulate external edit
        path = store._base / "devices" / "r1.md"
        path.write_text("# Device: r1\n\nEdited externally\n", encoding="utf-8")

        content2 = store.get_device("r1")
        assert "Edited externally" in content2


class TestGenericInterface:
    def test_save_and_read_device(self, store: MemoryStore) -> None:
        result = store.save("device", "r1", "Quirk noted")
        assert "Saved" in result
        content = store.read("device", "r1")
        assert "Quirk noted" in content

    def test_save_and_read_user(self, store: MemoryStore) -> None:
        store.save("user", "", "Pref X")
        content = store.read("user")
        assert "Pref X" in content

    def test_save_and_read_incident(self, store: MemoryStore) -> None:
        store.save("incident", "bgp-flap", "Details here")
        content = store.read("incident", "bgp-flap")
        assert "Details here" in content

    def test_read_listing(self, store: MemoryStore) -> None:
        store.save("device", "r1", "note")
        store.save("device", "r2", "note")
        listing = store.read("device")
        assert "r1" in listing
        assert "r2" in listing

    def test_unknown_category(self, store: MemoryStore) -> None:
        assert "Unknown" in store.save("bogus", "k", "v")
        assert "Unknown" in store.read("bogus")


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.max_file_size == 8000
        assert cfg.max_total_size == 24000

    def test_custom_values(self) -> None:
        cfg = MemoryConfig(enabled=False, max_file_size=4000, max_total_size=12000)
        assert cfg.enabled is False
        assert cfg.max_file_size == 4000
