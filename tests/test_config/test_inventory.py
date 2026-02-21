"""Tests for inventory loading and credential merging."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from jace.config.settings import (
    CredentialConfig,
    DeviceConfig,
    InventoryCategoryConfig,
    InventoryConfig,
    InventoryDeviceConfig,
    ScheduleConfig,
    Settings,
    _load_inventory,
    _merge_device_credentials,
    load_config,
)


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


# --- Inventory loading ---


def test_load_inventory_flattens_devices(tmp_path):
    """Devices from inventory categories appear in settings.devices."""
    inv = {
        "credentials": {
            "creds": {"username": "admin", "password": "secret"},
        },
        "categories": {
            "production": {
                "credentials": "creds",
                "devices": [
                    {"name": "r1", "host": "10.0.0.1"},
                    {"name": "r2", "host": "10.0.0.2"},
                ],
            },
            "lab": {
                "credentials": "creds",
                "devices": [
                    {"name": "r3", "host": "192.168.1.1"},
                ],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    assert len(settings.devices) == 3
    names = {d.name for d in settings.devices}
    assert names == {"r1", "r2", "r3"}


def test_load_inventory_sets_category(tmp_path):
    """Each device gets its category name assigned."""
    inv = {
        "credentials": {},
        "categories": {
            "production": {
                "devices": [{"name": "r1", "host": "10.0.0.1"}],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    assert settings.devices[0].category == "production"


def test_credential_merge_all_layers(tmp_path):
    """Credential merge: defaults → category → device-ref → device-explicit."""
    inv = {
        "credentials": {
            "cat-creds": {"username": "catuser", "password": "catpass"},
            "dev-creds": {"username": "devuser"},
        },
        "categories": {
            "prod": {
                "credentials": "cat-creds",
                "devices": [
                    {
                        "name": "r1",
                        "host": "10.0.0.1",
                        "credentials": "dev-creds",
                        "password": "explicit-pass",
                    },
                ],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    dev = settings.devices[0]
    # username comes from dev-creds (layer 3)
    assert dev.username == "devuser"
    # password comes from explicit device field (layer 4)
    assert dev.password == "explicit-pass"


def test_per_category_schedule_populates_device_schedules(tmp_path):
    """Per-category schedules are stored in device_schedules."""
    inv = {
        "credentials": {},
        "categories": {
            "lab": {
                "schedule": {"chassis": 600, "interfaces": 300},
                "devices": [
                    {"name": "r1", "host": "10.0.0.1"},
                    {"name": "r2", "host": "10.0.0.2"},
                ],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    assert "r1" in settings.device_schedules
    assert "r2" in settings.device_schedules
    assert settings.device_schedules["r1"].chassis == 600
    assert settings.device_schedules["r1"].interfaces == 300
    # Other fields keep ScheduleConfig defaults
    assert settings.device_schedules["r1"].routing == 180


def test_duplicate_device_names_raises(tmp_path):
    """Duplicate device names across categories should raise ValueError."""
    inv = {
        "credentials": {},
        "categories": {
            "cat1": {
                "devices": [{"name": "r1", "host": "10.0.0.1"}],
            },
            "cat2": {
                "devices": [{"name": "r1", "host": "10.0.0.2"}],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    with pytest.raises(ValueError, match="Duplicate device name"):
        _load_inventory(settings, "inventory.yaml", tmp_path)


def test_both_devices_and_inventory_raises(tmp_path):
    """Config with both devices and inventory should raise ValueError."""
    config = {
        "devices": [{"name": "r1", "host": "10.0.0.1"}],
        "inventory": "inventory.yaml",
    }
    _write_yaml(tmp_path / "config.yaml", config)
    _write_yaml(tmp_path / "inventory.yaml", {"credentials": {}, "categories": {}})

    with pytest.raises(ValueError, match="both.*devices.*inventory"):
        load_config(tmp_path / "config.yaml")


def test_missing_credential_ref_raises(tmp_path):
    """Reference to undefined credential set should raise ValueError."""
    inv = {
        "credentials": {},
        "categories": {
            "prod": {
                "credentials": "nonexistent",
                "devices": [{"name": "r1", "host": "10.0.0.1"}],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    with pytest.raises(ValueError, match="Unknown credential reference"):
        _load_inventory(settings, "inventory.yaml", tmp_path)


def test_missing_inventory_file_raises(tmp_path):
    """Missing inventory file should raise ValueError."""
    settings = Settings()
    with pytest.raises(ValueError, match="Inventory file not found"):
        _load_inventory(settings, "nonexistent.yaml", tmp_path)


def test_env_var_expansion_in_inventory(tmp_path, monkeypatch):
    """Environment variables in inventory should be expanded."""
    monkeypatch.setenv("INV_PASSWORD", "expanded-secret")

    inv = {
        "credentials": {
            "creds": {"username": "admin", "password": "${INV_PASSWORD}"},
        },
        "categories": {
            "prod": {
                "credentials": "creds",
                "devices": [{"name": "r1", "host": "10.0.0.1"}],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    assert settings.devices[0].password == "expanded-secret"


def test_backward_compat_flat_devices(tmp_path):
    """Flat devices list without inventory should work unchanged."""
    config = {
        "devices": [
            {"name": "r1", "host": "10.0.0.1", "username": "admin"},
        ],
    }
    _write_yaml(tmp_path / "config.yaml", config)

    settings = load_config(tmp_path / "config.yaml")
    assert len(settings.devices) == 1
    assert settings.devices[0].name == "r1"
    assert settings.devices[0].category == ""
    assert settings.device_schedules == {}


def test_empty_categories(tmp_path):
    """Inventory with no categories should produce empty device list."""
    inv = {
        "credentials": {},
        "categories": {},
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    assert settings.devices == []
    assert settings.device_schedules == {}


def test_inventory_via_load_config(tmp_path):
    """load_config with inventory key should load the inventory file."""
    inv = {
        "credentials": {
            "creds": {"username": "admin", "password": "pass"},
        },
        "categories": {
            "prod": {
                "credentials": "creds",
                "devices": [{"name": "r1", "host": "10.0.0.1"}],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)
    _write_yaml(tmp_path / "config.yaml", {"inventory": "inventory.yaml"})

    settings = load_config(tmp_path / "config.yaml")
    assert len(settings.devices) == 1
    assert settings.devices[0].category == "prod"
    assert settings.devices[0].username == "admin"
    assert settings.devices[0].password == "pass"


def test_per_device_credential_ref(tmp_path):
    """Per-device credentials reference overrides category credentials."""
    inv = {
        "credentials": {
            "cat-creds": {"username": "catuser", "password": "catpass"},
            "special": {"username": "special-user", "password": "special-pass"},
        },
        "categories": {
            "prod": {
                "credentials": "cat-creds",
                "devices": [
                    {
                        "name": "r1",
                        "host": "10.0.0.1",
                        "credentials": "special",
                    },
                ],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    _load_inventory(settings, "inventory.yaml", tmp_path)

    dev = settings.devices[0]
    assert dev.username == "special-user"
    assert dev.password == "special-pass"


def test_missing_per_device_credential_ref_raises(tmp_path):
    """Per-device reference to missing credential set should raise."""
    inv = {
        "credentials": {
            "cat-creds": {"username": "admin"},
        },
        "categories": {
            "prod": {
                "credentials": "cat-creds",
                "devices": [
                    {
                        "name": "r1",
                        "host": "10.0.0.1",
                        "credentials": "does-not-exist",
                    },
                ],
            },
        },
    }
    _write_yaml(tmp_path / "inventory.yaml", inv)

    settings = Settings()
    with pytest.raises(ValueError, match="Unknown credential reference"):
        _load_inventory(settings, "inventory.yaml", tmp_path)


# --- _merge_device_credentials unit tests ---


def test_merge_defaults_only():
    """No category or device creds → DeviceConfig defaults."""
    dev = InventoryDeviceConfig(name="r1", host="10.0.0.1")
    result = _merge_device_credentials(None, {}, dev, "test")
    assert result["username"] == "admin"
    assert result["password"] is None
    assert result["ssh_key"] is None


def test_merge_category_overrides_defaults():
    """Category credentials override defaults."""
    cat_creds = CredentialConfig(username="cat-user", password="cat-pass")
    dev = InventoryDeviceConfig(name="r1", host="10.0.0.1")
    result = _merge_device_credentials(cat_creds, {}, dev, "test")
    assert result["username"] == "cat-user"
    assert result["password"] == "cat-pass"


def test_merge_device_explicit_wins():
    """Explicit device fields override everything."""
    cat_creds = CredentialConfig(username="cat-user", password="cat-pass")
    dev = InventoryDeviceConfig(
        name="r1", host="10.0.0.1", password="explicit",
    )
    result = _merge_device_credentials(cat_creds, {}, dev, "test")
    assert result["username"] == "cat-user"  # from cat
    assert result["password"] == "explicit"  # from device
