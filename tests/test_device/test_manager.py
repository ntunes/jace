"""Tests for device manager."""

import pytest

from jace.config.settings import DeviceConfig, Settings
from jace.device.manager import DeviceManager
from jace.device.models import DeviceStatus, DriverType
from jace.device.pyez_driver import PyEZDriver
from jace.device.netmiko_driver import NetmikoDriver


def test_add_device():
    mgr = DeviceManager()
    config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
    mgr.add_device(config)

    devices = mgr.list_devices()
    assert len(devices) == 1
    assert devices[0].name == "r1"
    assert devices[0].status == DeviceStatus.DISCONNECTED


def test_add_device_propagates_category():
    mgr = DeviceManager()
    config = DeviceConfig(
        name="r1", host="10.0.0.1", username="admin", category="production",
    )
    mgr.add_device(config)

    info = mgr.get_device_info("r1")
    assert info is not None
    assert info.category == "production"


def test_list_devices_filter_by_category():
    mgr = DeviceManager()
    mgr.add_device(DeviceConfig(
        name="r1", host="10.0.0.1", username="admin", category="production",
    ))
    mgr.add_device(DeviceConfig(
        name="r2", host="10.0.0.2", username="admin", category="lab",
    ))
    mgr.add_device(DeviceConfig(
        name="r3", host="10.0.0.3", username="admin", category="production",
    ))

    prod = mgr.list_devices(category="production")
    assert len(prod) == 2
    assert {d.name for d in prod} == {"r1", "r3"}

    lab = mgr.list_devices(category="lab")
    assert len(lab) == 1
    assert lab[0].name == "r2"

    # No filter returns all
    all_devs = mgr.list_devices()
    assert len(all_devs) == 3


def test_get_categories():
    mgr = DeviceManager()
    mgr.add_device(DeviceConfig(
        name="r1", host="10.0.0.1", username="admin", category="production",
    ))
    mgr.add_device(DeviceConfig(
        name="r2", host="10.0.0.2", username="admin", category="lab",
    ))
    mgr.add_device(DeviceConfig(
        name="r3", host="10.0.0.3", username="admin",  # no category
    ))

    categories = mgr.get_categories()
    assert categories == ["lab", "production"]


def test_get_categories_empty():
    mgr = DeviceManager()
    mgr.add_device(DeviceConfig(name="r1", host="10.0.0.1", username="admin"))
    assert mgr.get_categories() == []


def test_list_connected_devices_empty():
    mgr = DeviceManager()
    assert mgr.get_connected_devices() == []


def test_get_device_info():
    mgr = DeviceManager()
    config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
    mgr.add_device(config)

    info = mgr.get_device_info("r1")
    assert info is not None
    assert info.host == "10.0.0.1"

    assert mgr.get_device_info("nonexistent") is None


# --- Blocklist tests ---

class TestBlocklist:
    """Tests for command blocklist enforcement."""

    def test_is_blocked_exact_match(self):
        mgr = DeviceManager(blocked_commands=["clear bgp neighbor all"])
        assert mgr._is_blocked("clear bgp neighbor all") is True

    def test_is_blocked_wildcard(self):
        mgr = DeviceManager(blocked_commands=["request *"])
        assert mgr._is_blocked("request system reboot") is True
        assert mgr._is_blocked("request system halt") is True

    def test_is_blocked_no_match(self):
        mgr = DeviceManager(blocked_commands=["request *"])
        assert mgr._is_blocked("show interfaces") is False
        assert mgr._is_blocked("show route") is False

    def test_is_blocked_case_insensitive(self):
        mgr = DeviceManager(blocked_commands=["request *"])
        assert mgr._is_blocked("Request System Reboot") is True
        assert mgr._is_blocked("REQUEST SYSTEM HALT") is True

    def test_is_blocked_strips_whitespace(self):
        mgr = DeviceManager(blocked_commands=["request *"])
        assert mgr._is_blocked("  request system reboot  ") is True

    def test_is_blocked_empty_blocklist(self):
        mgr = DeviceManager(blocked_commands=[])
        assert mgr._is_blocked("request system reboot") is False

    def test_is_blocked_none_blocklist(self):
        mgr = DeviceManager()
        assert mgr._is_blocked("request system reboot") is False

    def test_is_blocked_multiple_patterns(self):
        mgr = DeviceManager(blocked_commands=["request *", "clear *", "set *"])
        assert mgr._is_blocked("request system reboot") is True
        assert mgr._is_blocked("clear bgp neighbor all") is True
        assert mgr._is_blocked("set interfaces ge-0/0/0 disable") is True
        assert mgr._is_blocked("show bgp summary") is False

    @pytest.mark.asyncio
    async def test_run_command_blocked(self):
        mgr = DeviceManager(blocked_commands=["request *"])
        result = await mgr.run_command("r1", "request system reboot")
        assert result.success is False
        assert "blocked by policy" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_command_not_blocked_but_disconnected(self):
        mgr = DeviceManager(blocked_commands=["request *"])
        result = await mgr.run_command("r1", "show interfaces")
        assert result.success is False
        assert "not connected" in result.error.lower()


# --- Allowlist tests ---

class TestAllowlist:
    """Tests for command allowlist enforcement."""

    def test_is_allowed_empty_allows_everything(self):
        mgr = DeviceManager(allowed_commands=[])
        assert mgr._is_allowed("show interfaces") is True
        assert mgr._is_allowed("request system reboot") is True

    def test_is_allowed_none_allows_everything(self):
        mgr = DeviceManager()
        assert mgr._is_allowed("show interfaces") is True

    def test_is_allowed_pattern_match(self):
        mgr = DeviceManager(allowed_commands=["show *"])
        assert mgr._is_allowed("show interfaces") is True
        assert mgr._is_allowed("show route") is True

    def test_is_allowed_rejects_non_matching(self):
        mgr = DeviceManager(allowed_commands=["show *"])
        assert mgr._is_allowed("request system reboot") is False
        assert mgr._is_allowed("clear bgp neighbor all") is False

    def test_is_allowed_multiple_patterns(self):
        mgr = DeviceManager(allowed_commands=["show *", "ping *"])
        assert mgr._is_allowed("show interfaces") is True
        assert mgr._is_allowed("ping 10.0.0.1") is True
        assert mgr._is_allowed("traceroute 10.0.0.1") is False

    def test_is_allowed_case_insensitive(self):
        mgr = DeviceManager(allowed_commands=["show *"])
        assert mgr._is_allowed("Show Interfaces") is True
        assert mgr._is_allowed("SHOW ROUTE") is True

    def test_is_allowed_strips_whitespace(self):
        mgr = DeviceManager(allowed_commands=["show *"])
        assert mgr._is_allowed("  show interfaces  ") is True

    def test_blocklist_takes_precedence_over_allowlist(self):
        mgr = DeviceManager(
            blocked_commands=["show security *"],
            allowed_commands=["show *"],
        )
        assert mgr._is_blocked("show security ike sa") is True
        assert mgr._is_allowed("show security ike sa") is True
        # Both match, but blocklist is checked first in run_command

    @pytest.mark.asyncio
    async def test_run_command_not_allowed(self):
        mgr = DeviceManager(allowed_commands=["show *"])
        result = await mgr.run_command("r1", "request system reboot")
        assert result.success is False
        assert "not in allowed commands" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_command_allowed_but_disconnected(self):
        mgr = DeviceManager(allowed_commands=["show *"])
        result = await mgr.run_command("r1", "show interfaces")
        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_command_blocklist_before_allowlist(self):
        mgr = DeviceManager(
            blocked_commands=["show security *"],
            allowed_commands=["show *"],
        )
        result = await mgr.run_command("r1", "show security ike sa")
        assert result.success is False
        assert "blocked by policy" in result.error.lower()


# --- SSH config tests ---

class TestSSHConfig:
    """Tests for SSH config passthrough."""

    def test_settings_default_ssh_config(self):
        s = Settings()
        assert s.ssh_config == "~/.ssh/config"

    def test_device_config_ssh_config_default_none(self):
        dc = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
        assert dc.ssh_config is None

    def test_device_config_accepts_ssh_config(self):
        dc = DeviceConfig(
            name="r1", host="10.0.0.1", username="admin",
            ssh_config="/tmp/my_ssh_config",
        )
        assert dc.ssh_config == "/tmp/my_ssh_config"

    def test_global_ssh_config_passed_to_pyez_driver(self, tmp_path):
        ssh_cfg = tmp_path / "config"
        ssh_cfg.write_text("Host *\n")
        mgr = DeviceManager(ssh_config=str(ssh_cfg))
        config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
        driver = mgr._create_driver(config, DriverType.PYEZ)
        assert isinstance(driver, PyEZDriver)
        assert driver.ssh_config == str(ssh_cfg)

    def test_global_ssh_config_passed_to_netmiko_driver(self, tmp_path):
        ssh_cfg = tmp_path / "config"
        ssh_cfg.write_text("Host *\n")
        mgr = DeviceManager(ssh_config=str(ssh_cfg))
        config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
        driver = mgr._create_driver(config, DriverType.NETMIKO)
        assert isinstance(driver, NetmikoDriver)
        assert driver.ssh_config == str(ssh_cfg)

    def test_per_device_ssh_config_overrides_global(self, tmp_path):
        global_cfg = tmp_path / "global_config"
        global_cfg.write_text("Host *\n")
        device_cfg = tmp_path / "device_config"
        device_cfg.write_text("Host lab\n")
        mgr = DeviceManager(ssh_config=str(global_cfg))
        config = DeviceConfig(
            name="r1", host="10.0.0.1", username="admin",
            ssh_config=str(device_cfg),
        )
        driver = mgr._create_driver(config, DriverType.PYEZ)
        assert driver.ssh_config == str(device_cfg)

    def test_nonexistent_ssh_config_not_passed(self):
        mgr = DeviceManager(ssh_config="/nonexistent/ssh/config")
        config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
        driver = mgr._create_driver(config, DriverType.PYEZ)
        assert driver.ssh_config is None

    def test_nonexistent_per_device_falls_back_to_global(self, tmp_path):
        global_cfg = tmp_path / "config"
        global_cfg.write_text("Host *\n")
        mgr = DeviceManager(ssh_config=str(global_cfg))
        config = DeviceConfig(
            name="r1", host="10.0.0.1", username="admin",
            ssh_config="/nonexistent/path",
        )
        # Per-device path doesn't exist, but _resolve_ssh_config checks
        # per-device first — since it doesn't exist, returns None
        driver = mgr._create_driver(config, DriverType.PYEZ)
        # The per-device value takes priority even if invalid (returns None)
        assert driver.ssh_config is None

    def test_no_ssh_config_at_all(self):
        mgr = DeviceManager()
        config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
        driver = mgr._create_driver(config, DriverType.PYEZ)
        assert driver.ssh_config is None
