"""Tests for device manager."""

import pytest

from jace.config.settings import DeviceConfig
from jace.device.manager import DeviceManager
from jace.device.models import DeviceStatus


def test_add_device():
    mgr = DeviceManager()
    config = DeviceConfig(name="r1", host="10.0.0.1", username="admin")
    mgr.add_device(config)

    devices = mgr.list_devices()
    assert len(devices) == 1
    assert devices[0].name == "r1"
    assert devices[0].status == DeviceStatus.DISCONNECTED


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
