"""Tests for device data models."""

from jace.device.models import CommandResult, DeviceInfo, DeviceStatus


def test_command_result_defaults():
    result = CommandResult(command="show version", output="Junos 21.4R1")
    assert result.success is True
    assert result.error is None
    assert result.driver_used == ""


def test_command_result_failure():
    result = CommandResult(
        command="show version", output="", success=False, error="Connection refused",
    )
    assert result.success is False
    assert result.error == "Connection refused"


def test_device_info_defaults():
    info = DeviceInfo(name="router1", host="10.0.0.1")
    assert info.status == DeviceStatus.DISCONNECTED
    assert info.model == ""
    assert info.last_check is None
