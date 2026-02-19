"""Tests for device manager."""

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
