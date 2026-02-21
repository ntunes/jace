"""Device data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DriverType(str, Enum):
    PYEZ = "pyez"
    NETMIKO = "netmiko"
    AUTO = "auto"


class DeviceStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


@dataclass
class CommandResult:
    command: str
    output: str
    structured: Any = None  # ET.Element from PyEZ RPC, or dict
    driver_used: str = ""
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeviceInfo:
    name: str
    host: str
    category: str = ""
    model: str = ""
    version: str = ""
    serial: str = ""
    hostname: str = ""
    uptime: str = ""
    status: DeviceStatus = DeviceStatus.DISCONNECTED
    driver_type: str = ""
    last_check: datetime | None = None
    error: str = ""

    @property
    def device_key(self) -> str:
        if self.category:
            return f"{self.category}/{self.name}"
        return self.name


def make_device_key(category: str, name: str) -> str:
    """Build a composite device key: 'category/name' or bare 'name'."""
    if category:
        return f"{category}/{name}"
    return name


def parse_device_key(key: str) -> tuple[str, str]:
    """Split a device key into (category, name).

    Returns ("", name) for bare keys, (category, name) for composite ones.
    """
    if "/" in key:
        category, name = key.split("/", 1)
        return category, name
    return "", key
