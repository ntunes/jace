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
