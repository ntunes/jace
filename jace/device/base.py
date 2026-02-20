"""Abstract device driver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jace.device.models import CommandResult


class DeviceDriver(ABC):
    """Base class for device connectivity drivers."""

    def __init__(self, host: str, username: str, password: str | None = None,
                 ssh_key: str | None = None, port: int = 830,
                 ssh_config: str | None = None):
        self.host = host
        self.username = username
        self.password = password
        self.ssh_key = ssh_key
        self.port = port
        self.ssh_config = ssh_config
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the device."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def run_command(self, command: str) -> CommandResult:
        """Execute an operational command and return the result."""

    @abstractmethod
    async def get_config(self, section: str | None = None,
                         format: str = "text") -> str:
        """Retrieve device configuration."""

    @abstractmethod
    async def get_facts(self) -> dict:
        """Get device facts (model, version, serial, uptime, etc.)."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def driver_name(self) -> str:
        return self.__class__.__name__
