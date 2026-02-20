"""Netmiko/SSH driver for Junos devices (fallback)."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from jace.device.base import DeviceDriver
from jace.device.models import CommandResult

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


class NetmikoDriver(DeviceDriver):
    """Junos device driver using Netmiko (SSH)."""

    def __init__(self, host: str, username: str, password: str | None = None,
                 ssh_key: str | None = None, port: int = 22,
                 ssh_config: str | None = None):
        super().__init__(host, username, password, ssh_key, port, ssh_config=ssh_config)
        self._conn = None

    async def connect(self) -> None:
        from netmiko import ConnectHandler

        kwargs: dict = {
            "device_type": "juniper_junos",
            "host": self.host,
            "username": self.username,
            "port": self.port,
        }
        if self.password:
            kwargs["password"] = self.password
        if self.ssh_key:
            kwargs["key_file"] = self.ssh_key
        if self.ssh_config:
            kwargs["ssh_config_file"] = self.ssh_config

        loop = asyncio.get_running_loop()
        self._conn = await loop.run_in_executor(
            _executor, partial(ConnectHandler, **kwargs)
        )
        self._connected = True
        logger.info("Netmiko connected to %s", self.host)

    async def disconnect(self) -> None:
        if self._conn is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_executor, self._conn.disconnect)
            self._connected = False
            self._conn = None
            logger.info("Netmiko disconnected from %s", self.host)

    async def run_command(self, command: str) -> CommandResult:
        if not self._connected or self._conn is None:
            return CommandResult(
                command=command, output="", success=False,
                error="Not connected", driver_used="netmiko",
            )

        loop = asyncio.get_running_loop()
        try:
            output = await loop.run_in_executor(
                _executor,
                partial(self._conn.send_command, command),
            )
            return CommandResult(
                command=command, output=output,
                driver_used="netmiko", success=True,
            )
        except Exception as exc:
            logger.error("Netmiko command failed: %s", exc)
            return CommandResult(
                command=command, output="", success=False,
                error=str(exc), driver_used="netmiko",
            )

    async def get_config(self, section: str | None = None,
                         format: str = "text") -> str:
        cmd = "show configuration"
        if section:
            cmd += f" {section}"
        if format == "set":
            cmd += " | display set"
        result = await self.run_command(cmd)
        return result.output if result.success else f"Error: {result.error}"

    async def get_facts(self) -> dict:
        result = await self.run_command("show version")
        if not result.success:
            return {"error": result.error}
        return {"raw_version": result.output}
