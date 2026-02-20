"""Device manager — connection pool and command dispatch."""

from __future__ import annotations

import logging
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from jace.config.settings import DeviceConfig
from jace.device.base import DeviceDriver
from jace.device.models import CommandResult, DeviceInfo, DeviceStatus, DriverType
from jace.device.netmiko_driver import NetmikoDriver
from jace.device.pyez_driver import PyEZDriver

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages connections to multiple Junos devices."""

    def __init__(
        self,
        blocked_commands: list[str] | None = None,
        allowed_commands: list[str] | None = None,
        ssh_config: str | None = None,
    ) -> None:
        self._devices: dict[str, DeviceConfig] = {}
        self._drivers: dict[str, DeviceDriver] = {}
        self._fallback_drivers: dict[str, DeviceDriver] = {}
        self._info: dict[str, DeviceInfo] = {}
        self._blocked_commands = [p.strip().lower() for p in (blocked_commands or [])]
        self._allowed_commands = [p.strip().lower() for p in (allowed_commands or [])]
        self._ssh_config = ssh_config

    def add_device(self, config: DeviceConfig) -> None:
        self._devices[config.name] = config
        self._info[config.name] = DeviceInfo(
            name=config.name, host=config.host,
        )

    async def connect_all(self) -> None:
        for name in self._devices:
            await self.connect(name)

    async def connect(self, device_name: str) -> bool:
        config = self._devices.get(device_name)
        if config is None:
            logger.error("Unknown device: %s", device_name)
            return False

        info = self._info[device_name]
        info.status = DeviceStatus.CONNECTING
        driver_type = DriverType(config.driver) if config.driver != "auto" else DriverType.AUTO

        # Primary driver
        primary = self._create_driver(config, driver_type)
        try:
            await primary.connect()
            self._drivers[device_name] = primary
            info.status = DeviceStatus.CONNECTED
            info.driver_type = primary.driver_name

            # Fetch facts
            try:
                facts = await primary.get_facts()
                info.model = str(facts.get("model", ""))
                info.version = str(facts.get("version", ""))
                info.serial = str(facts.get("serialnumber", ""))
                info.hostname = str(facts.get("hostname", ""))
                info.uptime = str(facts.get("RE0", {}).get("up_time", "")) if isinstance(facts.get("RE0"), dict) else ""
            except Exception:
                pass

            logger.info("Connected to %s via %s", device_name, primary.driver_name)

            # Set up fallback if using PyEZ
            if isinstance(primary, PyEZDriver):
                try:
                    ssh_cfg = self._resolve_ssh_config(config)
                    fallback = NetmikoDriver(
                        host=config.host, username=config.username,
                        password=config.password, ssh_key=config.ssh_key,
                        port=22, ssh_config=ssh_cfg,
                    )
                    await fallback.connect()
                    self._fallback_drivers[device_name] = fallback
                    logger.info("Netmiko fallback ready for %s", device_name)
                except Exception as exc:
                    logger.debug("Netmiko fallback not available for %s: %s", device_name, exc)

            return True
        except Exception as exc:
            logger.error("Failed to connect to %s: %s", device_name, exc)
            info.status = DeviceStatus.ERROR
            return False

    async def disconnect_all(self) -> None:
        for name in list(self._drivers):
            await self.disconnect(name)

    async def disconnect(self, device_name: str) -> None:
        driver = self._drivers.pop(device_name, None)
        if driver:
            await driver.disconnect()

        fallback = self._fallback_drivers.pop(device_name, None)
        if fallback:
            await fallback.disconnect()

        if device_name in self._info:
            self._info[device_name].status = DeviceStatus.DISCONNECTED

    def _is_blocked(self, command: str) -> bool:
        """Check if a command matches any blocked pattern."""
        normalized = command.strip().lower()
        return any(fnmatch(normalized, pattern) for pattern in self._blocked_commands)

    def _is_allowed(self, command: str) -> bool:
        """Check if a command matches the allowlist. Returns True if allowlist is empty."""
        if not self._allowed_commands:
            return True
        normalized = command.strip().lower()
        return any(fnmatch(normalized, pattern) for pattern in self._allowed_commands)

    async def run_command(self, device_name: str, command: str) -> CommandResult:
        """Run a command on a device, with fallback to Netmiko if PyEZ fails."""
        if self._is_blocked(command):
            logger.warning("Blocked command: %s", command)
            return CommandResult(
                command=command, output="", success=False,
                error=f"Command blocked by policy: {command}",
            )

        if not self._is_allowed(command):
            logger.warning("Command not in allowed list: %s", command)
            return CommandResult(
                command=command, output="", success=False,
                error=f"Command not in allowed commands: {command}",
            )

        driver = self._drivers.get(device_name)
        if driver is None:
            return CommandResult(
                command=command, output="", success=False,
                error=f"Device '{device_name}' not connected",
            )

        result = await driver.run_command(command)

        # Try fallback if primary failed
        if not result.success and device_name in self._fallback_drivers:
            logger.info("Primary driver failed, trying Netmiko fallback for %s", device_name)
            result = await self._fallback_drivers[device_name].run_command(command)

        if device_name in self._info:
            self._info[device_name].last_check = datetime.now()

        return result

    async def get_config(self, device_name: str, section: str | None = None,
                         format: str = "text") -> str:
        driver = self._drivers.get(device_name)
        if driver is None:
            return f"Error: Device '{device_name}' not connected"
        return await driver.get_config(section, format)

    async def get_facts(self, device_name: str) -> dict:
        driver = self._drivers.get(device_name)
        if driver is None:
            return {"error": f"Device '{device_name}' not connected"}
        return await driver.get_facts()

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._info.values())

    def get_device_info(self, device_name: str) -> DeviceInfo | None:
        return self._info.get(device_name)

    def get_connected_devices(self) -> list[str]:
        return [name for name, d in self._drivers.items() if d.is_connected]

    def _resolve_ssh_config(self, config: DeviceConfig) -> str | None:
        """Resolve SSH config path: per-device override > global default.

        Returns the expanded path if the file exists, otherwise None.
        """
        raw = config.ssh_config or self._ssh_config
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path)
        return None

    def _create_driver(self, config: DeviceConfig, driver_type: DriverType) -> DeviceDriver:
        ssh_cfg = self._resolve_ssh_config(config)
        if driver_type == DriverType.NETMIKO:
            return NetmikoDriver(
                host=config.host, username=config.username,
                password=config.password, ssh_key=config.ssh_key,
                port=22, ssh_config=ssh_cfg,
            )
        # Default to PyEZ (for AUTO and PYEZ)
        return PyEZDriver(
            host=config.host, username=config.username,
            password=config.password, ssh_key=config.ssh_key,
            port=config.port, ssh_config=ssh_cfg,
        )
