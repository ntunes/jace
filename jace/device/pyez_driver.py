"""PyEZ/NETCONF driver for Junos devices."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from jace.device.base import DeviceDriver
from jace.device.models import CommandResult

logger = logging.getLogger(__name__)

try:
    from lxml import etree as lxml_etree
except ImportError:
    lxml_etree = None  # type: ignore[assignment]

# Maps CLI commands to PyEZ RPC method names and kwargs
RPC_MAP: dict[str, tuple[str, dict]] = {
    "show chassis alarms":              ("get_alarm_information", {}),
    "show system alarms":               ("get_system_alarm_information", {}),
    "show chassis routing-engine":      ("get_route_engine_information", {}),
    "show chassis environment":         ("get_environment_information", {}),
    "show chassis fpc":                 ("get_fpc_information", {}),
    "show chassis hardware":            ("get_chassis_inventory", {}),
    "show interfaces terse":            ("get_interface_information", {"terse": True}),
    "show interfaces extensive":        ("get_interface_information", {"extensive": True}),
    "show interfaces statistics":       ("get_interface_information", {"statistics": True}),
    "show bgp summary":                 ("get_bgp_summary_information", {}),
    "show bgp neighbor":                ("get_bgp_neighbor_information", {}),
    "show ospf neighbor":               ("get_ospf_neighbor_information", {}),
    "show ospf interface":              ("get_ospf_interface_information", {}),
    "show isis adjacency":              ("get_isis_adjacency_information", {}),
    "show route summary":               ("get_route_summary_information", {}),
    "show system storage":              ("get_system_storage", {}),
    "show system processes extensive":  ("get_system_process_information", {}),
    "show system uptime":               ("get_system_uptime_information", {}),
    "show system memory":               ("get_system_memory_information", {}),
    "show pfe statistics exceptions":   ("get_pfe_statistics", {}),
    "show configuration":               ("get_config", {}),
    "show version":                     ("get_software_information", {}),
}

_executor = ThreadPoolExecutor(max_workers=4)


def _xml_to_str(element: object) -> str:
    """Convert an XML element to a pretty-printed string."""
    if lxml_etree is not None:
        return lxml_etree.tostring(element, encoding="unicode", pretty_print=True)
    from xml.etree import ElementTree as ET
    return ET.tostring(element, encoding="unicode")


def _extract_config_text(element: object, format: str) -> str:
    """Extract config content from a PyEZ get_config RPC response.

    PyEZ always returns an lxml element.  For ``format="text"`` the actual
    curly-brace config is inside ``<configuration-text>``, and for
    ``format="set"`` it is inside ``<configuration-set>``.  Only for
    ``format="xml"`` do we serialise the XML tree directly.
    """
    if format == "xml" or not hasattr(element, 'tag'):
        if hasattr(element, 'tag'):
            return _xml_to_str(element)
        return str(element)

    # text / set — look for the text payload inside the wrapper element
    tag = "configuration-text" if format == "text" else "configuration-set"
    if hasattr(element, 'find'):
        child = element.find(f".//{tag}")  # type: ignore[union-attr]
        if child is not None and child.text:
            return child.text

    # Some PyEZ versions put the text directly on the root element
    if hasattr(element, 'text') and element.text and element.text.strip():  # type: ignore[union-attr]
        return element.text  # type: ignore[union-attr]

    # Last resort — serialise as XML so we return *something*
    return _xml_to_str(element)


class PyEZDriver(DeviceDriver):
    """Junos device driver using PyEZ (NETCONF)."""

    def __init__(self, host: str, username: str, password: str | None = None,
                 ssh_key: str | None = None, port: int = 830,
                 ssh_config: str | None = None, timeout: int = 30):
        super().__init__(host, username, password, ssh_key, port,
                         ssh_config=ssh_config, timeout=timeout)
        self._dev = None

    async def connect(self) -> None:
        from jnpr.junos import Device as JunosDevice

        kwargs: dict = {
            "host": self.host,
            "user": self.username,
            "port": self.port,
            "conn_open_timeout": self.timeout,
            "auto_probe": 0,
        }
        if self.password:
            kwargs["passwd"] = self.password
        if self.ssh_key:
            kwargs["ssh_private_key_file"] = self.ssh_key
        if self.ssh_config:
            kwargs["ssh_config"] = self.ssh_config

        loop = asyncio.get_running_loop()
        self._dev = JunosDevice(**kwargs)
        await loop.run_in_executor(_executor, self._dev.open)
        self._dev.timeout = self.timeout
        self._connected = True
        logger.info("PyEZ connected to %s", self.host)

    async def disconnect(self) -> None:
        if self._dev is not None:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(_executor, self._dev.close)
            except Exception as exc:
                logger.warning("PyEZ close error for %s: %s", self.host, exc)
            self._connected = False
            self._dev = None
            logger.info("PyEZ disconnected from %s", self.host)

    async def run_command(self, command: str) -> CommandResult:
        if not self._connected or self._dev is None:
            return CommandResult(
                command=command, output="", success=False,
                error="Not connected", driver_used="pyez",
            )

        loop = asyncio.get_running_loop()
        normalized = command.strip().lower()

        # Check for interface-specific commands
        rpc_entry = self._match_rpc(normalized)

        if rpc_entry is not None:
            rpc_name, rpc_kwargs = rpc_entry
            try:
                rpc_func = getattr(self._dev.rpc, rpc_name)
                result = await loop.run_in_executor(
                    _executor, partial(rpc_func, **rpc_kwargs)
                )
                if hasattr(result, 'tag'):
                    output_str = _xml_to_str(result)
                    structured = result
                else:
                    output_str = str(result)
                    structured = None
                return CommandResult(
                    command=command, output=output_str,
                    structured=structured,
                    driver_used="pyez", success=True,
                )
            except Exception as exc:
                logger.warning("PyEZ RPC %s failed: %s", rpc_name, exc)
                return CommandResult(
                    command=command, output="", success=False,
                    error=str(exc), driver_used="pyez",
                )

        # Fallback: use PyEZ cli() for unrecognized commands
        try:
            result = await loop.run_in_executor(
                _executor, partial(self._dev.cli, command, warning=False)
            )
            return CommandResult(
                command=command, output=result,
                driver_used="pyez-cli", success=True,
            )
        except Exception as exc:
            return CommandResult(
                command=command, output="", success=False,
                error=str(exc), driver_used="pyez-cli",
            )

    async def get_config(self, section: str | None = None,
                         format: str = "text") -> str:
        if not self._connected or self._dev is None:
            return ""
        loop = asyncio.get_running_loop()
        try:
            if section:
                cmd = f"show configuration {section}"
                if format == "set":
                    cmd += " | display set"
                result = await loop.run_in_executor(
                    _executor,
                    partial(self._dev.cli, cmd, warning=False),
                )
                return str(result)
            options: dict = {"format": format}
            result = await loop.run_in_executor(
                _executor,
                partial(self._dev.rpc.get_config, **options),
            )
            return _extract_config_text(result, format)
        except Exception as exc:
            logger.error("PyEZ get_config failed: %s", exc)
            return f"Error: {exc}"

    async def get_facts(self) -> dict:
        if not self._connected or self._dev is None:
            return {}
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(_executor, self._dev.facts_refresh)
            return dict(self._dev.facts)
        except Exception as exc:
            logger.error("PyEZ get_facts failed: %s", exc)
            return {"error": str(exc)}

    def _match_rpc(self, command: str) -> tuple[str, dict] | None:
        """Match a CLI command to an RPC entry, handling parameterized commands."""
        if command in RPC_MAP:
            return RPC_MAP[command]

        # Handle "show interfaces <name> terse" etc.
        for pattern, entry in RPC_MAP.items():
            if command.startswith(pattern):
                return entry

        return None
