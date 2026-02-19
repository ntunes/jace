"""Configuration audit checks — security, best practices, unused items."""

from __future__ import annotations

from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


async def audit_security(device_manager: DeviceManager,
                         device_name: str) -> dict[str, CommandResult]:
    """Audit security settings — insecure crypto, open management, etc."""
    results = {}
    # Get relevant config sections for security audit
    config = await device_manager.get_config(device_name, section=None, format="set")
    results["configuration (set format)"] = CommandResult(
        command="show configuration | display set",
        output=config,
        driver_used="config-audit",
        success=bool(config and not config.startswith("Error")),
    )
    return results


async def audit_best_practices(device_manager: DeviceManager,
                               device_name: str) -> dict[str, CommandResult]:
    """Audit best practices — syslog, NTP, SNMP community strings."""
    results = {}
    config = await device_manager.get_config(device_name, section=None, format="text")
    results["configuration (text format)"] = CommandResult(
        command="show configuration",
        output=config,
        driver_used="config-audit",
        success=bool(config and not config.startswith("Error")),
    )
    return results
