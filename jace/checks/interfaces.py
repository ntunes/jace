"""Interface health checks — status, errors, utilization."""

from __future__ import annotations

from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


async def check_interface_status(device_manager: DeviceManager,
                                 device_name: str) -> dict[str, CommandResult]:
    """Check interface status — identify down and admin-down interfaces."""
    results = {}
    results["show interfaces terse"] = await device_manager.run_command(
        device_name, "show interfaces terse"
    )
    return results


async def check_interface_errors(device_manager: DeviceManager,
                                 device_name: str) -> dict[str, CommandResult]:
    """Check interface errors — CRC, input/output errors, drops."""
    results = {}
    results["show interfaces statistics"] = await device_manager.run_command(
        device_name, "show interfaces statistics"
    )
    return results
