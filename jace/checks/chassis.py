"""Chassis health checks — alarms, environment, FPC status."""

from __future__ import annotations

from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


async def check_alarms(device_manager: DeviceManager,
                       device_name: str) -> dict[str, CommandResult]:
    """Check chassis and system alarms."""
    results = {}
    results["show chassis alarms"] = await device_manager.run_command(
        device_name, "show chassis alarms"
    )
    results["show system alarms"] = await device_manager.run_command(
        device_name, "show system alarms"
    )
    return results


async def check_environment(device_manager: DeviceManager,
                            device_name: str) -> dict[str, CommandResult]:
    """Check chassis environment — temperatures, fans, power supplies."""
    results = {}
    results["show chassis environment"] = await device_manager.run_command(
        device_name, "show chassis environment"
    )
    results["show chassis routing-engine"] = await device_manager.run_command(
        device_name, "show chassis routing-engine"
    )
    return results


async def check_fpc(device_manager: DeviceManager,
                    device_name: str) -> dict[str, CommandResult]:
    """Check FPC (line card) status and per-FPC CPU/memory."""
    results = {}
    results["show chassis fpc"] = await device_manager.run_command(
        device_name, "show chassis fpc"
    )
    return results


async def check_pfe_exceptions(device_manager: DeviceManager,
                               device_name: str) -> dict[str, CommandResult]:
    """Check PFE statistics exceptions for sudden traffic-affecting issues."""
    results = {}
    results["show pfe statistics exceptions"] = await device_manager.run_command(
        device_name, "show pfe statistics exceptions"
    )
    return results
