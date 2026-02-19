"""System health checks — CPU, memory, storage, processes."""

from __future__ import annotations

from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


async def check_resource_usage(device_manager: DeviceManager,
                               device_name: str) -> dict[str, CommandResult]:
    """Check RE CPU, memory, and load average."""
    results = {}
    results["show chassis routing-engine"] = await device_manager.run_command(
        device_name, "show chassis routing-engine"
    )
    return results


async def check_storage(device_manager: DeviceManager,
                        device_name: str) -> dict[str, CommandResult]:
    """Check disk utilization."""
    results = {}
    results["show system storage"] = await device_manager.run_command(
        device_name, "show system storage"
    )
    return results


async def check_processes(device_manager: DeviceManager,
                          device_name: str) -> dict[str, CommandResult]:
    """Check top CPU-consuming processes."""
    results = {}
    results["show system processes extensive"] = await device_manager.run_command(
        device_name, "show system processes extensive"
    )
    return results
