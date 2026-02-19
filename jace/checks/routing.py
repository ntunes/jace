"""Routing health checks — BGP, OSPF, route tables."""

from __future__ import annotations

from jace.device.manager import DeviceManager
from jace.device.models import CommandResult


async def check_bgp(device_manager: DeviceManager,
                    device_name: str) -> dict[str, CommandResult]:
    """Check BGP peer states and prefix counts."""
    results = {}
    results["show bgp summary"] = await device_manager.run_command(
        device_name, "show bgp summary"
    )
    return results


async def check_ospf(device_manager: DeviceManager,
                     device_name: str) -> dict[str, CommandResult]:
    """Check OSPF neighbor adjacencies."""
    results = {}
    results["show ospf neighbor"] = await device_manager.run_command(
        device_name, "show ospf neighbor"
    )
    return results


async def check_isis(device_manager: DeviceManager,
                     device_name: str) -> dict[str, CommandResult]:
    """Check IS-IS adjacencies."""
    results = {}
    results["show isis adjacency"] = await device_manager.run_command(
        device_name, "show isis adjacency"
    )
    return results


async def check_routes(device_manager: DeviceManager,
                       device_name: str) -> dict[str, CommandResult]:
    """Check route table summary for unexpected changes."""
    results = {}
    results["show route summary"] = await device_manager.run_command(
        device_name, "show route summary"
    )
    return results
