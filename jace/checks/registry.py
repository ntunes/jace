"""Check registry — discovery and dispatch for health checks."""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

from jace.device.manager import DeviceManager
from jace.device.models import CommandResult

logger = logging.getLogger(__name__)

# Type alias for check functions
CheckFunc = Callable[[DeviceManager, str], Awaitable[dict[str, CommandResult]]]


class CheckRegistry:
    """Registry for health check functions organized by category."""

    def __init__(self) -> None:
        self._checks: dict[str, list[CheckFunc]] = {}

    def register(self, category: str, func: CheckFunc) -> None:
        self._checks.setdefault(category, []).append(func)

    def get_checks(self, category: str) -> list[CheckFunc]:
        return self._checks.get(category, [])

    def categories(self) -> list[str]:
        return list(self._checks.keys())

    async def run_category(self, category: str, device_manager: DeviceManager,
                           device_name: str) -> dict[str, CommandResult]:
        """Run all checks in a category and merge results."""
        all_results: dict[str, CommandResult] = {}
        for check_func in self.get_checks(category):
            try:
                results = await check_func(device_manager, device_name)
                all_results.update(results)
            except Exception as exc:
                logger.error("Check %s failed for %s: %s",
                             check_func.__name__, device_name, exc)
        return all_results


def build_default_registry() -> CheckRegistry:
    """Build a registry with all default health checks."""
    from jace.checks.chassis import check_alarms, check_environment, check_fpc, check_pfe_exceptions
    from jace.checks.interfaces import check_interface_status, check_interface_errors
    from jace.checks.routing import check_bgp, check_ospf, check_routes
    from jace.checks.system import check_resource_usage, check_storage, check_processes
    from jace.checks.config_audit import audit_security, audit_best_practices

    registry = CheckRegistry()

    registry.register("chassis", check_alarms)
    registry.register("chassis", check_environment)
    registry.register("chassis", check_fpc)
    registry.register("chassis", check_pfe_exceptions)

    registry.register("interfaces", check_interface_status)
    registry.register("interfaces", check_interface_errors)

    registry.register("routing", check_bgp)
    registry.register("routing", check_ospf)
    registry.register("routing", check_routes)

    registry.register("system", check_resource_usage)
    registry.register("system", check_storage)
    registry.register("system", check_processes)

    registry.register("config", audit_security)
    registry.register("config", audit_best_practices)

    return registry
