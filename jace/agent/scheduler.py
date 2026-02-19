"""Background health check scheduler."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from jace.config.settings import ScheduleConfig

logger = logging.getLogger(__name__)

# Type for the callback that processes a check category
ScheduleCallback = Callable[[str, str], Awaitable[None]]  # (category, device_name)


class Scheduler:
    """Schedules periodic health checks per category per device."""

    def __init__(self, config: ScheduleConfig) -> None:
        self._intervals: dict[str, int] = {
            "chassis": config.chassis,
            "interfaces": config.interfaces,
            "routing": config.routing,
            "system": config.system,
            "config": config.config,
        }
        self._tasks: list[asyncio.Task] = []
        self._running = False

    def start(self, devices: list[str], callback: ScheduleCallback) -> None:
        """Start scheduling health checks for all devices and categories."""
        self._running = True
        for device_name in devices:
            for category, interval in self._intervals.items():
                task = asyncio.create_task(
                    self._run_loop(category, device_name, interval, callback),
                    name=f"check-{category}-{device_name}",
                )
                self._tasks.append(task)
        logger.info("Scheduler started: %d check loops", len(self._tasks))

    async def stop(self) -> None:
        """Stop all scheduled check loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Scheduler stopped")

    async def _run_loop(self, category: str, device_name: str,
                        interval: int, callback: ScheduleCallback) -> None:
        """Run a check category on a schedule."""
        # Initial delay: stagger checks to avoid thundering herd
        stagger = hash(f"{category}-{device_name}") % min(30, interval)
        await asyncio.sleep(stagger)

        while self._running:
            try:
                logger.debug("Running scheduled check: %s on %s", category, device_name)
                await callback(category, device_name)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Scheduled check %s/%s failed: %s",
                             category, device_name, exc)
            await asyncio.sleep(interval)

    def update_interval(self, category: str, interval: int) -> None:
        """Update the interval for a category (takes effect on next cycle)."""
        self._intervals[category] = interval
