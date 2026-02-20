"""Lightweight metric watch system — background collection without LLM cost."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass

from jace.agent.metrics_store import MetricPoint, MetricsStore
from jace.device.manager import DeviceManager

logger = logging.getLogger(__name__)

MIN_INTERVAL = 30


@dataclass
class Watch:
    id: str
    device: str
    command: str
    metric_name: str
    interval: int
    parse_pattern: str
    unit: str = ""


class WatchManager:
    """Manages lightweight metric watches — periodic command + regex collection."""

    def __init__(
        self,
        device_manager: DeviceManager,
        metrics_store: MetricsStore,
    ) -> None:
        self._device_manager = device_manager
        self._metrics_store = metrics_store
        self._watches: dict[str, Watch] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def _make_id(device: str, command: str, metric_name: str) -> str:
        key = f"{device}:{command}:{metric_name}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def add(self, watch: Watch) -> str:
        """Register a watch and start its collection loop.

        Idempotent — same (device, command, metric_name) returns the same id.
        Raises ValueError if parse_pattern is not a valid regex or lacks
        a ``value`` named group.
        """
        # Validate regex before registering
        try:
            compiled = re.compile(watch.parse_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        if "value" not in compiled.groupindex:
            raise ValueError(
                "parse_pattern must contain a named group 'value' "
                "(e.g. r'(?P<value>\\d+)')"
            )

        # Enforce minimum interval
        watch.interval = max(MIN_INTERVAL, watch.interval)

        # Deterministic id
        watch.id = self._make_id(watch.device, watch.command, watch.metric_name)

        if watch.id in self._watches:
            return watch.id

        self._watches[watch.id] = watch
        self._tasks[watch.id] = asyncio.create_task(
            self._collection_loop(watch),
            name=f"watch-{watch.id}",
        )
        logger.info("Watch added: %s (%s on %s every %ds)",
                     watch.id, watch.metric_name, watch.device, watch.interval)
        return watch.id

    def remove(self, watch_id: str) -> bool:
        """Stop and remove a watch. Returns True if found."""
        watch = self._watches.pop(watch_id, None)
        if watch is None:
            return False

        task = self._tasks.pop(watch_id, None)
        if task is not None:
            task.cancel()

        logger.info("Watch removed: %s (%s)", watch_id, watch.metric_name)
        return True

    def list_watches(self) -> list[Watch]:
        return list(self._watches.values())

    def stop_all(self) -> None:
        """Cancel all collection loops."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._watches.clear()
        logger.info("All watches stopped")

    async def _collect_once(self, watch: Watch, compiled: re.Pattern) -> None:
        """Run one collection cycle for a watch."""
        result = await self._device_manager.run_command(
            watch.device, watch.command,
        )
        if not result.success:
            logger.warning("Watch %s: command failed — %s",
                           watch.id, result.error)
            return

        match = compiled.search(result.output or "")
        if not match:
            logger.debug("Watch %s: no regex match in output", watch.id)
            return

        try:
            value = float(match.group("value"))
        except (ValueError, IndexError):
            logger.debug("Watch %s: could not parse value", watch.id)
            return

        point = MetricPoint(
            device=watch.device,
            category="watch",
            metric=watch.metric_name,
            value=value,
            unit=watch.unit,
        )
        await self._metrics_store.record(point)
        logger.debug("Watch %s: recorded %s=%s%s",
                     watch.id, watch.metric_name, value, watch.unit)

    async def _collection_loop(self, watch: Watch) -> None:
        """Run command, extract metric, record to store — repeat."""
        # Stagger initial start to avoid thundering herd
        stagger = hash(watch.id) % min(10, watch.interval)
        await asyncio.sleep(stagger)

        compiled = re.compile(watch.parse_pattern)

        while True:
            try:
                await self._collect_once(watch, compiled)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Watch %s: unexpected error — %s", watch.id, exc)

            await asyncio.sleep(watch.interval)
