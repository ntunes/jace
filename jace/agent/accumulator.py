"""Temporal anomaly batching — groups anomalies per device over a sliding window."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from jace.agent.anomaly import AnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEntry:
    """Single category's anomaly data within a batch."""
    category: str
    anomalies: list[AnomalyResult]
    raw_data: str


@dataclass
class AnomalyBatch:
    """All anomaly entries for a single device collected over a window."""
    device: str
    entries: list[AnomalyEntry] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        return [e.category for e in self.entries]


# Callback type: receives a batch and investigates it
BatchCallback = Callable[[AnomalyBatch], Awaitable[None]]


class AnomalyAccumulator:
    """Batches anomalies per device over a sliding time window.

    Each ``submit()`` for a device restarts the window timer, so a burst
    of categories firing within ``window_seconds`` of each other all land
    in the same batch.
    """

    def __init__(self, window_seconds: float = 30.0) -> None:
        self._window = window_seconds
        self._batches: dict[str, AnomalyBatch] = {}
        self._timers: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._callback: BatchCallback | None = None

    def set_callback(self, callback: BatchCallback) -> None:
        self._callback = callback

    @property
    def pending_count(self) -> int:
        return len(self._batches)

    async def submit(
        self, device: str, category: str,
        anomalies: list[AnomalyResult], raw_data: str,
    ) -> None:
        """Add an anomaly entry and (re)start the flush timer for *device*."""
        entry = AnomalyEntry(
            category=category, anomalies=anomalies, raw_data=raw_data,
        )
        async with self._lock:
            if device not in self._batches:
                self._batches[device] = AnomalyBatch(device=device)
            self._batches[device].entries.append(entry)

            # Cancel existing timer (resets the window)
            existing = self._timers.pop(device, None)
            if existing is not None:
                existing.cancel()

            # Start new timer
            self._timers[device] = asyncio.create_task(
                self._flush_after(device, self._window),
            )

    async def _flush_after(self, device: str, delay: float) -> None:
        """Wait *delay* seconds then flush *device*'s batch."""
        await asyncio.sleep(delay)
        await self._flush(device)

    async def _flush(self, device: str) -> None:
        """Atomically pop the batch for *device* and dispatch to callback."""
        async with self._lock:
            batch = self._batches.pop(device, None)
            self._timers.pop(device, None)

        if batch is None or not batch.entries:
            return

        if self._callback is None:
            logger.warning("Accumulator flushed but no callback set")
            return

        try:
            await self._callback(batch)
        except Exception as exc:
            logger.error("Accumulator callback failed for %s: %s", device, exc)

    async def flush_all(self) -> None:
        """Flush all pending batches immediately."""
        async with self._lock:
            devices = list(self._batches.keys())

        for device in devices:
            await self._flush(device)

    async def stop(self) -> None:
        """Cancel all pending timers and flush remaining batches."""
        async with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()

        for timer in timers:
            timer.cancel()
            try:
                await timer
            except asyncio.CancelledError:
                pass

        await self.flush_all()
