"""SQLite time-series metrics store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    device: str
    category: str
    metric: str
    value: float
    unit: str = ""
    ts: str = ""
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "category": self.category,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "ts": self.ts,
            "tags": self.tags,
        }


class MetricsStore:
    """Append-only SQLite time-series store for device metrics."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._db_path = storage_path / "metrics.db"
        self._db: aiosqlite.Connection | None = None

    async def initialize(self, retention_days: int = 30) -> None:
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                device   TEXT NOT NULL,
                category TEXT NOT NULL,
                metric   TEXT NOT NULL,
                value    REAL NOT NULL,
                unit     TEXT NOT NULL DEFAULT '',
                ts       TEXT NOT NULL,
                tags     TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_device_metric_ts
                ON metrics (device, metric, ts)
        """)
        await self._db.commit()
        deleted = await self.cleanup_old(retention_days)
        if deleted:
            logger.info("Cleaned up %d old metric rows", deleted)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def record(self, point: MetricPoint) -> None:
        if not self._db:
            return
        if not point.ts:
            point.ts = datetime.now().isoformat()
        await self._db.execute(
            "INSERT INTO metrics (device, category, metric, value, unit, ts, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (point.device, point.category, point.metric, point.value,
             point.unit, point.ts, json.dumps(point.tags)),
        )
        await self._db.commit()

    async def record_many(self, points: list[MetricPoint]) -> None:
        if not self._db or not points:
            return
        now = datetime.now().isoformat()
        rows = []
        for p in points:
            if not p.ts:
                p.ts = now
            rows.append((p.device, p.category, p.metric, p.value,
                          p.unit, p.ts, json.dumps(p.tags)))
        await self._db.executemany(
            "INSERT INTO metrics (device, category, metric, value, unit, ts, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await self._db.commit()

    async def query(self, device: str, metric: str,
                    since_hours: int = 24,
                    limit: int = 1000) -> list[MetricPoint]:
        if not self._db:
            return []
        since = (datetime.now() - timedelta(hours=since_hours)).isoformat()
        async with self._db.execute(
            "SELECT device, category, metric, value, unit, ts, tags "
            "FROM metrics WHERE device = ? AND metric = ? AND ts >= ? "
            "ORDER BY ts ASC LIMIT ?",
            (device, metric, since, limit),
        ) as cursor:
            return [self._row_to_point(row) async for row in cursor]

    async def latest(self, device: str, metric: str) -> MetricPoint | None:
        if not self._db:
            return None
        async with self._db.execute(
            "SELECT device, category, metric, value, unit, ts, tags "
            "FROM metrics WHERE device = ? AND metric = ? "
            "ORDER BY ts DESC LIMIT 1",
            (device, metric),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_point(row) if row else None

    async def list_metrics(self, device: str) -> list[str]:
        if not self._db:
            return []
        async with self._db.execute(
            "SELECT DISTINCT metric FROM metrics WHERE device = ? ORDER BY metric",
            (device,),
        ) as cursor:
            return [row[0] async for row in cursor]

    async def cleanup_old(self, retention_days: int = 30) -> int:
        if not self._db:
            return 0
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = await self._db.execute(
            "DELETE FROM metrics WHERE ts < ?", (cutoff,),
        )
        await self._db.commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_point(row: tuple) -> MetricPoint:
        return MetricPoint(
            device=row[0],
            category=row[1],
            metric=row[2],
            value=row[3],
            unit=row[4],
            ts=row[5],
            tags=json.loads(row[6]) if row[6] else {},
        )
