"""Findings tracker — deduplication, severity, history, SQLite persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    id: str
    device: str
    severity: Severity
    category: str
    title: str
    detail: str
    recommendation: str
    first_seen: str
    last_seen: str
    resolved: bool = False
    raw_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


def _generate_finding_id(device: str, category: str, title: str) -> str:
    """Generate a deterministic ID for deduplication."""
    key = f"{device}:{category}:{title}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class FindingsTracker:
    """Manages findings with deduplication and SQLite persistence."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._db_path = storage_path / "findings.db"
        self._active: dict[str, Finding] = {}
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                device TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                detail TEXT,
                recommendation TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                raw_data TEXT
            )
        """)
        await self._db.commit()

        # Load active (unresolved) findings into memory
        async with self._db.execute(
            "SELECT * FROM findings WHERE resolved = 0"
        ) as cursor:
            async for row in cursor:
                finding = self._row_to_finding(row)
                self._active[finding.id] = finding

        logger.info("Loaded %d active findings", len(self._active))

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def add_or_update(self, device: str, severity: Severity,
                            category: str, title: str, detail: str,
                            recommendation: str,
                            raw_data: dict | None = None) -> tuple[Finding, bool]:
        """Add a new finding or update an existing one.

        Returns (finding, is_new) tuple.
        """
        finding_id = _generate_finding_id(device, category, title)
        now = datetime.now().isoformat()

        if finding_id in self._active:
            # Update existing finding
            existing = self._active[finding_id]
            existing.last_seen = now
            existing.detail = detail
            existing.severity = severity
            existing.recommendation = recommendation
            if raw_data:
                existing.raw_data = raw_data
            await self._persist(existing)
            return existing, False

        # New finding
        finding = Finding(
            id=finding_id,
            device=device,
            severity=severity,
            category=category,
            title=title,
            detail=detail,
            recommendation=recommendation,
            first_seen=now,
            last_seen=now,
            raw_data=raw_data or {},
        )
        self._active[finding_id] = finding
        await self._persist(finding)
        return finding, True

    async def resolve_missing(self, device: str, category: str,
                              current_titles: set[str]) -> list[Finding]:
        """Mark findings as resolved if they weren't reported this cycle."""
        resolved = []
        for finding in list(self._active.values()):
            if (finding.device == device and finding.category == category
                    and finding.title not in current_titles):
                finding.resolved = True
                finding.last_seen = datetime.now().isoformat()
                await self._persist(finding)
                del self._active[finding.id]
                resolved.append(finding)
        return resolved

    def get_active(self, device: str | None = None,
                   severity: Severity | None = None,
                   category: str | None = None) -> list[Finding]:
        findings = list(self._active.values())
        if device:
            findings = [f for f in findings if f.device == device]
        if severity:
            findings = [f for f in findings if f.severity == severity]
        if category:
            findings = [f for f in findings if f.category == category]
        return sorted(findings, key=lambda f: (
            {"critical": 0, "warning": 1, "info": 2}[f.severity.value],
            f.last_seen,
        ))

    async def get_history(self, device: str | None = None,
                          include_resolved: bool = True,
                          limit: int = 100) -> list[Finding]:
        if not self._db:
            return []
        query = "SELECT * FROM findings"
        params: list = []
        conditions = []
        if device:
            conditions.append("device = ?")
            params.append(device)
        if not include_resolved:
            conditions.append("resolved = 0")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            return [self._row_to_finding(row) async for row in cursor]

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self._active.values()
                   if f.severity == Severity.CRITICAL)

    async def _persist(self, finding: Finding) -> None:
        if not self._db:
            return
        await self._db.execute("""
            INSERT OR REPLACE INTO findings
            (id, device, severity, category, title, detail, recommendation,
             first_seen, last_seen, resolved, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            finding.id, finding.device, finding.severity.value,
            finding.category, finding.title, finding.detail,
            finding.recommendation, finding.first_seen, finding.last_seen,
            int(finding.resolved), json.dumps(finding.raw_data),
        ))
        await self._db.commit()

    @staticmethod
    def _row_to_finding(row: tuple) -> Finding:
        return Finding(
            id=row[0], device=row[1], severity=Severity(row[2]),
            category=row[3], title=row[4], detail=row[5] or "",
            recommendation=row[6] or "", first_seen=row[7],
            last_seen=row[8], resolved=bool(row[9]),
            raw_data=json.loads(row[10]) if row[10] else {},
        )
