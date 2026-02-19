"""Z-score anomaly detection for time-series metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from jace.agent.metrics_store import MetricPoint, MetricsStore


@dataclass
class AnomalyResult:
    metric: str
    value: float
    mean: float
    stddev: float
    z_score: float
    unit: str

    def to_context_line(self) -> str:
        return (
            f"ANOMALY: {self.metric} = {self.value}{self.unit} "
            f"(mean={self.mean:.2f}, stddev={self.stddev:.2f}, "
            f"z-score={self.z_score:.2f})"
        )


class AnomalyDetector:
    """Stateless Z-score anomaly detector backed by MetricsStore."""

    def __init__(
        self,
        store: MetricsStore,
        z_threshold: float = 3.0,
        window_hours: int = 24,
        min_samples: int = 10,
    ) -> None:
        self._store = store
        self._z_threshold = z_threshold
        self._window_hours = window_hours
        self._min_samples = min_samples

    async def check(
        self, device: str, metric: str, current_value: float, unit: str = "",
    ) -> AnomalyResult | None:
        points = await self._store.query(
            device, metric, since_hours=self._window_hours,
        )
        if len(points) < self._min_samples:
            return None

        values = [p.value for p in points]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = math.sqrt(variance)

        if stddev == 0:
            return None

        z_score = abs(current_value - mean) / stddev
        if z_score >= self._z_threshold:
            return AnomalyResult(
                metric=metric,
                value=current_value,
                mean=mean,
                stddev=stddev,
                z_score=z_score,
                unit=unit,
            )
        return None

    async def check_many(
        self, device: str, points: list[MetricPoint],
    ) -> list[AnomalyResult]:
        results: list[AnomalyResult] = []
        for point in points:
            result = await self.check(
                device, point.metric, point.value, point.unit,
            )
            if result:
                results.append(result)
        return results
