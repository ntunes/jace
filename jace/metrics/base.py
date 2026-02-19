"""Shared types for metric extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedMetric:
    """A single metric extracted from command output."""
    metric: str
    value: float
    unit: str = ""
    tags: dict = field(default_factory=dict)
    is_counter: bool = False
