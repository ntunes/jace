"""Metric extraction — maps check categories to extractor functions."""

from __future__ import annotations

from typing import Callable

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric
from jace.metrics.chassis import extract_chassis_metrics
from jace.metrics.interfaces import extract_interface_metrics
from jace.metrics.routing import extract_routing_metrics
from jace.metrics.system import extract_system_metrics

ExtractorFunc = Callable[[dict[str, CommandResult]], list[ExtractedMetric]]

EXTRACTORS: dict[str, ExtractorFunc] = {
    "routing": extract_routing_metrics,
    "chassis": extract_chassis_metrics,
    "interfaces": extract_interface_metrics,
    "system": extract_system_metrics,
}
