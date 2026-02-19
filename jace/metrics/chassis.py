"""Chassis metric extractors — RE CPU/memory, PFE exceptions."""

from __future__ import annotations

import re

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric


def extract_chassis_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    # --- show chassis routing-engine ---
    re_result = results.get("show chassis routing-engine")
    if re_result and re_result.success and re_result.output:
        output = re_result.output

        # CPU utilization: "CPU utilization: X percent"
        m = re.search(r"CPU\s+utilization[:\s]+(\d+)\s*percent", output, re.IGNORECASE)
        if m:
            metrics.append(ExtractedMetric(
                metric="re_cpu_pct", value=float(m.group(1)), unit="%",
            ))

        # Memory utilization: "Memory utilization X percent"
        m = re.search(r"Memory\s+utilization[:\s]+(\d+)\s*percent", output, re.IGNORECASE)
        if m:
            metrics.append(ExtractedMetric(
                metric="re_memory_pct", value=float(m.group(1)), unit="%",
            ))

    # --- show pfe statistics exceptions ---
    pfe_result = results.get("show pfe statistics exceptions")
    if pfe_result and pfe_result.success and pfe_result.output:
        output = pfe_result.output
        # Parse "ExceptionName: N" lines
        for m in re.finditer(r"^\s*(\S[\w\s-]+?):\s+(\d+)", output, re.MULTILINE):
            name = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            value = float(m.group(2))
            if value > 0:
                metrics.append(ExtractedMetric(
                    metric=f"pfe_exception_{name}",
                    value=value,
                    unit="exceptions",
                    is_counter=True,
                ))

    return metrics
