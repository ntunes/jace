"""System metric extractors — disk, load average."""

from __future__ import annotations

import re

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric


def extract_system_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    # --- show system storage ---
    storage_result = results.get("show system storage")
    if storage_result and storage_result.success and storage_result.output:
        output = storage_result.output
        # Junos storage output lines: "filesystem  size  used  avail  capacity  mounted"
        # Look for the root filesystem or highest usage
        max_pct = 0.0
        for m in re.finditer(r"(\d+)%\s+(/\S*)", output):
            pct = float(m.group(1))
            if pct > max_pct:
                max_pct = pct
        if max_pct > 0:
            metrics.append(ExtractedMetric(
                metric="disk_used_pct", value=max_pct, unit="%",
            ))

    # --- show chassis routing-engine (for load average) ---
    re_result = results.get("show chassis routing-engine")
    if re_result and re_result.success and re_result.output:
        output = re_result.output
        # "Load averages: 1 minute   5 minute  15 minute
        #                    0.25      0.18      0.15"
        # Or single-line: "Load average: 0.25"
        m = re.search(
            r"Load\s+averages?.*?(\d+\.\d+)", output, re.IGNORECASE | re.DOTALL,
        )
        if m:
            metrics.append(ExtractedMetric(
                metric="re_load_avg", value=float(m.group(1)), unit="load",
            ))

    return metrics
