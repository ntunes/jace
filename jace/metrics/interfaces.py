"""Interface metric extractors — status counts, error counts."""

from __future__ import annotations

import re

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric


def extract_interface_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    # --- show interfaces terse ---
    terse_result = results.get("show interfaces terse")
    if terse_result and terse_result.success and terse_result.output:
        output = terse_result.output
        up_count = 0
        down_count = 0
        for line in output.splitlines():
            parts = line.split()
            # Typical line: "ge-0/0/0  up  up"
            # Skip header, sub-interfaces (.0, .100 etc), and indented lines
            if len(parts) >= 3 and not line.startswith(" "):
                iface_name = parts[0]
                # Skip sub-interfaces (contain a dot unit number)
                if "." in iface_name:
                    continue
                admin_status = parts[1].lower()
                oper_status = parts[2].lower()
                if admin_status == "up" and oper_status == "up":
                    up_count += 1
                elif admin_status == "up" and oper_status == "down":
                    down_count += 1
        if up_count or down_count:
            metrics.append(ExtractedMetric(
                metric="iface_up_count", value=float(up_count),
                unit="interfaces",
            ))
            metrics.append(ExtractedMetric(
                metric="iface_down_count", value=float(down_count),
                unit="interfaces",
            ))

    # --- show interfaces statistics ---
    stats_result = results.get("show interfaces statistics")
    if stats_result and stats_result.success and stats_result.output:
        output = stats_result.output
        total_errors = 0
        for m in re.finditer(
            r"(?:Input|Output)\s+errors:\s+(\d+)", output, re.IGNORECASE,
        ):
            total_errors += int(m.group(1))
        if total_errors > 0:
            metrics.append(ExtractedMetric(
                metric="iface_error_count", value=float(total_errors),
                unit="errors", is_counter=True,
            ))

    return metrics
