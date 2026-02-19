"""System metric extractors — disk, load average."""

from __future__ import annotations

import re
from typing import Any

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric, xml_findall, xml_float


def extract_system_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    storage_result = results.get("show system storage")
    if storage_result and storage_result.success:
        if storage_result.structured is not None:
            metrics.extend(_storage_xml(storage_result.structured))
        elif storage_result.output:
            metrics.extend(_storage_text(storage_result.output))

    re_result = results.get("show chassis routing-engine")
    if re_result and re_result.success:
        if re_result.structured is not None:
            metrics.extend(_load_avg_xml(re_result.structured))
        elif re_result.output:
            metrics.extend(_load_avg_text(re_result.output))

    return metrics


# ── XML parsers (PyEZ RPC output) ───────────────────────────────────

def _storage_xml(xml: Any) -> list[ExtractedMetric]:
    max_pct = 0.0
    for fs in xml_findall(xml, "filesystem"):
        pct = xml_float(fs, "used-percent")
        if pct > max_pct:
            max_pct = pct
    metrics: list[ExtractedMetric] = []
    if max_pct > 0:
        metrics.append(ExtractedMetric(
            metric="disk_used_pct", value=max_pct, unit="%",
        ))
    return metrics


def _load_avg_xml(xml: Any) -> list[ExtractedMetric]:
    re_elements = xml_findall(xml, "route-engine")
    if not re_elements:
        return []
    load = xml_float(re_elements[0], "load-average-one", default=-1.0)
    if load >= 0:
        return [ExtractedMetric(metric="re_load_avg", value=load, unit="load")]
    return []


# ── Text parsers (Netmiko CLI fallback) ─────────────────────────────

def _storage_text(output: str) -> list[ExtractedMetric]:
    max_pct = 0.0
    for m in re.finditer(r"(\d+)%\s+(/\S*)", output):
        pct = float(m.group(1))
        if pct > max_pct:
            max_pct = pct
    metrics: list[ExtractedMetric] = []
    if max_pct > 0:
        metrics.append(ExtractedMetric(
            metric="disk_used_pct", value=max_pct, unit="%",
        ))
    return metrics


def _load_avg_text(output: str) -> list[ExtractedMetric]:
    m = re.search(
        r"Load\s+averages?.*?(\d+\.\d+)", output, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return [ExtractedMetric(
            metric="re_load_avg", value=float(m.group(1)), unit="load",
        )]
    return []
