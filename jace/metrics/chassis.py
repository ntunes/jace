"""Chassis metric extractors — RE CPU/memory, PFE exceptions."""

from __future__ import annotations

import re
from typing import Any

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric, xml_findall, xml_findtext, xml_float


def extract_chassis_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    re_result = results.get("show chassis routing-engine")
    if re_result and re_result.success:
        if re_result.structured is not None:
            metrics.extend(_routing_engine_xml(re_result.structured))
        elif re_result.output:
            metrics.extend(_routing_engine_text(re_result.output))

    pfe_result = results.get("show pfe statistics exceptions")
    if pfe_result and pfe_result.success:
        if pfe_result.structured is not None:
            metrics.extend(_pfe_exceptions_xml(pfe_result.structured))
        elif pfe_result.output:
            metrics.extend(_pfe_exceptions_text(pfe_result.output))

    return metrics


# ── XML parsers (PyEZ RPC output) ───────────────────────────────────

def _routing_engine_xml(xml: Any) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    # Use first (master) RE slot
    re_elements = xml_findall(xml, "route-engine")
    if not re_elements:
        return metrics
    re_el = re_elements[0]

    # CPU: 100 - idle
    cpu_idle = xml_float(re_el, "cpu-idle", default=-1.0)
    if cpu_idle >= 0:
        metrics.append(ExtractedMetric(
            metric="re_cpu_pct", value=100.0 - cpu_idle, unit="%",
        ))

    # Memory utilization (MX series: memory-buffer-utilization)
    mem_pct = xml_float(re_el, "memory-buffer-utilization", default=-1.0)
    if mem_pct >= 0:
        metrics.append(ExtractedMetric(
            metric="re_memory_pct", value=mem_pct, unit="%",
        ))

    return metrics


_PFE_DISCARD_TAGS = [
    "bad-route-discard",
    "nexthop-discard",
    "invalid-iif-discard",
    "timeout-discard",
    "data-error-discard",
    "stack-underflow-discard",
    "stack-overflow-discard",
    "truncated-key-discard",
    "bits-to-test-discard",
    "info-cell-discard",
    "fabric-discard",
]


def _pfe_exceptions_xml(xml: Any) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    discard_sections = xml_findall(xml, "pfe-hardware-discard-statistics")
    if not discard_sections:
        # Also check local traffic stats for software drops
        for section in xml_findall(xml, "pfe-local-traffic-statistics"):
            hw_drops = xml_float(section, "hardware-input-drops")
            if hw_drops > 0:
                metrics.append(ExtractedMetric(
                    metric="pfe_exception_hardware_input_drops",
                    value=hw_drops, unit="exceptions", is_counter=True,
                ))
        return metrics

    for section in discard_sections:
        for tag in _PFE_DISCARD_TAGS:
            value = xml_float(section, tag)
            if value > 0:
                name = tag.replace("-", "_")
                metrics.append(ExtractedMetric(
                    metric=f"pfe_exception_{name}",
                    value=value, unit="exceptions", is_counter=True,
                ))
    return metrics


# ── Text parsers (Netmiko CLI fallback) ─────────────────────────────

def _routing_engine_text(output: str) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    m = re.search(r"CPU\s+utilization[:\s]+(\d+)\s*percent", output, re.IGNORECASE)
    if m:
        metrics.append(ExtractedMetric(
            metric="re_cpu_pct", value=float(m.group(1)), unit="%",
        ))
    m = re.search(r"Memory\s+utilization[:\s]+(\d+)\s*percent", output, re.IGNORECASE)
    if m:
        metrics.append(ExtractedMetric(
            metric="re_memory_pct", value=float(m.group(1)), unit="%",
        ))
    return metrics


def _pfe_exceptions_text(output: str) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    for m in re.finditer(r"^\s*(\S[\w\s-]+?):\s+(\d+)", output, re.MULTILINE):
        name = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        value = float(m.group(2))
        if value > 0:
            metrics.append(ExtractedMetric(
                metric=f"pfe_exception_{name}",
                value=value, unit="exceptions", is_counter=True,
            ))
    return metrics
