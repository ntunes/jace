"""Interface metric extractors — status counts, error counts."""

from __future__ import annotations

import re
from typing import Any

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric, xml_findall, xml_findtext, xml_float


def extract_interface_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    terse_result = results.get("show interfaces terse")
    if terse_result and terse_result.success:
        if terse_result.structured is not None:
            metrics.extend(_interfaces_terse_xml(terse_result.structured))
        elif terse_result.output:
            metrics.extend(_interfaces_terse_text(terse_result.output))

    stats_result = results.get("show interfaces statistics")
    if stats_result and stats_result.success:
        if stats_result.structured is not None:
            metrics.extend(_interfaces_stats_xml(stats_result.structured))
        elif stats_result.output:
            metrics.extend(_interfaces_stats_text(stats_result.output))

    return metrics


# ── XML parsers (PyEZ RPC output) ───────────────────────────────────

def _interfaces_terse_xml(xml: Any) -> list[ExtractedMetric]:
    up_count = 0
    down_count = 0
    for iface in xml_findall(xml, "physical-interface"):
        admin = xml_findtext(iface, "admin-status").lower()
        oper = xml_findtext(iface, "oper-status").lower()
        if admin == "up" and oper == "up":
            up_count += 1
        elif admin == "up" and oper == "down":
            down_count += 1
    metrics: list[ExtractedMetric] = []
    if up_count or down_count:
        metrics.append(ExtractedMetric(
            metric="iface_up_count", value=float(up_count), unit="interfaces",
        ))
        metrics.append(ExtractedMetric(
            metric="iface_down_count", value=float(down_count), unit="interfaces",
        ))
    return metrics


def _interfaces_stats_xml(xml: Any) -> list[ExtractedMetric]:
    total_errors = 0
    for iface in xml_findall(xml, "physical-interface"):
        total_errors += xml_float(iface, "input-errors")
        total_errors += xml_float(iface, "output-errors")
    metrics: list[ExtractedMetric] = []
    if total_errors > 0:
        metrics.append(ExtractedMetric(
            metric="iface_error_count", value=total_errors,
            unit="errors", is_counter=True,
        ))
    return metrics


# ── Text parsers (Netmiko CLI fallback) ─────────────────────────────

def _interfaces_terse_text(output: str) -> list[ExtractedMetric]:
    up_count = 0
    down_count = 0
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and not line.startswith(" "):
            iface_name = parts[0]
            if "." in iface_name:
                continue
            admin_status = parts[1].lower()
            oper_status = parts[2].lower()
            if admin_status == "up" and oper_status == "up":
                up_count += 1
            elif admin_status == "up" and oper_status == "down":
                down_count += 1
    metrics: list[ExtractedMetric] = []
    if up_count or down_count:
        metrics.append(ExtractedMetric(
            metric="iface_up_count", value=float(up_count), unit="interfaces",
        ))
        metrics.append(ExtractedMetric(
            metric="iface_down_count", value=float(down_count), unit="interfaces",
        ))
    return metrics


def _interfaces_stats_text(output: str) -> list[ExtractedMetric]:
    total_errors = 0
    for m in re.finditer(
        r"(?:Input|Output)\s+errors:\s+(\d+)", output, re.IGNORECASE,
    ):
        total_errors += int(m.group(1))
    metrics: list[ExtractedMetric] = []
    if total_errors > 0:
        metrics.append(ExtractedMetric(
            metric="iface_error_count", value=float(total_errors),
            unit="errors", is_counter=True,
        ))
    return metrics
