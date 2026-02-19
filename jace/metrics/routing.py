"""Routing metric extractors — routes, BGP, OSPF."""

from __future__ import annotations

import re
from typing import Any

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric, xml_findall, xml_findtext, xml_float


def extract_routing_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    route_result = results.get("show route summary")
    if route_result and route_result.success:
        if route_result.structured is not None:
            metrics.extend(_route_summary_xml(route_result.structured))
        elif route_result.output:
            metrics.extend(_route_summary_text(route_result.output))

    bgp_result = results.get("show bgp summary")
    if bgp_result and bgp_result.success:
        if bgp_result.structured is not None:
            metrics.extend(_bgp_summary_xml(bgp_result.structured))
        elif bgp_result.output:
            metrics.extend(_bgp_summary_text(bgp_result.output))

    ospf_result = results.get("show ospf neighbor")
    if ospf_result and ospf_result.success:
        if ospf_result.structured is not None:
            metrics.extend(_ospf_neighbor_xml(ospf_result.structured))
        elif ospf_result.output:
            metrics.extend(_ospf_neighbor_text(ospf_result.output))

    return metrics


# ── XML parsers (PyEZ RPC output) ───────────────────────────────────

def _route_summary_xml(xml: Any) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    total_routes = 0
    total_active = 0
    for table in xml_findall(xml, "route-table"):
        total_routes += xml_float(table, "total-route-count")
        total_active += xml_float(table, "active-route-count")
    if total_routes:
        metrics.append(ExtractedMetric(
            metric="route_total", value=total_routes, unit="routes",
        ))
    if total_active:
        metrics.append(ExtractedMetric(
            metric="route_active", value=total_active, unit="routes",
        ))
    return metrics


def _bgp_summary_xml(xml: Any) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    peer_count = xml_float(xml, "peer-count")
    if peer_count:
        metrics.append(ExtractedMetric(
            metric="bgp_peer_count", value=peer_count, unit="peers",
        ))
    established = sum(
        1 for peer in xml_findall(xml, "bgp-peer")
        if xml_findtext(peer, "peer-state") == "Established"
    )
    metrics.append(ExtractedMetric(
        metric="bgp_established_count", value=float(established), unit="peers",
    ))
    return metrics


def _ospf_neighbor_xml(xml: Any) -> list[ExtractedMetric]:
    neighbors = xml_findall(xml, "ospf-neighbor")
    return [ExtractedMetric(
        metric="ospf_neighbor_count", value=float(len(neighbors)),
        unit="neighbors",
    )]


# ── Text parsers (Netmiko CLI fallback) ─────────────────────────────

def _route_summary_text(output: str) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    m = re.search(r"(\d+)\s+destinations,\s+(\d+)\s+routes", output)
    if m:
        metrics.append(ExtractedMetric(
            metric="route_total", value=float(m.group(2)), unit="routes",
        ))
    m = re.search(r"\d+\s+routes\s+\((\d+)\s+active", output)
    if m:
        metrics.append(ExtractedMetric(
            metric="route_active", value=float(m.group(1)), unit="routes",
        ))
    return metrics


def _bgp_summary_text(output: str) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []
    peer_lines = re.findall(
        r"^\s*\d+\.\d+\.\d+\.\d+\s+", output, re.MULTILINE,
    )
    if peer_lines:
        metrics.append(ExtractedMetric(
            metric="bgp_peer_count", value=float(len(peer_lines)), unit="peers",
        ))
    established = 0
    for line in output.splitlines():
        parts = line.split()
        if parts and re.match(r"\d+\.\d+\.\d+\.\d+", parts[0]):
            state = parts[-1] if parts else ""
            if "/" in state or state.isdigit():
                established += 1
    if peer_lines:
        metrics.append(ExtractedMetric(
            metric="bgp_established_count", value=float(established), unit="peers",
        ))
    return metrics


def _ospf_neighbor_text(output: str) -> list[ExtractedMetric]:
    neighbor_lines = re.findall(
        r"^\s*\d+\.\d+\.\d+\.\d+\s+", output, re.MULTILINE,
    )
    return [ExtractedMetric(
        metric="ospf_neighbor_count", value=float(len(neighbor_lines)),
        unit="neighbors",
    )]
