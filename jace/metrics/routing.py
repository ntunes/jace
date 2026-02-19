"""Routing metric extractors — routes, BGP, OSPF."""

from __future__ import annotations

import re

from jace.device.models import CommandResult
from jace.metrics.base import ExtractedMetric


def extract_routing_metrics(
    results: dict[str, CommandResult],
) -> list[ExtractedMetric]:
    metrics: list[ExtractedMetric] = []

    # --- show route summary ---
    route_result = results.get("show route summary")
    if route_result and route_result.success and route_result.output:
        output = route_result.output

        # Total routes: "Router: N destinations, M routes (..."
        m = re.search(r"(\d+)\s+destinations,\s+(\d+)\s+routes", output)
        if m:
            metrics.append(ExtractedMetric(
                metric="route_total", value=float(m.group(2)), unit="routes",
            ))

        # Active routes: "N routes (N active, ..."
        m = re.search(r"\d+\s+routes\s+\((\d+)\s+active", output)
        if m:
            metrics.append(ExtractedMetric(
                metric="route_active", value=float(m.group(1)), unit="routes",
            ))

    # --- show bgp summary ---
    bgp_result = results.get("show bgp summary")
    if bgp_result and bgp_result.success and bgp_result.output:
        output = bgp_result.output
        # Count peer lines: lines starting with an IP address
        peer_lines = re.findall(
            r"^\s*\d+\.\d+\.\d+\.\d+\s+", output, re.MULTILINE,
        )
        if peer_lines:
            metrics.append(ExtractedMetric(
                metric="bgp_peer_count", value=float(len(peer_lines)),
                unit="peers",
            ))

        # Established peers show prefix counts like "4/8/8/0" in state column;
        # non-established show "Active", "Idle", "Connect", etc.
        established = 0
        for line in output.splitlines():
            parts = line.split()
            if parts and re.match(r"\d+\.\d+\.\d+\.\d+", parts[0]):
                # Last column is state or prefix-count (e.g. "4/8/8/0")
                state = parts[-1] if parts else ""
                if "/" in state or state.isdigit():
                    established += 1
        if peer_lines:
            metrics.append(ExtractedMetric(
                metric="bgp_established_count", value=float(established),
                unit="peers",
            ))

    # --- show ospf neighbor ---
    ospf_result = results.get("show ospf neighbor")
    if ospf_result and ospf_result.success and ospf_result.output:
        output = ospf_result.output
        neighbor_lines = re.findall(
            r"^\s*\d+\.\d+\.\d+\.\d+\s+", output, re.MULTILINE,
        )
        metrics.append(ExtractedMetric(
            metric="ospf_neighbor_count", value=float(len(neighbor_lines)),
            unit="neighbors",
        ))

    return metrics
