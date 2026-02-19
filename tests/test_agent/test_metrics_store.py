"""Tests for MetricsStore."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from jace.agent.metrics_store import MetricPoint, MetricsStore


@pytest.fixture
async def store(tmp_path: Path):
    s = MetricsStore(tmp_path)
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_record_and_query(store: MetricsStore):
    point = MetricPoint(
        device="mx-01", category="routing", metric="route_total",
        value=1500.0, unit="routes",
    )
    await store.record(point)

    results = await store.query("mx-01", "route_total", since_hours=1)
    assert len(results) == 1
    assert results[0].value == 1500.0
    assert results[0].device == "mx-01"
    assert results[0].metric == "route_total"
    assert results[0].unit == "routes"


@pytest.mark.asyncio
async def test_latest(store: MetricsStore):
    await store.record(MetricPoint(
        device="mx-01", category="chassis", metric="re_cpu_pct",
        value=30.0, unit="%",
    ))
    await store.record(MetricPoint(
        device="mx-01", category="chassis", metric="re_cpu_pct",
        value=45.0, unit="%",
    ))

    latest = await store.latest("mx-01", "re_cpu_pct")
    assert latest is not None
    assert latest.value == 45.0


@pytest.mark.asyncio
async def test_latest_nonexistent(store: MetricsStore):
    latest = await store.latest("mx-01", "nonexistent")
    assert latest is None


@pytest.mark.asyncio
async def test_record_many(store: MetricsStore):
    points = [
        MetricPoint(device="mx-01", category="routing", metric="bgp_peer_count",
                     value=5.0, unit="peers"),
        MetricPoint(device="mx-01", category="routing", metric="bgp_established_count",
                     value=4.0, unit="peers"),
        MetricPoint(device="mx-01", category="routing", metric="ospf_neighbor_count",
                     value=3.0, unit="neighbors"),
    ]
    await store.record_many(points)

    results = await store.query("mx-01", "bgp_peer_count", since_hours=1)
    assert len(results) == 1
    assert results[0].value == 5.0

    results = await store.query("mx-01", "ospf_neighbor_count", since_hours=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_list_metrics(store: MetricsStore):
    await store.record_many([
        MetricPoint(device="mx-01", category="routing", metric="route_total",
                     value=100.0),
        MetricPoint(device="mx-01", category="chassis", metric="re_cpu_pct",
                     value=30.0),
        MetricPoint(device="mx-01", category="routing", metric="bgp_peer_count",
                     value=5.0),
    ])

    names = await store.list_metrics("mx-01")
    assert sorted(names) == ["bgp_peer_count", "re_cpu_pct", "route_total"]


@pytest.mark.asyncio
async def test_list_metrics_empty(store: MetricsStore):
    names = await store.list_metrics("mx-01")
    assert names == []


@pytest.mark.asyncio
async def test_cleanup_old(store: MetricsStore):
    old_ts = (datetime.now() - timedelta(days=60)).isoformat()
    recent_ts = datetime.now().isoformat()

    await store.record(MetricPoint(
        device="mx-01", category="routing", metric="route_total",
        value=100.0, ts=old_ts,
    ))
    await store.record(MetricPoint(
        device="mx-01", category="routing", metric="route_total",
        value=200.0, ts=recent_ts,
    ))

    deleted = await store.cleanup_old(retention_days=30)
    assert deleted == 1

    results = await store.query("mx-01", "route_total", since_hours=24 * 365)
    assert len(results) == 1
    assert results[0].value == 200.0


@pytest.mark.asyncio
async def test_query_respects_since_hours(store: MetricsStore):
    old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
    recent_ts = datetime.now().isoformat()

    await store.record(MetricPoint(
        device="mx-01", category="routing", metric="route_total",
        value=100.0, ts=old_ts,
    ))
    await store.record(MetricPoint(
        device="mx-01", category="routing", metric="route_total",
        value=200.0, ts=recent_ts,
    ))

    results = await store.query("mx-01", "route_total", since_hours=24)
    assert len(results) == 1
    assert results[0].value == 200.0


@pytest.mark.asyncio
async def test_to_dict(store: MetricsStore):
    point = MetricPoint(
        device="mx-01", category="routing", metric="route_total",
        value=1500.0, unit="routes", tags={"table": "inet.0"},
    )
    d = point.to_dict()
    assert d["device"] == "mx-01"
    assert d["metric"] == "route_total"
    assert d["value"] == 1500.0
    assert d["tags"] == {"table": "inet.0"}
