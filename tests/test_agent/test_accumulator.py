"""Tests for AnomalyAccumulator — temporal batching of anomalies."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from jace.agent.accumulator import (
    AnomalyAccumulator,
    AnomalyBatch,
    AnomalyEntry,
)
from jace.agent.anomaly import AnomalyResult


def _anomaly(metric: str = "cpu_temp") -> AnomalyResult:
    return AnomalyResult(
        metric=metric, value=95.0, mean=60.0,
        stddev=5.0, z_score=7.0, unit="C",
    )


@pytest.mark.asyncio
async def test_submit_creates_batch():
    acc = AnomalyAccumulator(window_seconds=10.0)
    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    assert acc.pending_count == 1


@pytest.mark.asyncio
async def test_flush_fires_after_window():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=0.05)
    acc.set_callback(callback)
    await acc.submit("r1", "chassis", [_anomaly()], "raw1")

    await asyncio.sleep(0.15)

    callback.assert_called_once()
    batch = callback.call_args[0][0]
    assert isinstance(batch, AnomalyBatch)
    assert batch.device == "r1"
    assert len(batch.entries) == 1
    assert batch.entries[0].category == "chassis"
    assert acc.pending_count == 0


@pytest.mark.asyncio
async def test_multiple_categories_same_device_single_batch():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=0.1)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly("cpu_temp")], "raw1")
    await acc.submit("r1", "interfaces", [_anomaly("err_count")], "raw2")

    assert acc.pending_count == 1  # same device, one batch

    await asyncio.sleep(0.25)

    callback.assert_called_once()
    batch = callback.call_args[0][0]
    assert batch.categories == ["chassis", "interfaces"]


@pytest.mark.asyncio
async def test_separate_devices_get_separate_batches():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=0.05)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    await acc.submit("r2", "chassis", [_anomaly()], "raw2")

    assert acc.pending_count == 2

    await asyncio.sleep(0.15)

    assert callback.call_count == 2
    devices = {call.args[0].device for call in callback.call_args_list}
    assert devices == {"r1", "r2"}


@pytest.mark.asyncio
async def test_timer_resets_on_new_submit():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=0.15)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    await asyncio.sleep(0.08)
    # Submit again — timer should reset
    await acc.submit("r1", "interfaces", [_anomaly()], "raw2")
    await asyncio.sleep(0.08)

    # Not enough time for the reset timer to fire
    callback.assert_not_called()

    # Now wait for it to fire
    await asyncio.sleep(0.15)
    callback.assert_called_once()
    batch = callback.call_args[0][0]
    assert len(batch.entries) == 2


@pytest.mark.asyncio
async def test_flush_all_dispatches_immediately():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=999.0)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    await acc.submit("r2", "routing", [_anomaly()], "raw2")

    await acc.flush_all()

    assert callback.call_count == 2
    assert acc.pending_count == 0


@pytest.mark.asyncio
async def test_stop_cancels_timers_and_flushes():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=999.0)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    await acc.stop()

    callback.assert_called_once()
    assert acc.pending_count == 0


@pytest.mark.asyncio
async def test_submit_after_flush_creates_fresh_batch():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=0.05)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    await asyncio.sleep(0.15)

    callback.assert_called_once()

    # Submit again — should create a new batch
    await acc.submit("r1", "routing", [_anomaly()], "raw2")
    await asyncio.sleep(0.15)

    assert callback.call_count == 2
    second_batch = callback.call_args_list[1][0][0]
    assert second_batch.categories == ["routing"]


@pytest.mark.asyncio
async def test_callback_exception_doesnt_break_subsequent():
    callback = AsyncMock(side_effect=[RuntimeError("boom"), None])
    acc = AnomalyAccumulator(window_seconds=0.05)
    acc.set_callback(callback)

    await acc.submit("r1", "chassis", [_anomaly()], "raw1")
    await asyncio.sleep(0.15)

    # First callback failed, but accumulator should still work
    await acc.submit("r2", "routing", [_anomaly()], "raw2")
    await asyncio.sleep(0.15)

    assert callback.call_count == 2


@pytest.mark.asyncio
async def test_no_callback_when_nothing_pending():
    callback = AsyncMock()
    acc = AnomalyAccumulator(window_seconds=0.05)
    acc.set_callback(callback)

    await acc.flush_all()
    callback.assert_not_called()

    await acc.stop()
    callback.assert_not_called()


def test_anomaly_batch_categories_property():
    batch = AnomalyBatch(
        device="r1",
        entries=[
            AnomalyEntry(category="chassis", anomalies=[], raw_data=""),
            AnomalyEntry(category="interfaces", anomalies=[], raw_data=""),
        ],
    )
    assert batch.categories == ["chassis", "interfaces"]
