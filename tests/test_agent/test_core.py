"""Tests for AgentCore — conditional LLM analysis gated on anomalies."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.anomaly import AnomalyDetector, AnomalyResult
from jace.agent.context import ConversationContext
from jace.agent.core import AgentCore
from jace.agent.findings import FindingsTracker
from jace.agent.metrics_store import MetricPoint, MetricsStore
from jace.checks.registry import CheckRegistry
from jace.config.settings import LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.device.models import CommandResult
from jace.llm.base import Response


def _make_agent(
    *,
    llm: AsyncMock | None = None,
    metrics_store: AsyncMock | None = None,
    anomaly_detector: AsyncMock | None = None,
    findings_tracker: AsyncMock | None = None,
    check_registry: AsyncMock | None = None,
    device_manager: MagicMock | None = None,
) -> AgentCore:
    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )
    return AgentCore(
        settings=settings,
        llm=llm or AsyncMock(),
        device_manager=device_manager or MagicMock(spec=DeviceManager),
        check_registry=check_registry or AsyncMock(spec=CheckRegistry),
        findings_tracker=findings_tracker or AsyncMock(spec=FindingsTracker),
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
    )


def _sample_results() -> dict[str, CommandResult]:
    return {
        "show chassis alarms": CommandResult(
            command="show chassis alarms",
            output="No alarms currently active",
            driver_used="pyez",
            success=True,
        ),
    }


def _sample_anomaly() -> AnomalyResult:
    return AnomalyResult(
        metric="cpu_temp", value=95.0, mean=60.0,
        stddev=5.0, z_score=7.0, unit="C",
    )


@pytest.mark.asyncio
async def test_run_check_skips_llm_when_no_anomalies():
    """When extractor exists and no anomalies, LLM should not be called."""
    llm = AsyncMock()
    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
    )

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=60.0, unit="C", tags={}, is_counter=False),
    ])}):
        await agent._run_check("chassis", "test-router")

    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_check_calls_llm_on_anomalies():
    """When anomalies are detected, LLM should be called with anomaly prompt."""
    anomaly = _sample_anomaly()

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(content="[]", stop_reason="end_turn"))

    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[anomaly])

    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.add_or_update = AsyncMock()
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
        findings_tracker=findings_tracker,
    )

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=95.0, unit="C", tags={}, is_counter=False),
    ])}):
        await agent._run_check("chassis", "test-router")

    llm.chat.assert_called()
    # Verify the prompt contains anomaly context
    call_kwargs = llm.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") or call_kwargs[0][0] if call_kwargs[0] else None
    if messages is None:
        messages = call_kwargs.kwargs["messages"]
    prompt_text = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    assert "anomal" in prompt_text.lower()


@pytest.mark.asyncio
async def test_run_check_always_calls_llm_for_config():
    """Config category has no extractor — LLM should always be called."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(content="[]", stop_reason="end_turn"))

    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.add_or_update = AsyncMock()
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        findings_tracker=findings_tracker,
    )

    # "config" is not in EXTRACTORS so _extract_and_check_metrics returns []
    await agent._run_check("config", "test-router")

    llm.chat.assert_called()


@pytest.mark.asyncio
async def test_run_check_always_stores_metrics():
    """Metrics should be stored even when no anomalies are detected."""
    llm = AsyncMock()
    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
    )

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=60.0, unit="C", tags={}, is_counter=False),
    ])}):
        await agent._run_check("chassis", "test-router")

    metrics_store.record_many.assert_called_once()


@pytest.mark.asyncio
async def test_run_check_logs_normal_status(caplog):
    """When no anomalies, a 'Normal' log message should appear."""
    llm = AsyncMock()
    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
    )

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=60.0, unit="C", tags={}, is_counter=False),
    ])}):
        with caplog.at_level(logging.INFO, logger="jace.agent.core"):
            await agent._run_check("chassis", "test-router")

    assert any("Normal" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_extract_returns_anomaly_list():
    """_extract_and_check_metrics should return list[AnomalyResult]."""
    anomaly = _sample_anomaly()

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[anomaly])

    agent = _make_agent(
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
    )

    results = _sample_results()

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=95.0, unit="C", tags={}, is_counter=False),
    ])}):
        result = await agent._extract_and_check_metrics("chassis", "r1", results)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], AnomalyResult)
    assert result[0].metric == "cpu_temp"
