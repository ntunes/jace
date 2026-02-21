"""Tests for AgentCore — conditional LLM analysis gated on anomalies."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.accumulator import AnomalyAccumulator, AnomalyBatch, AnomalyEntry
from jace.agent.anomaly import AnomalyDetector, AnomalyResult
from jace.agent.context import ConversationContext
from jace.agent.core import (
    ANALYSIS_PROMPT_TEMPLATE,
    ANOMALY_PROMPT_TEMPLATE,
    AgentCore,
)
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.agent.metrics_store import MetricPoint, MetricsStore
from jace.checks.registry import CheckRegistry
from jace.config.settings import CorrelationConfig, LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.device.models import CommandResult
from jace.llm.base import Response, Role


def _make_agent(
    *,
    llm: AsyncMock | None = None,
    metrics_store: AsyncMock | None = None,
    anomaly_detector: AsyncMock | None = None,
    findings_tracker: AsyncMock | None = None,
    check_registry: AsyncMock | None = None,
    device_manager: MagicMock | None = None,
    anomaly_accumulator: AnomalyAccumulator | None = None,
    device_schedules: dict | None = None,
) -> AgentCore:
    settings = Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
        device_schedules=device_schedules or {},
    )
    return AgentCore(
        settings=settings,
        llm=llm or AsyncMock(),
        device_manager=device_manager or MagicMock(spec=DeviceManager),
        check_registry=check_registry or AsyncMock(spec=CheckRegistry),
        findings_tracker=findings_tracker or AsyncMock(spec=FindingsTracker),
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
        anomaly_accumulator=anomaly_accumulator,
    )


def _make_finding(
    device: str = "r1", severity: Severity = Severity.WARNING,
    category: str = "interfaces", title: str = "High error rate",
    detail: str = "CRC errors rising",
) -> Finding:
    return Finding(
        id="abc123", device=device, severity=severity,
        category=category, title=title, detail=detail,
        recommendation="Check optics", first_seen="2024-01-01",
        last_seen="2024-01-01",
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


# ---------- Cross-context enrichment tests ----------


@pytest.mark.asyncio
async def test_gather_context_includes_same_device_cross_category():
    """Active findings on the same device (different category) should appear."""
    findings_tracker = MagicMock(spec=FindingsTracker)
    findings_tracker.get_active = MagicMock(side_effect=lambda **kw: {
        # First call: same-device findings
        None: [
            _make_finding(device="r1", category="interfaces", title="CRC errors"),
            _make_finding(device="r1", category="chassis", title="Fan alarm"),
        ],
        # Severity-filtered calls
        Severity.CRITICAL: [],
        Severity.WARNING: [],
    }.get(kw.get("severity")) if kw.get("severity") else [
        _make_finding(device="r1", category="interfaces", title="CRC errors"),
        _make_finding(device="r1", category="chassis", title="Fan alarm"),
    ])

    agent = _make_agent(findings_tracker=findings_tracker)
    ctx = agent._gather_investigation_context("r1", "chassis")

    assert "interfaces" in ctx
    assert "CRC errors" in ctx
    # Should exclude the current category
    assert "Fan alarm" not in ctx


@pytest.mark.asyncio
async def test_gather_context_includes_fleet_critical_warning():
    """Critical/warning findings on other devices should appear."""
    fleet_finding = _make_finding(
        device="r2", severity=Severity.CRITICAL,
        category="routing", title="BGP down",
    )
    same_device_finding = _make_finding(
        device="r1", severity=Severity.CRITICAL,
        category="routing", title="OSPF flap",
    )

    findings_tracker = MagicMock(spec=FindingsTracker)

    def fake_get_active(**kw):
        device = kw.get("device")
        severity = kw.get("severity")
        if device == "r1" and severity is None:
            return []
        if severity == Severity.CRITICAL:
            return [fleet_finding, same_device_finding]
        if severity == Severity.WARNING:
            return []
        return []

    findings_tracker.get_active = MagicMock(side_effect=fake_get_active)

    agent = _make_agent(findings_tracker=findings_tracker)
    ctx = agent._gather_investigation_context("r1", "interfaces")

    assert "r2" in ctx
    assert "BGP down" in ctx
    # Same-device findings from fleet query should be excluded
    assert "OSPF flap" not in ctx


@pytest.mark.asyncio
async def test_gather_context_excludes_info_fleet():
    """Info-severity findings on other devices should NOT appear."""
    findings_tracker = MagicMock(spec=FindingsTracker)

    def fake_get_active(**kw):
        severity = kw.get("severity")
        if severity == Severity.CRITICAL:
            return []
        if severity == Severity.WARNING:
            return []
        return []

    findings_tracker.get_active = MagicMock(side_effect=fake_get_active)

    agent = _make_agent(findings_tracker=findings_tracker)
    ctx = agent._gather_investigation_context("r1", "chassis")
    assert ctx == ""


@pytest.mark.asyncio
async def test_gather_context_returns_empty_when_no_findings():
    """No active findings → empty string."""
    findings_tracker = MagicMock(spec=FindingsTracker)
    findings_tracker.get_active = MagicMock(return_value=[])

    agent = _make_agent(findings_tracker=findings_tracker)
    ctx = agent._gather_investigation_context("r1", "chassis")
    assert ctx == ""


# ---------- Accumulator integration tests ----------


@pytest.mark.asyncio
async def test_run_check_submits_to_accumulator():
    """When accumulator is present, anomalies go to accumulator (not LLM)."""
    llm = AsyncMock()
    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[_sample_anomaly()])

    accumulator = AsyncMock(spec=AnomalyAccumulator)

    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.get_active = MagicMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
        findings_tracker=findings_tracker,
        anomaly_accumulator=accumulator,
    )

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=95.0, unit="C", tags={}, is_counter=False),
    ])}):
        await agent._run_check("chassis", "test-router")

    accumulator.submit.assert_called_once()
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_check_user_triggered_bypasses_accumulator():
    """User-triggered checks bypass the accumulator and call LLM directly."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(content="[]", stop_reason="end_turn"))

    registry = AsyncMock(spec=CheckRegistry)
    registry.run_category = AsyncMock(return_value=_sample_results())

    metrics_store = AsyncMock(spec=MetricsStore)
    metrics_store.record_many = AsyncMock()

    anomaly_detector = AsyncMock(spec=AnomalyDetector)
    anomaly_detector.check_many = AsyncMock(return_value=[_sample_anomaly()])

    accumulator = AsyncMock(spec=AnomalyAccumulator)

    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.get_active = MagicMock(return_value=[])
    findings_tracker.add_or_update = AsyncMock()
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(
        llm=llm,
        check_registry=registry,
        metrics_store=metrics_store,
        anomaly_detector=anomaly_detector,
        findings_tracker=findings_tracker,
        anomaly_accumulator=accumulator,
    )

    with patch("jace.agent.core.EXTRACTORS", {"chassis": MagicMock(return_value=[
        MagicMock(metric="cpu_temp", value=95.0, unit="C", tags={}, is_counter=False),
    ])}):
        await agent._run_check("chassis", "test-router", _user_triggered=True)

    accumulator.submit.assert_not_called()
    llm.chat.assert_called()


# ---------- Batch investigation tests ----------


@pytest.mark.asyncio
async def test_investigate_anomaly_batch_calls_llm():
    """_investigate_anomaly_batch should call LLM with correlated prompt."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=Response(content="[]", stop_reason="end_turn"))

    findings_tracker = MagicMock(spec=FindingsTracker)
    findings_tracker.get_active = MagicMock(return_value=[])
    findings_tracker.add_or_update = AsyncMock()
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(llm=llm, findings_tracker=findings_tracker)

    batch = AnomalyBatch(device="r1", entries=[
        AnomalyEntry(
            category="chassis",
            anomalies=[_sample_anomaly()],
            raw_data="chassis data",
        ),
        AnomalyEntry(
            category="interfaces",
            anomalies=[AnomalyResult(
                metric="err_count", value=500, mean=10,
                stddev=20, z_score=24.5, unit="",
            )],
            raw_data="interface data",
        ),
    ])

    await agent._investigate_anomaly_batch(batch)

    llm.chat.assert_called()
    call_kwargs = llm.chat.call_args
    messages = call_kwargs.kwargs["messages"]
    prompt_text = messages[-1].content
    assert "chassis" in prompt_text
    assert "interfaces" in prompt_text
    assert "common root" in prompt_text.lower() or "holistically" in prompt_text.lower()


@pytest.mark.asyncio
async def test_process_batch_analysis_routes_by_category():
    """Findings with category field are routed to correct tracker bucket."""
    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.add_or_update = AsyncMock(
        return_value=(_make_finding(), True),
    )
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(findings_tracker=findings_tracker)

    analysis = json.dumps([
        {"category": "chassis", "severity": "warning",
         "title": "High temp", "detail": "d", "recommendation": "r"},
        {"category": "interfaces", "severity": "critical",
         "title": "Link errors", "detail": "d", "recommendation": "r"},
    ])

    await agent._process_batch_analysis("r1", ["chassis", "interfaces"], analysis)

    # add_or_update called twice — once per finding
    assert findings_tracker.add_or_update.call_count == 2
    categories_used = [
        call.kwargs["category"]
        for call in findings_tracker.add_or_update.call_args_list
    ]
    assert "chassis" in categories_used
    assert "interfaces" in categories_used

    # resolve_missing called once per category
    assert findings_tracker.resolve_missing.call_count == 2


@pytest.mark.asyncio
async def test_process_batch_analysis_fallback_category():
    """Findings without category field fall back to first category."""
    findings_tracker = AsyncMock(spec=FindingsTracker)
    findings_tracker.add_or_update = AsyncMock(
        return_value=(_make_finding(), True),
    )
    findings_tracker.resolve_missing = AsyncMock(return_value=[])

    agent = _make_agent(findings_tracker=findings_tracker)

    analysis = json.dumps([
        {"severity": "warning", "title": "Something wrong",
         "detail": "d", "recommendation": "r"},
    ])

    await agent._process_batch_analysis("r1", ["chassis", "interfaces"], analysis)

    call = findings_tracker.add_or_update.call_args
    assert call.kwargs["category"] == "chassis"  # fallback to first


# ---------- Prompt memory instruction tests ----------


def test_anomaly_prompt_contains_memory_instructions():
    """ANOMALY_PROMPT_TEMPLATE should mention read_memory and save_memory."""
    assert "read_memory" in ANOMALY_PROMPT_TEMPLATE
    assert "save_memory" in ANOMALY_PROMPT_TEMPLATE


def test_analysis_prompt_contains_memory_instructions():
    """ANALYSIS_PROMPT_TEMPLATE should mention read_memory and save_memory."""
    assert "read_memory" in ANALYSIS_PROMPT_TEMPLATE
    assert "save_memory" in ANALYSIS_PROMPT_TEMPLATE


# ---------- stop_monitoring integration ----------


@pytest.mark.asyncio
async def test_stop_monitoring_calls_accumulator_stop():
    """stop_monitoring should call accumulator.stop() before stopping scheduler."""
    accumulator = AsyncMock(spec=AnomalyAccumulator)
    agent = _make_agent(anomaly_accumulator=accumulator)

    await agent.stop_monitoring()

    accumulator.stop.assert_awaited_once()


# ---------- CorrelationConfig defaults ----------


def test_correlation_config_defaults():
    config = CorrelationConfig()
    assert config.enabled is True
    assert config.window_seconds == 30.0


# ---------- get_chat_history tests ----------


def test_get_chat_history_filters_user_assistant():
    """get_chat_history should only return user and assistant messages."""
    agent = _make_agent()
    ctx = agent._interactive_ctx
    ctx.add_user("hello")
    ctx.add_assistant(
        MagicMock(role=Role.ASSISTANT, content="hi", tool_calls=None),
    )
    ctx.add_tool_result("call-1", "tool result")
    ctx.add_user("follow up")

    history = agent.get_chat_history()
    assert len(history) == 3
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi"}
    assert history[2] == {"role": "user", "content": "follow up"}


def test_get_chat_history_respects_limit():
    """get_chat_history should respect the limit parameter."""
    agent = _make_agent()
    ctx = agent._interactive_ctx
    for i in range(10):
        ctx.add_user(f"msg {i}")

    history = agent.get_chat_history(limit=3)
    assert len(history) == 3
    assert history[0]["content"] == "msg 7"
    assert history[2]["content"] == "msg 9"


def test_get_chat_history_empty():
    """get_chat_history should return empty list when no messages."""
    agent = _make_agent()
    assert agent.get_chat_history() == []


# ---------- device_schedules wiring ----------


def test_scheduler_receives_device_schedules():
    """Scheduler should receive device_schedules from settings."""
    sched = ScheduleConfig(chassis=600, interfaces=300)
    agent = _make_agent(device_schedules={"lab/r1": sched})

    assert agent._scheduler._device_schedules == {"lab/r1": sched}
    assert agent._scheduler._default_intervals["chassis"] == 300  # from default ScheduleConfig


def test_scheduler_no_device_schedules():
    """Scheduler should work without device_schedules."""
    agent = _make_agent()
    assert agent._scheduler._device_schedules == {}
