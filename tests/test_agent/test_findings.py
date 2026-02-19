"""Tests for findings tracker."""

import pytest

from jace.agent.findings import FindingsTracker, Severity


@pytest.mark.asyncio
async def test_add_new_finding(findings_tracker: FindingsTracker):
    finding, is_new = await findings_tracker.add_or_update(
        device="r1", severity=Severity.WARNING, category="chassis",
        title="Fan speed high", detail="Fan 1 running at 90%",
        recommendation="Check ambient temperature",
    )
    assert is_new is True
    assert finding.device == "r1"
    assert finding.severity == Severity.WARNING
    assert finding.resolved is False


@pytest.mark.asyncio
async def test_update_existing_finding(findings_tracker: FindingsTracker):
    await findings_tracker.add_or_update(
        device="r1", severity=Severity.WARNING, category="chassis",
        title="Fan speed high", detail="Fan 1 at 90%",
        recommendation="Check temps",
    )
    finding, is_new = await findings_tracker.add_or_update(
        device="r1", severity=Severity.WARNING, category="chassis",
        title="Fan speed high", detail="Fan 1 still at 90%",
        recommendation="Check temps",
    )
    assert is_new is False
    assert finding.detail == "Fan 1 still at 90%"


@pytest.mark.asyncio
async def test_resolve_missing(findings_tracker: FindingsTracker):
    await findings_tracker.add_or_update(
        device="r1", severity=Severity.WARNING, category="chassis",
        title="Fan speed high", detail="", recommendation="",
    )
    assert findings_tracker.active_count == 1

    resolved = await findings_tracker.resolve_missing("r1", "chassis", set())
    assert len(resolved) == 1
    assert resolved[0].resolved is True
    assert findings_tracker.active_count == 0


@pytest.mark.asyncio
async def test_get_active_filtering(findings_tracker: FindingsTracker):
    await findings_tracker.add_or_update(
        device="r1", severity=Severity.CRITICAL, category="routing",
        title="BGP peer down", detail="", recommendation="",
    )
    await findings_tracker.add_or_update(
        device="r2", severity=Severity.INFO, category="system",
        title="CPU normal", detail="", recommendation="",
    )

    critical = findings_tracker.get_active(severity=Severity.CRITICAL)
    assert len(critical) == 1
    assert critical[0].title == "BGP peer down"

    r2_findings = findings_tracker.get_active(device="r2")
    assert len(r2_findings) == 1
