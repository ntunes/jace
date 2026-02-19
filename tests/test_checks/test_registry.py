"""Tests for check registry."""

from unittest.mock import AsyncMock

from jace.checks.registry import CheckRegistry, build_default_registry
from jace.device.models import CommandResult


def test_registry_register_and_get():
    registry = CheckRegistry()

    async def dummy_check(dm, dev):
        return {"cmd": CommandResult(command="test", output="ok")}

    registry.register("test_category", dummy_check)
    checks = registry.get_checks("test_category")
    assert len(checks) == 1
    assert checks[0] is dummy_check


def test_registry_categories():
    registry = CheckRegistry()

    async def check1(dm, dev):
        return {}

    async def check2(dm, dev):
        return {}

    registry.register("a", check1)
    registry.register("b", check2)
    assert set(registry.categories()) == {"a", "b"}


def test_default_registry_has_all_categories():
    registry = build_default_registry()
    categories = set(registry.categories())
    assert "chassis" in categories
    assert "interfaces" in categories
    assert "routing" in categories
    assert "system" in categories
    assert "config" in categories
