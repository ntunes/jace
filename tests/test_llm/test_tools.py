"""Tests for LLM tool definitions."""

from jace.llm.tools import AGENT_TOOLS


def test_all_tools_have_required_fields():
    for tool in AGENT_TOOLS:
        assert tool.name, "Tool must have a name"
        assert tool.description, "Tool must have a description"
        assert tool.parameters, "Tool must have parameters"
        assert tool.parameters.get("type") == "object"


def test_tool_names_unique():
    names = [t.name for t in AGENT_TOOLS]
    assert len(names) == len(set(names)), "Tool names must be unique"


def test_expected_tools_exist():
    names = {t.name for t in AGENT_TOOLS}
    expected = {
        "run_command", "get_config", "get_device_facts",
        "list_devices", "get_findings", "run_health_check", "compare_config",
    }
    assert expected <= names
