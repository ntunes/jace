"""Tool definitions shared across LLM backends."""

from __future__ import annotations

from jace.llm.base import ToolDefinition

AGENT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="run_command",
        description=(
            "Execute a Junos operational command on a device. "
            "Use this for any 'show' command or operational mode command. "
            "Examples: 'show chassis alarms', 'show bgp summary', 'show interfaces terse'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Name of the target device (e.g., 'mx-core-01')",
                },
                "command": {
                    "type": "string",
                    "description": "The Junos operational command to execute",
                },
            },
            "required": ["device", "command"],
        },
    ),
    ToolDefinition(
        name="get_config",
        description=(
            "Retrieve device configuration. Can retrieve full configuration "
            "or a specific section. Supports text and set formats."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Name of the target device",
                },
                "section": {
                    "type": "string",
                    "description": "Configuration section to retrieve (e.g., 'protocols bgp', 'interfaces'). Omit for full config.",
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "set", "xml"],
                    "description": "Output format (default: text)",
                },
            },
            "required": ["device"],
        },
    ),
    ToolDefinition(
        name="get_device_facts",
        description=(
            "Get device information including model, Junos version, "
            "serial number, hostname, and uptime."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Name of the target device",
                },
            },
            "required": ["device"],
        },
    ),
    ToolDefinition(
        name="list_devices",
        description=(
            "List all managed devices and their current connection status, "
            "model info, and last check time."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="get_findings",
        description=(
            "Retrieve current and historical findings from health checks. "
            "Filter by device, severity, or category."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Filter by device name (optional)",
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "warning", "info"],
                    "description": "Filter by severity (optional)",
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category (optional)",
                },
                "include_resolved": {
                    "type": "boolean",
                    "description": "Include resolved findings (default: false)",
                },
            },
        },
    ),
    ToolDefinition(
        name="run_health_check",
        description=(
            "Trigger a specific health check category immediately. "
            "Categories: chassis, interfaces, routing, system, config."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Target device name",
                },
                "category": {
                    "type": "string",
                    "enum": ["chassis", "interfaces", "routing", "system", "config"],
                    "description": "Health check category to run",
                },
            },
            "required": ["device", "category"],
        },
    ),
    ToolDefinition(
        name="get_metrics",
        description=(
            "Retrieve time-series metrics for a device. "
            "Omit the 'metric' parameter to list all available metric names. "
            "Provide a metric name to get its recent values."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Name of the target device",
                },
                "metric": {
                    "type": "string",
                    "description": "Metric name to query (omit to list available metrics)",
                },
                "since_hours": {
                    "type": "integer",
                    "description": "How many hours of history to retrieve (default: 24)",
                },
            },
            "required": ["device"],
        },
    ),
    ToolDefinition(
        name="compare_config",
        description=(
            "Compare the current device configuration against a previous "
            "rollback version. Shows differences."
        ),
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Name of the target device",
                },
                "rollback": {
                    "type": "integer",
                    "description": "Rollback number to compare against (0-49, default: 1)",
                },
            },
            "required": ["device"],
        },
    ),
    ToolDefinition(
        name="manage_heartbeat",
        description=(
            "Manage the heartbeat checklist — periodic monitoring instructions "
            "that JACE executes autonomously. Use 'list' to view, 'add' to "
            "append, 'remove' to delete by number, 'replace' to overwrite all."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "remove", "replace"],
                    "description": "The action to perform",
                },
                "instruction": {
                    "type": "string",
                    "description": (
                        "The instruction text (for add) or full new "
                        "content (for replace)"
                    ),
                },
                "index": {
                    "type": "integer",
                    "description": (
                        "1-based line number to remove (for remove action)"
                    ),
                },
            },
            "required": ["action"],
        },
    ),
]
