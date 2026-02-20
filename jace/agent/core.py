"""Autonomous agent core — background checks + interactive handler."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

from jace.agent.anomaly import AnomalyDetector, AnomalyResult
from jace.agent.context import ConversationContext
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.agent.heartbeat import HeartbeatManager
from jace.agent.metrics_store import MetricPoint, MetricsStore
from jace.agent.scheduler import Scheduler
from jace.checks.registry import CheckRegistry
from jace.config.settings import Settings
from jace.device.manager import DeviceManager
from jace.llm.base import LLMClient, Message, Response, Role, ToolCall
from jace.llm.tools import AGENT_TOOLS
from jace.metrics import EXTRACTORS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert Junos network engineer AI agent. You autonomously monitor \
Junos MX series routers, analyze health data, troubleshoot issues, and audit \
configurations.

When analyzing device output:
- Identify anomalies, errors, and deviations from best practices
- Provide specific, actionable recommendations
- Reference relevant Junos documentation or KB articles when applicable
- Consider the impact severity: critical (service-affecting), warning \
(potential issue), info (notable but benign)

When responding to user queries:
- Use the available tools to gather data before answering
- Show your work — explain which commands you ran and why
- Be concise but thorough

You have access to tools for running commands on Junos devices, retrieving \
configurations, and checking findings from health monitors.\
"""

ANALYSIS_PROMPT_TEMPLATE = """\
Analyze the following health check data from device '{device}' \
(category: {category}).

For each issue found, respond with a JSON array of findings. Each finding \
should have these fields:
- severity: "critical", "warning", or "info"
- title: short summary (one line)
- detail: explanation of the issue
- recommendation: suggested action to resolve

If no issues are found, return an empty array: []

Raw data:
{data}\
"""

ANOMALY_PROMPT_TEMPLATE = """\
Statistical anomalies detected on device '{device}' ({category} check).

Detected anomalies:
{anomalies}

Raw command output:
{data}

First, review the data above and decide whether the anomalies can already be \
explained as benign (e.g. a scheduled maintenance window, a counter reset, or \
normal daily variance). If so, return an empty JSON array: []

Otherwise, troubleshoot by running commands on the device. Suggested sequence:
1. Use run_command to gather correlated counters or logs \
(e.g. "show log messages", "show interfaces diagnostics optics", etc.)
2. Use get_config to check whether a recent config change could explain the shift.
3. Use get_metrics to pull historical trends for the affected metrics.

After investigating, respond with a JSON array of findings:
- severity: "critical", "warning", or "info"
- title: short summary (one line)
- detail: explanation including evidence from the commands you ran
- recommendation: suggested action to resolve\
"""

HEARTBEAT_PROMPT_TEMPLATE = """\
You are running a scheduled heartbeat check. Evaluate each instruction below \
using your available tools. Check all connected devices as appropriate.

IMPORTANT: Prefer already-collected data first. Use get_findings and get_metrics \
to check existing health check results and historical metrics before running \
new commands on devices. Only use run_command or get_config when the existing \
data is insufficient to evaluate an instruction.

If everything is normal, respond with an empty JSON array: []
If you find issues, respond with a JSON array of findings (same format as \
health checks: severity, title, detail, recommendation).

Heartbeat instructions:
{instructions}\
"""

# Notification callback type
NotifyCallback = Callable[[Finding, bool], Awaitable[None]]  # (finding, is_new)


class AgentCore:
    """Main agent — runs background health checks and handles interactive queries."""

    def __init__(self, settings: Settings, llm: LLMClient,
                 device_manager: DeviceManager,
                 check_registry: CheckRegistry,
                 findings_tracker: FindingsTracker,
                 metrics_store: MetricsStore | None = None,
                 anomaly_detector: AnomalyDetector | None = None,
                 heartbeat_manager: HeartbeatManager | None = None) -> None:
        self._settings = settings
        self._llm = llm
        self._device_manager = device_manager
        self._registry = check_registry
        self._findings = findings_tracker
        self._metrics_store = metrics_store
        self._anomaly_detector = anomaly_detector
        self._heartbeat_manager = heartbeat_manager
        self._scheduler = Scheduler(settings.schedule)
        self._interactive_ctx = ConversationContext()
        self._notify_callback: NotifyCallback | None = None
        self._heartbeat_task: asyncio.Task | None = None

    def set_notify_callback(self, callback: NotifyCallback) -> None:
        self._notify_callback = callback

    def start_monitoring(self) -> None:
        """Start the background health check scheduler."""
        devices = self._device_manager.get_connected_devices()
        if not devices:
            logger.warning("No connected devices — scheduler not started")
            return
        self._scheduler.start(devices, self._run_check)
        logger.info("Monitoring started for devices: %s", devices)

        if (self._settings.heartbeat.enabled
                and self._heartbeat_manager is not None):
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(self._settings.heartbeat.interval),
                name="heartbeat",
            )
            logger.info("Heartbeat loop started (interval=%ds)",
                         self._settings.heartbeat.interval)

    async def stop_monitoring(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("Heartbeat loop stopped")
        await self._scheduler.stop()

    async def handle_user_input(self, user_input: str) -> str:
        """Process a user query through the LLM with tool use."""
        self._interactive_ctx.add_user(user_input)

        response = await self._llm_tool_loop(self._interactive_ctx)
        return response

    async def _run_check(self, category: str, device_name: str) -> None:
        """Run a health check category and analyze results with the LLM."""
        logger.info("Running %s check on %s", category, device_name)

        results = await self._registry.run_category(
            category, self._device_manager, device_name,
        )

        if not results:
            return

        # Always extract metrics and check anomalies (builds baseline)
        anomalies = await self._extract_and_check_metrics(
            category, device_name, results,
        )

        # Format raw data
        data_parts = []
        for cmd, result in results.items():
            status = "SUCCESS" if result.success else f"FAILED: {result.error}"
            data_parts.append(f"--- {cmd} [{status}] ---\n{result.output}\n")
        data_text = "\n".join(data_parts)

        # Decide whether to call the LLM
        has_extractor = category in EXTRACTORS

        if has_extractor and not anomalies:
            # Normal — log status and skip LLM
            logger.info("%s check on %s: Normal", category, device_name)
            return

        # Anomalies detected or non-metric category (config) — call LLM
        if anomalies:
            logger.info("%s check on %s: %d anomaly(s) detected",
                        category, device_name, len(anomalies))
            anomaly_text = "\n".join(a.to_context_line() for a in anomalies)
            prompt = ANOMALY_PROMPT_TEMPLATE.format(
                device=device_name, category=category,
                anomalies=anomaly_text, data=data_text,
            )
        else:
            # Config category — use general template
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                device=device_name, category=category, data=data_text,
            )

        ctx = ConversationContext()
        ctx.add_user(prompt)

        try:
            response_text = await self._llm_tool_loop(ctx)
            await self._process_analysis(device_name, category, response_text)
        except Exception as exc:
            logger.error("LLM analysis failed for %s/%s: %s",
                         category, device_name, exc)

    async def _heartbeat_loop(self, interval: int) -> None:
        """Run heartbeat checks on a schedule."""
        # Stagger initial start
        stagger = hash("heartbeat") % min(30, interval)
        await asyncio.sleep(stagger)

        while True:
            try:
                await self._run_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat cycle failed: %s", exc)
            await asyncio.sleep(interval)

    async def _run_heartbeat(self) -> None:
        """Execute one heartbeat cycle."""
        if self._heartbeat_manager is None:
            return

        instructions = self._heartbeat_manager.get_instructions()
        if not instructions.strip():
            logger.debug("Heartbeat: no instructions configured, skipping")
            return

        logger.info("Running heartbeat cycle")
        prompt = HEARTBEAT_PROMPT_TEMPLATE.format(instructions=instructions)

        ctx = ConversationContext()
        ctx.add_user(prompt)

        try:
            response_text = await self._llm_tool_loop(ctx)
            await self._process_analysis(
                device_name="*", category="heartbeat",
                analysis=response_text,
            )
        except Exception as exc:
            logger.error("Heartbeat LLM analysis failed: %s", exc)

    async def _process_analysis(self, device_name: str, category: str,
                                analysis: str) -> None:
        """Parse LLM analysis and create/update findings."""
        # Extract JSON array from response
        findings_data = self._extract_json_array(analysis)
        current_titles: set[str] = set()

        for item in findings_data:
            title = item.get("title", "Unknown issue")
            current_titles.add(title)

            try:
                severity = Severity(item.get("severity", "info"))
            except ValueError:
                severity = Severity.INFO

            finding, is_new = await self._findings.add_or_update(
                device=device_name,
                severity=severity,
                category=category,
                title=title,
                detail=item.get("detail", ""),
                recommendation=item.get("recommendation", ""),
            )

            if self._notify_callback and is_new:
                await self._notify_callback(finding, is_new)

        # Resolve findings that are no longer reported
        resolved = await self._findings.resolve_missing(
            device_name, category, current_titles,
        )
        for finding in resolved:
            if self._notify_callback:
                await self._notify_callback(finding, False)

    async def _extract_and_check_metrics(
        self, category: str, device_name: str,
        results: dict[str, Any],
    ) -> list[AnomalyResult]:
        """Extract metrics from results, store them, and check for anomalies."""
        if not self._metrics_store:
            return []

        extractor = EXTRACTORS.get(category)
        if not extractor:
            return []

        try:
            extracted = extractor(results)
        except Exception as exc:
            logger.error("Metric extraction failed for %s/%s: %s",
                         category, device_name, exc)
            return []

        if not extracted:
            return []

        points: list[MetricPoint] = []
        for em in extracted:
            point = MetricPoint(
                device=device_name, category=category,
                metric=em.metric, value=em.value,
                unit=em.unit, tags=em.tags,
            )
            points.append(point)

            # For counters, compute delta from previous value
            if em.is_counter:
                prev = await self._metrics_store.latest(device_name, em.metric)
                if prev is not None:
                    delta = max(0.0, em.value - prev.value)
                    delta_point = MetricPoint(
                        device=device_name, category=category,
                        metric=f"{em.metric}_delta", value=delta,
                        unit=em.unit, tags=em.tags,
                    )
                    points.append(delta_point)

        await self._metrics_store.record_many(points)

        # Check for anomalies
        if not self._anomaly_detector:
            return []

        anomalies = await self._anomaly_detector.check_many(device_name, points)
        return anomalies

    async def _llm_tool_loop(self, ctx: ConversationContext,
                             max_iterations: int = 10) -> str:
        """Run the LLM tool-use loop until the LLM produces a final text response."""
        for _ in range(max_iterations):
            response = await self._llm.chat(
                messages=ctx.messages,
                tools=AGENT_TOOLS,
                system=self._settings.llm.system_prompt or SYSTEM_PROMPT,
                max_tokens=self._settings.llm.max_tokens,
            )

            if not response.has_tool_calls:
                # Final text response
                if response.content:
                    ctx.add_assistant(Message(
                        role=Role.ASSISTANT, content=response.content,
                    ))
                return response.content

            # Process tool calls
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            ctx.add_assistant(assistant_msg)

            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                ctx.add_tool_result(tool_call.id, result)

        return "Maximum tool iterations reached."

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string."""
        name = tool_call.name
        args = tool_call.arguments

        try:
            if name == "run_command":
                result = await self._device_manager.run_command(
                    args["device"], args["command"],
                )
                if result.success:
                    return result.output or "(no output)"
                return f"Error: {result.error}"

            elif name == "get_config":
                return await self._device_manager.get_config(
                    args["device"],
                    section=args.get("section"),
                    format=args.get("format", "text"),
                )

            elif name == "get_device_facts":
                facts = await self._device_manager.get_facts(args["device"])
                return json.dumps(facts, indent=2, default=str)

            elif name == "list_devices":
                devices = self._device_manager.list_devices()
                return json.dumps(
                    [{"name": d.name, "host": d.host, "status": d.status.value,
                      "model": d.model, "version": d.version}
                     for d in devices],
                    indent=2,
                )

            elif name == "get_findings":
                findings = self._findings.get_active(
                    device=args.get("device"),
                    severity=Severity(args["severity"]) if "severity" in args else None,
                    category=args.get("category"),
                )
                return json.dumps([f.to_dict() for f in findings], indent=2)

            elif name == "run_health_check":
                await self._run_check(args["category"], args["device"])
                findings = self._findings.get_active(
                    device=args["device"], category=args["category"],
                )
                if findings:
                    return json.dumps([f.to_dict() for f in findings], indent=2)
                return "Health check completed. No issues found."

            elif name == "get_metrics":
                if not self._metrics_store:
                    return "Metrics store not configured."
                device = args["device"]
                metric = args.get("metric")
                if not metric:
                    names = await self._metrics_store.list_metrics(device)
                    if not names:
                        return "No metrics recorded for this device yet."
                    return json.dumps(names, indent=2)
                since = args.get("since_hours", 24)
                points = await self._metrics_store.query(
                    device, metric, since_hours=since,
                )
                if not points:
                    return f"No data for metric '{metric}' in the last {since}h."
                return json.dumps([p.to_dict() for p in points], indent=2)

            elif name == "compare_config":
                rollback = args.get("rollback", 1)
                result = await self._device_manager.run_command(
                    args["device"],
                    f"show configuration | compare rollback {rollback}",
                )
                return result.output if result.success else f"Error: {result.error}"

            elif name == "manage_heartbeat":
                if not self._heartbeat_manager:
                    return "Heartbeat not configured."
                action = args["action"]
                if action == "list":
                    return self._heartbeat_manager.list_instructions()
                elif action == "add":
                    return self._heartbeat_manager.add_instruction(
                        args["instruction"],
                    )
                elif action == "remove":
                    return self._heartbeat_manager.remove_instruction(
                        args["index"],
                    )
                elif action == "replace":
                    return self._heartbeat_manager.replace_instructions(
                        args["instruction"],
                    )
                return f"Unknown heartbeat action: {action}"

            else:
                return f"Unknown tool: {name}"

        except Exception as exc:
            logger.error("Tool execution error (%s): %s", name, exc)
            return f"Tool error: {exc}"

    @staticmethod
    def _extract_json_array(text: str) -> list[dict]:
        """Extract a JSON array from LLM response text."""
        # Try to find JSON array in the response
        text = text.strip()

        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Find JSON array within text (e.g., surrounded by markdown code blocks)
        import re
        patterns = [
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```',
            r'(\[[\s\S]*?\])',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue

        return []
