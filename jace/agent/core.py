"""Autonomous agent core — background checks + interactive handler."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

from jace.agent.accumulator import AnomalyAccumulator, AnomalyBatch
from jace.agent.anomaly import AnomalyDetector, AnomalyResult
from jace.agent.context import ConversationContext
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.agent.heartbeat import HeartbeatManager
from jace.agent.memory import MemoryStore
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

You are a hands-on operator, not an advisor. When you detect an issue, \
DO NOT suggest troubleshooting steps for the user — instead, use your tools \
to investigate it yourself immediately. Run the commands, pull the configs, \
check the metrics, and report back with a concrete diagnosis.

When analyzing device output:
- Identify anomalies, errors, and deviations from best practices
- Proactively run additional commands to confirm and diagnose issues
- Reference relevant Junos documentation or KB articles when applicable
- Consider the impact severity: critical (service-affecting), warning \
(potential issue), info (notable but benign)

When responding to user queries:
- Use the available tools to gather data before answering
- Show your work — explain which commands you ran and why
- Be concise but thorough
- Never tell the user to run a command themselves — run it for them

You have access to tools for running commands on Junos devices, retrieving \
configurations, and checking findings from health monitors.

You also have persistent memory across sessions. Use save_memory to store \
important observations (device quirks, baselines, operator preferences, \
incident patterns) and read_memory to recall them. Memory is organized into \
three categories: 'device' (per-device profiles), 'user' (operator \
preferences), and 'incident' (past incident records).\
"""

ANALYSIS_PROMPT_TEMPLATE = """\
Analyze the following health check data from device '{device}' \
(category: {category}).

Before analyzing, use read_memory to check for known device baselines, \
previous incidents on this device, or operator preferences that might \
inform your analysis.

If you spot potential issues in the data, DO NOT just report them — use your \
tools to investigate further. Run correlated commands (run_command), check the \
config (get_config), and pull historical metrics (get_metrics) to confirm the \
issue and determine root cause before creating a finding.

For each confirmed issue, respond with a JSON array of findings:
- severity: "critical", "warning", or "info"
- title: short summary (one line)
- detail: explanation including evidence from the commands you ran
- recommendation: what you did to diagnose, and any remaining action \
the operator should take (e.g. hardware replacement, vendor RMA)

If no issues are found, return an empty array: []

After analyzing, if you discovered any noteworthy patterns (device quirks, \
recurring baselines, incident context), use save_memory to persist them.

Raw data:
{data}\
"""

ANOMALY_PROMPT_TEMPLATE = """\
Statistical anomalies detected on device '{device}' ({category} check).

Before investigating, use read_memory to check for known baselines, \
previous incidents, or device profiles that might explain these anomalies.

Detected anomalies:
{anomalies}

Raw command output:
{data}
{context}\
First, review the data above and decide whether the anomalies can already be \
explained as benign (e.g. a scheduled maintenance window, a counter reset, or \
normal daily variance). If so, return an empty JSON array: []

Otherwise, you MUST investigate by running commands on the device — do not \
simply report the anomaly and suggest what to check. Follow this sequence:
1. Use run_command to gather correlated counters or logs \
(e.g. "show log messages", "show interfaces diagnostics optics", etc.)
2. Use get_config to check whether a recent config change could explain the shift.
3. Use get_metrics to pull historical trends for the affected metrics.

After investigating, respond with a JSON array of findings:
- severity: "critical", "warning", or "info"
- title: short summary (one line)
- detail: root cause analysis including evidence from the commands you ran
- recommendation: what you did to diagnose, and any remaining action \
only the operator can take (e.g. hardware replacement, vendor RMA)

After investigating, use save_memory to persist any new device baselines, \
incident patterns, or root cause information you discovered.\
"""

CORRELATED_ANOMALY_PROMPT_TEMPLATE = """\
Multiple anomaly categories detected simultaneously on device '{device}' \
(categories: {categories}).

Before investigating, use read_memory to check for known baselines, \
previous incidents, or device profiles that might explain these anomalies.

{category_blocks}
{context}\
These anomalies fired within a short time window and may share a common root \
cause. Investigate holistically — look for a single underlying issue before \
treating them as independent problems.

You MUST investigate by running commands on the device — do not simply report \
the anomalies and suggest what to check. Follow this sequence:
1. Use run_command to gather correlated counters or logs \
(e.g. "show log messages", "show interfaces diagnostics optics", etc.)
2. Use get_config to check whether a recent config change could explain the shift.
3. Use get_metrics to pull historical trends for the affected metrics.

After investigating, respond with a JSON array of findings. Each finding \
must include a "category" field indicating which check category it belongs to \
(one of: {categories}):
- category: the check category this finding belongs to
- severity: "critical", "warning", or "info"
- title: short summary (one line)
- detail: root cause analysis including evidence from the commands you ran
- recommendation: what you did to diagnose, and any remaining action \
only the operator can take (e.g. hardware replacement, vendor RMA)

If all anomalies are benign, return an empty JSON array: []

After investigating, use save_memory to persist any new device baselines, \
incident patterns, or root cause information you discovered.\
"""

HEARTBEAT_PROMPT_TEMPLATE = """\
You are running a scheduled heartbeat check. Evaluate each instruction below \
using your available tools. Check all connected devices as appropriate.

IMPORTANT: Prefer already-collected data first. Use get_findings and get_metrics \
to check existing health check results and historical metrics before running \
new commands on devices. Only use run_command or get_config when the existing \
data is insufficient to evaluate an instruction.

If you find a potential issue, investigate it fully before creating a finding. \
Run additional commands to confirm and diagnose — do not just flag it and \
suggest the operator investigate.

If everything is normal, respond with an empty JSON array: []
If you find confirmed issues, respond with a JSON array of findings:
- severity: "critical", "warning", or "info"
- title: short summary (one line)
- detail: root cause analysis including evidence from the commands you ran
- recommendation: what you did to diagnose, and any remaining action \
only the operator can take (e.g. hardware replacement, vendor RMA)

Heartbeat instructions:
{instructions}\
"""

MEMORY_FLUSH_PROMPT = """\
Review the conversation above and save anything important to persistent memory \
using the save_memory tool. Focus on:
- Device quirks, baselines, or patterns learned during troubleshooting
- Operator preferences (alert thresholds, output formats, workflow habits)
- Incident details with root causes and resolutions

Only save genuinely useful observations. If nothing is worth saving, do nothing.\
"""

SUMMARIZE_PROMPT = """\
Summarize the conversation above in a compact paragraph. Focus on:
- What devices were discussed and their current state
- Any issues found, diagnosed, or resolved
- Pending actions or open questions
- Key decisions made

Be concise — this summary will replace older messages to free context space.\
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
                 heartbeat_manager: HeartbeatManager | None = None,
                 memory_store: MemoryStore | None = None,
                 anomaly_accumulator: AnomalyAccumulator | None = None) -> None:
        self._settings = settings
        self._llm = llm
        self._device_manager = device_manager
        self._registry = check_registry
        self._findings = findings_tracker
        self._metrics_store = metrics_store
        self._anomaly_detector = anomaly_detector
        self._heartbeat_manager = heartbeat_manager
        self._memory_store = memory_store
        self._accumulator = anomaly_accumulator
        if self._accumulator is not None:
            self._accumulator.set_callback(self._investigate_anomaly_batch)
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
        if self._accumulator is not None:
            await self._accumulator.stop()
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

    async def _run_check(self, category: str, device_name: str,
                         *, _user_triggered: bool = False) -> None:
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

            # Route through accumulator if available and not user-triggered
            if self._accumulator is not None and not _user_triggered:
                await self._accumulator.submit(
                    device_name, category, anomalies, data_text,
                )
                return

            anomaly_text = "\n".join(a.to_context_line() for a in anomalies)
            context = self._gather_investigation_context(
                device_name, category,
            )
            prompt = ANOMALY_PROMPT_TEMPLATE.format(
                device=device_name, category=category,
                anomalies=anomaly_text, data=data_text,
                context=context,
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

    def _gather_investigation_context(
        self, device_name: str, category: str | None = None,
    ) -> str:
        """Build a context block from active findings on the same device
        and critical/warning findings fleet-wide.

        When *category* is ``None`` (batch mode), all same-device findings
        are included.
        """
        lines: list[str] = []

        # Same-device findings (exclude current category if specified)
        same_device = self._findings.get_active(device=device_name)
        for f in same_device:
            if category is not None and f.category == category:
                continue
            lines.append(
                f"  [{f.severity.value.upper()}] {f.category}: "
                f"{f.title} — {f.detail}"
            )

        # Fleet-wide critical/warning findings on other devices
        for sev in (Severity.CRITICAL, Severity.WARNING):
            for f in self._findings.get_active(severity=sev):
                if f.device == device_name:
                    continue
                lines.append(
                    f"  [{f.severity.value.upper()}] {f.device}/{f.category}: "
                    f"{f.title}"
                )

        if not lines:
            return ""

        header = "\nRelated active findings across the network:\n"
        return header + "\n".join(lines) + "\n\n"

    async def _investigate_anomaly_batch(self, batch: AnomalyBatch) -> None:
        """Investigate a batch of correlated anomalies for a single device."""
        device = batch.device
        categories = batch.categories
        logger.info("Investigating correlated anomalies on %s: %s",
                     device, categories)

        # Build per-category blocks
        category_blocks: list[str] = []
        for entry in batch.entries:
            anomaly_text = "\n".join(
                a.to_context_line() for a in entry.anomalies
            )
            block = (
                f"=== {entry.category} ===\n"
                f"Detected anomalies:\n{anomaly_text}\n\n"
                f"Raw command output:\n{entry.raw_data}\n"
            )
            category_blocks.append(block)

        context = self._gather_investigation_context(device, category=None)
        categories_str = ", ".join(categories)

        prompt = CORRELATED_ANOMALY_PROMPT_TEMPLATE.format(
            device=device,
            categories=categories_str,
            category_blocks="\n".join(category_blocks),
            context=context,
        )

        ctx = ConversationContext()
        ctx.add_user(prompt)

        try:
            response_text = await self._llm_tool_loop(ctx)
            await self._process_batch_analysis(
                device, categories, response_text,
            )
        except Exception as exc:
            logger.error("Correlated anomaly investigation failed for %s: %s",
                         device, exc)

    async def _process_batch_analysis(
        self, device_name: str, categories: list[str], analysis: str,
    ) -> None:
        """Parse LLM analysis from a batched investigation and route
        findings to the correct category trackers."""
        findings_data = self._extract_json_array(analysis)

        # Group findings by category
        by_category: dict[str, list[dict]] = {c: [] for c in categories}
        fallback_category = categories[0] if categories else "unknown"

        for item in findings_data:
            cat = item.get("category", fallback_category)
            if cat not in by_category:
                cat = fallback_category
            by_category[cat].append(item)

        # Process each category like _process_analysis does
        for cat, items in by_category.items():
            current_titles: set[str] = set()

            for item in items:
                title = item.get("title", "Unknown issue")
                current_titles.add(title)

                try:
                    severity = Severity(item.get("severity", "info"))
                except ValueError:
                    severity = Severity.INFO

                finding, is_new = await self._findings.add_or_update(
                    device=device_name,
                    severity=severity,
                    category=cat,
                    title=title,
                    detail=item.get("detail", ""),
                    recommendation=item.get("recommendation", ""),
                )

                if self._notify_callback and is_new:
                    await self._notify_callback(finding, is_new)

            resolved = await self._findings.resolve_missing(
                device_name, cat, current_titles,
            )
            for finding in resolved:
                if self._notify_callback:
                    await self._notify_callback(finding, False)

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

    def _build_system_prompt(self) -> str:
        """Build system prompt with injected memory context."""
        base = self._settings.llm.system_prompt or SYSTEM_PROMPT
        if not self._memory_store:
            return base
        device_names = self._device_manager.get_connected_devices()
        memory_ctx = self._memory_store.build_memory_context(device_names)
        if not memory_ctx:
            return base
        return base + "\n\n" + memory_ctx

    async def _compact_context(self, ctx: ConversationContext) -> None:
        """Flush important memories then summarize and compact the context."""
        logger.info("Compacting conversation context (%d messages)",
                     ctx.message_count)

        # 1. Memory flush — run on a disposable copy of the conversation
        if self._memory_store:
            try:
                flush_ctx = ConversationContext(max_messages=ctx.message_count + 10)
                for msg in ctx.messages:
                    if msg.role == Role.USER:
                        flush_ctx.add_user(msg.content)
                    elif msg.role == Role.ASSISTANT:
                        flush_ctx.add_assistant(msg)
                    elif msg.role == Role.TOOL and msg.tool_call_id:
                        flush_ctx.add_tool_result(msg.tool_call_id, msg.content)
                flush_ctx.add_user(MEMORY_FLUSH_PROMPT)
                await self._llm_tool_loop(flush_ctx, max_iterations=5)
            except Exception as exc:
                logger.warning("Memory flush failed: %s", exc)

        # 2. Summarize — single LLM call, no tools
        summary = ""
        try:
            summary_messages = list(ctx.messages)
            summary_messages.append(Message(
                role=Role.USER, content=SUMMARIZE_PROMPT,
            ))
            response = await self._llm.chat(
                messages=summary_messages,
                tools=None,
                system=self._settings.llm.system_prompt or SYSTEM_PROMPT,
                max_tokens=1024,
            )
            summary = response.content or ""
        except Exception as exc:
            logger.warning("Context summarization failed: %s", exc)
            summary = "(Context was compacted but summarization failed.)"

        # 3. Compact
        ctx.compact(summary, keep_recent=10)
        logger.info("Context compacted — summary length: %d chars", len(summary))

    async def _llm_tool_loop(self, ctx: ConversationContext,
                             max_iterations: int = 10) -> str:
        """Run the LLM tool-use loop until the LLM produces a final text response."""
        # Check if interactive context needs compaction
        if ctx is self._interactive_ctx and ctx.needs_compaction:
            await self._compact_context(ctx)

        system_prompt = self._build_system_prompt()
        for _ in range(max_iterations):
            response = await self._llm.chat(
                messages=ctx.messages,
                tools=AGENT_TOOLS,
                system=system_prompt,
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
                await self._run_check(
                    args["category"], args["device"],
                    _user_triggered=True,
                )
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

            elif name == "save_memory":
                if not self._memory_store:
                    return "Memory store not configured."
                return self._memory_store.save(
                    args["category"], args.get("key", ""), args["content"],
                )

            elif name == "read_memory":
                if not self._memory_store:
                    return "Memory store not configured."
                return self._memory_store.read(
                    args["category"], args.get("key"),
                )

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
