"""Application orchestrator — wires together all components."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from jace.agent.accumulator import AnomalyAccumulator
from jace.agent.anomaly import AnomalyDetector
from jace.agent.core import AgentCore
from jace.agent.findings import FindingsTracker
from jace.agent.heartbeat import HeartbeatManager
from jace.agent.memory import MemoryStore
from jace.agent.metrics_store import MetricsStore
from jace.agent.watch import WatchManager
from jace.checks.registry import build_default_registry
from jace.mcp.manager import MCPManager
from jace.config.settings import Settings, load_config
from jace.device.manager import DeviceManager
from jace.llm import create_llm_client
from jace.ui.tui import JACE

logger = logging.getLogger(__name__)


class BufferedLogHandler(logging.Handler):
    """In-memory log handler that stores recent entries for API access."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self._entries: deque[dict[str, str]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self._entries.append({
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        })

    def get_entries(self, lines: int = 50) -> list[dict[str, str]]:
        """Return the most recent *lines* log entries."""
        entries = list(self._entries)
        return entries[-lines:]


class Application:
    """Top-level application orchestrator."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.settings = load_config(config_path)
        ssh_config_path = Path(self.settings.ssh_config).expanduser()
        self.device_manager = DeviceManager(
            blocked_commands=self.settings.blocked_commands,
            allowed_commands=self.settings.allowed_commands,
            ssh_config=str(ssh_config_path) if ssh_config_path.is_file() else None,
        )
        self.llm = create_llm_client(self.settings.llm)
        self.check_registry = build_default_registry()
        self.findings_tracker = FindingsTracker(self.settings.storage_path)
        self.metrics_store = MetricsStore(self.settings.storage_path)
        self.anomaly_detector = AnomalyDetector(
            store=self.metrics_store,
            z_threshold=self.settings.metrics.anomaly_z_threshold,
            window_hours=self.settings.metrics.anomaly_window_hours,
            min_samples=self.settings.metrics.anomaly_min_samples,
        )

        # Heartbeat manager (optional)
        self.heartbeat_manager: HeartbeatManager | None = None
        if self.settings.heartbeat.enabled:
            hb_path = Path(self.settings.heartbeat.file)
            if not hb_path.is_absolute():
                # Resolve relative to config directory or cwd
                if config_path is not None:
                    hb_path = Path(config_path).parent / hb_path
            self.heartbeat_manager = HeartbeatManager(hb_path)

        # Memory store (optional)
        self.memory_store: MemoryStore | None = None
        if self.settings.memory.enabled:
            self.memory_store = MemoryStore(
                base_path=self.settings.storage_path,
                max_file_size=self.settings.memory.max_file_size,
                max_total_size=self.settings.memory.max_total_size,
            )
            self.memory_store.initialize()

        # Anomaly accumulator (optional — batches cross-category anomalies)
        self.anomaly_accumulator: AnomalyAccumulator | None = None
        if self.settings.correlation.enabled:
            self.anomaly_accumulator = AnomalyAccumulator(
                window_seconds=self.settings.correlation.window_seconds,
            )

        # MCP server manager (optional)
        self.mcp_manager: MCPManager | None = None
        if self.settings.mcp_servers:
            self.mcp_manager = MCPManager(self.settings.mcp_servers)

        # Watch manager — lightweight metric collection
        self.watch_manager = WatchManager(
            device_manager=self.device_manager,
            metrics_store=self.metrics_store,
        )

        self.agent = AgentCore(
            settings=self.settings,
            llm=self.llm,
            device_manager=self.device_manager,
            check_registry=self.check_registry,
            findings_tracker=self.findings_tracker,
            metrics_store=self.metrics_store,
            anomaly_detector=self.anomaly_detector,
            heartbeat_manager=self.heartbeat_manager,
            memory_store=self.memory_store,
            anomaly_accumulator=self.anomaly_accumulator,
            watch_manager=self.watch_manager,
            mcp_manager=self.mcp_manager,
        )
        self._api_server = None
        self._api_app = None
        self._log_handler: BufferedLogHandler | None = None

    async def start(self, *, api: bool = False) -> None:
        """Initialize and start the application."""
        self._setup_logging()

        # Initialize storage
        await self.findings_tracker.initialize()
        await self.metrics_store.initialize(
            retention_days=self.settings.metrics.retention_days,
        )

        # Add and connect devices
        for dev_config in self.settings.devices:
            self.device_manager.add_device(dev_config)

        # Start API server if requested
        if api or self.settings.api.enabled:
            await self._start_api()

        # Connect to MCP servers (if configured)
        if self.mcp_manager is not None:
            from jace.llm.tools import AGENT_TOOLS
            builtin_names = {t.name for t in AGENT_TOOLS}
            await self.mcp_manager.connect_all(builtin_names=builtin_names)

        # Run Textual TUI — device connections happen in the background
        tui = JACE(
            agent=self.agent,
            device_manager=self.device_manager,
            findings_tracker=self.findings_tracker,
        )

        # Expose TUI to API server for screenshot/tab endpoints
        if self._api_app is not None:
            self._api_app.state.tui = tui

        try:
            await tui.run_async()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        if self.mcp_manager is not None:
            await self.mcp_manager.close()
        self.watch_manager.stop_all()
        await self.agent.stop_monitoring()
        await self.device_manager.disconnect_all()
        await self.findings_tracker.close()
        await self.metrics_store.close()
        if self._api_server:
            self._api_server.should_exit = True
        logger.info("Shutdown complete.")

    async def _start_api(self) -> None:
        """Start the FastAPI server in the background."""
        from jace.api.server import create_api_app
        import uvicorn

        app = create_api_app(self.agent, self.device_manager, self.findings_tracker)
        self._api_app = app

        # Attach log handler so the /logs endpoint can access entries
        if self._log_handler is not None:
            app.state.log_handler = self._log_handler

        config = uvicorn.Config(
            app,
            host=self.settings.api.host,
            port=self.settings.api.port,
            log_level="warning",
        )
        self._api_server = uvicorn.Server(config)
        asyncio.create_task(self._api_server.serve())
        logger.info("API server started on %s:%d",
                     self.settings.api.host, self.settings.api.port)

    def _setup_logging(self) -> None:
        # Set level only — TUI installs its own handler in on_mount()
        logging.root.setLevel(logging.INFO)

        # Install buffered log handler for API /logs endpoint
        self._log_handler = BufferedLogHandler()
        logging.root.addHandler(self._log_handler)

        # Quiet noisy libraries — JACE wraps all calls in try/except and
        # provides its own error messages, so internal SSH tracebacks from
        # paramiko/ncclient/netmiko only add noise (especially before the TUI
        # installs its log handler, when they leak to stderr).
        logging.getLogger("paramiko").setLevel(logging.CRITICAL)
        logging.getLogger("ncclient").setLevel(logging.CRITICAL)
        logging.getLogger("netmiko").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
