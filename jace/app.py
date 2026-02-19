"""Application orchestrator — wires together all components."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from jace.agent.anomaly import AnomalyDetector
from jace.agent.core import AgentCore
from jace.agent.findings import FindingsTracker
from jace.agent.metrics_store import MetricsStore
from jace.checks.registry import build_default_registry
from jace.config.settings import Settings, load_config
from jace.device.manager import DeviceManager
from jace.llm import create_llm_client
from jace.ui.shell import InteractiveShell

logger = logging.getLogger(__name__)


class Application:
    """Top-level application orchestrator."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.settings = load_config(config_path)
        self.device_manager = DeviceManager(
            blocked_commands=self.settings.blocked_commands,
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
        self.agent = AgentCore(
            settings=self.settings,
            llm=self.llm,
            device_manager=self.device_manager,
            check_registry=self.check_registry,
            findings_tracker=self.findings_tracker,
            metrics_store=self.metrics_store,
            anomaly_detector=self.anomaly_detector,
        )
        self._api_server = None

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

        logger.info("Connecting to %d device(s)...", len(self.settings.devices))
        await self.device_manager.connect_all()

        connected = self.device_manager.get_connected_devices()
        logger.info("Connected to %d device(s): %s", len(connected), connected)

        # Start background monitoring
        self.agent.start_monitoring()

        # Start API server if requested
        if api or self.settings.api.enabled:
            await self._start_api()

        # Run interactive shell
        shell = InteractiveShell(self.agent)
        try:
            await shell.run()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
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
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # Quiet noisy libraries
        logging.getLogger("paramiko").setLevel(logging.WARNING)
        logging.getLogger("ncclient").setLevel(logging.WARNING)
        logging.getLogger("netmiko").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
