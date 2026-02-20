"""YAML config loader with environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _expand_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    pattern = re.compile(r"\$\{([^}]+)\}")
    def replacer(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))
    return pattern.sub(replacer, value)


def _walk_and_expand(obj: object) -> object:
    """Recursively expand environment variables in strings."""
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_expand(item) for item in obj]
    return obj


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    system_prompt: str | None = None
    log_file: str | None = None
    log_format: str = "text"


class DeviceConfig(BaseModel):
    name: str
    host: str
    username: str = "admin"
    password: str | None = None
    ssh_key: str | None = None
    driver: str = "auto"
    port: int = 830
    platform: str = "junos"
    ssh_config: str | None = None


class ScheduleConfig(BaseModel):
    chassis: int = 300
    interfaces: int = 120
    routing: int = 180
    system: int = 300
    config: int = 86400


class MetricsConfig(BaseModel):
    retention_days: int = 30
    anomaly_z_threshold: float = 3.0
    anomaly_window_hours: int = 24
    anomaly_min_samples: int = 10


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_file_size: int = 8000      # max chars per memory file
    max_total_size: int = 24000    # max chars injected into system prompt


class HeartbeatConfig(BaseModel):
    enabled: bool = False
    interval: int = 1800  # seconds (default 30 min)
    file: str = "heartbeat.md"


class CorrelationConfig(BaseModel):
    enabled: bool = True
    window_seconds: float = 30.0


class APIConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080


class StorageConfig(BaseModel):
    path: str = "~/.jace/"


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    devices: list[DeviceConfig] = Field(default_factory=list)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    blocked_commands: list[str] = Field(default_factory=list)
    allowed_commands: list[str] = Field(default_factory=list)
    ssh_config: str = "~/.ssh/config"

    @property
    def storage_path(self) -> Path:
        return Path(self.storage.path).expanduser()


def load_config(path: str | Path | None = None) -> Settings:
    """Load configuration from a YAML file, falling back to defaults."""
    if path is None:
        candidates = [
            Path("config.yaml"),
            Path("config.yml"),
            Path.home() / ".jace" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if path is not None:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _walk_and_expand(raw)
        return Settings.model_validate(raw)

    return Settings()
