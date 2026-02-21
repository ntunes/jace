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
    port: int = 22
    platform: str = "junos"
    ssh_config: str | None = None
    timeout: int = 30
    category: str = ""


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


class CredentialConfig(BaseModel):
    username: str | None = None
    password: str | None = None
    ssh_key: str | None = None


class InventoryDeviceConfig(BaseModel):
    name: str
    host: str
    username: str | None = None
    password: str | None = None
    ssh_key: str | None = None
    credentials: str | None = None  # optional per-device credential ref
    driver: str = "auto"
    port: int = 22
    platform: str = "junos"
    ssh_config: str | None = None
    timeout: int = 30


class InventoryCategoryConfig(BaseModel):
    credentials: str | None = None
    schedule: ScheduleConfig | None = None
    devices: list[InventoryDeviceConfig] = Field(default_factory=list)


class InventoryConfig(BaseModel):
    credentials: dict[str, CredentialConfig] = Field(default_factory=dict)
    categories: dict[str, InventoryCategoryConfig] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    name: str
    transport: str = "stdio"                          # stdio | sse | streamable-http
    command: str | None = None                        # stdio only
    args: list[str] = Field(default_factory=list)     # stdio only
    url: str | None = None                            # sse / streamable-http
    headers: dict[str, str] | None = None             # sse / streamable-http
    env: dict[str, str] | None = None                 # stdio only


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
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    blocked_commands: list[str] = Field(default_factory=list)
    allowed_commands: list[str] = Field(default_factory=list)
    ssh_config: str = "~/.ssh/config"
    device_schedules: dict[str, ScheduleConfig] = Field(
        default_factory=dict, exclude=True,
    )

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

        inventory_path = raw.pop("inventory", None)

        has_devices = bool(raw.get("devices"))
        if has_devices and inventory_path:
            raise ValueError(
                "Config defines both 'devices' and 'inventory'. "
                "Use one or the other, not both."
            )

        settings = Settings.model_validate(raw)

        if inventory_path:
            config_dir = path.parent
            _load_inventory(settings, inventory_path, config_dir)

        return settings

    return Settings()


def _resolve_credentials(
    all_creds: dict[str, CredentialConfig],
    name: str,
    context: str,
) -> CredentialConfig:
    """Look up a named credential set, raising ValueError if missing."""
    if name not in all_creds:
        raise ValueError(
            f"Unknown credential reference '{name}' in {context}"
        )
    return all_creds[name]


def _merge_device_credentials(
    cat_creds: CredentialConfig | None,
    all_creds: dict[str, CredentialConfig],
    dev: InventoryDeviceConfig,
    context: str,
) -> dict[str, str | None]:
    """Merge credentials: DeviceConfig defaults → category creds → device
    creds ref → device explicit fields.  Returns dict of credential fields."""
    # Start with DeviceConfig defaults
    merged: dict[str, str | None] = {
        "username": "admin",
        "password": None,
        "ssh_key": None,
    }

    # Layer 2: category credentials
    if cat_creds:
        for field in ("username", "password", "ssh_key"):
            val = getattr(cat_creds, field)
            if val is not None:
                merged[field] = val

    # Layer 3: per-device credential reference
    if dev.credentials:
        dev_creds = _resolve_credentials(all_creds, dev.credentials, context)
        for field in ("username", "password", "ssh_key"):
            val = getattr(dev_creds, field)
            if val is not None:
                merged[field] = val

    # Layer 4: explicit device-level fields
    for field in ("username", "password", "ssh_key"):
        val = getattr(dev, field)
        if val is not None:
            merged[field] = val

    return merged


def _load_inventory(
    settings: Settings,
    inventory_path: str,
    config_dir: Path,
) -> None:
    """Load an inventory file and flatten it into settings.devices."""
    full_path = (config_dir / inventory_path).resolve()
    if not full_path.is_file():
        raise ValueError(f"Inventory file not found: {full_path}")

    with open(full_path) as f:
        raw = yaml.safe_load(f) or {}
    raw = _walk_and_expand(raw)

    inventory = InventoryConfig.model_validate(raw)

    seen_names: set[str] = set()
    devices: list[DeviceConfig] = []

    for cat_name, cat in inventory.categories.items():
        # Resolve category-level credentials
        cat_creds: CredentialConfig | None = None
        if cat.credentials:
            cat_creds = _resolve_credentials(
                inventory.credentials,
                cat.credentials,
                f"category '{cat_name}'",
            )

        # Build per-device schedule mapping
        if cat.schedule:
            for dev in cat.devices:
                settings.device_schedules[dev.name] = cat.schedule

        for dev in cat.devices:
            if dev.name in seen_names:
                raise ValueError(
                    f"Duplicate device name '{dev.name}' in inventory"
                )
            seen_names.add(dev.name)

            creds = _merge_device_credentials(
                cat_creds, inventory.credentials, dev,
                f"device '{dev.name}' in category '{cat_name}'",
            )

            devices.append(DeviceConfig(
                name=dev.name,
                host=dev.host,
                username=creds["username"] or "admin",
                password=creds["password"],
                ssh_key=creds["ssh_key"],
                driver=dev.driver,
                port=dev.port,
                platform=dev.platform,
                ssh_config=dev.ssh_config,
                timeout=dev.timeout,
                category=cat_name,
            ))

    settings.devices = devices
