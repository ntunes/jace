# JACE: Autonomous Control Engine

An autonomous AI agent that connects to Junos MX series network devices via SSH and NETCONF, proactively monitors health, audits configurations, and provides an interactive troubleshooting interface.

JACE works in the background — it decides what to check and when, surfaces issues proactively, and lets you ask natural language questions about your network.

## Features

- **Autonomous monitoring** — scheduled health checks across chassis, interfaces, routing, system resources, and configuration, analyzed by an LLM
- **Interactive REPL** — ask natural language questions about your devices and get answers backed by real command output
- **Dual transport** — PyEZ (NETCONF) as the primary driver with automatic Netmiko (SSH) fallback
- **Pluggable LLM backend** — supports Anthropic (Claude) and any OpenAI-compatible API (OpenAI, Ollama, vLLM, LiteLLM)
- **Findings tracking** — deduplicated findings with severity levels, persisted to SQLite, with automatic resolution detection
- **REST API** — optional FastAPI server with WebSocket support for real-time findings
- **Config auditing** — security and best-practice checks against device configurations

## Quick Start

```bash
# Clone and set up
git clone <repo-url> && cd jace
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your devices and API keys

# Run
python -m jace
# or
jace -c config.yaml
```

## Configuration

JACE is configured via a YAML file. Environment variables can be referenced with `${VAR_NAME}` syntax.

```yaml
llm:
  provider: anthropic          # or "openai"
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}
  # For OpenAI-compatible endpoints:
  # provider: openai
  # base_url: http://localhost:11434/v1
  # api_key: ${OPENAI_API_KEY}
  # model: gpt-4o

devices:
  - name: mx-core-01
    host: 10.0.0.1
    username: admin
    password: ${DEVICE_PASSWORD}
    ssh_key: ~/.ssh/id_rsa
    driver: auto               # auto | pyez | netmiko
    port: 830

schedule:
  chassis:    300              # seconds between checks
  interfaces: 120
  routing:    180
  system:     300
  config:     3600

api:
  enabled: false
  host: 127.0.0.1
  port: 8080
```

Config is loaded from `config.yaml` in the current directory, or specify a path with `-c`.

## Usage

### Interactive Shell

Once started, JACE opens a Rich-based REPL. Type natural language questions or use commands:

```
jace> what alarms are active on mx-core-01?
jace> show me the BGP peer status
jace> is there anything wrong with the interfaces?
```

Shell commands:

| Command | Description |
|---|---|
| `/devices` | List managed devices and connection status |
| `/findings` | Show active findings summary |
| `/check <device> <category>` | Run a health check now |
| `/clear` | Clear the screen |
| `/help` | Show help |
| `/quit` | Exit |

Background findings are displayed as alerts above the prompt as they are discovered.

### API Server

Start with the `--api` flag or set `api.enabled: true` in config:

```bash
jace --api
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Agent health status |
| `GET` | `/devices` | List managed devices |
| `GET` | `/findings` | Current findings (filterable) |
| `POST` | `/chat` | Send a message, get a response |
| `WS` | `/ws` | Real-time findings stream |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   JACE Agent Core                │
│                                                  │
│  ┌──────────────┐         ┌───────────────────┐  │
│  │  Scheduler    │────────▶│  Health Checks    │  │
│  │  (background) │         │  chassis/intf/    │  │
│  └──────────────┘         │  routing/system/  │  │
│         │                  │  config           │  │
│         │                  └────────┬──────────┘  │
│         ▼                           ▼             │
│  ┌──────────────┐         ┌───────────────────┐  │
│  │  LLM Client   │◀───────│  Device Manager   │  │
│  │  (Anthropic/  │         │  (PyEZ + Netmiko) │  │
│  │   OpenAI)     │         └───────────────────┘  │
│  └──────┬───────┘                                │
│         │                                         │
│         ▼                                         │
│  ┌──────────────┐         ┌───────────────────┐  │
│  │  Findings     │────────▶│  Notifications    │  │
│  │  Tracker      │         │  (REPL + API)     │  │
│  │  (SQLite)     │         └───────────────────┘  │
│  └──────────────┘                                │
└──────────────────────────────────────────────────┘
```

### Key Components

| Package | Purpose |
|---|---|
| `jace/llm/` | Pluggable LLM abstraction with tool-use loop |
| `jace/device/` | Device connectivity — PyEZ/Netmiko drivers, connection pool |
| `jace/agent/` | Autonomous agent loop, scheduler, findings tracker |
| `jace/checks/` | Health check definitions organized by category |
| `jace/ui/` | Rich-based interactive REPL and notification rendering |
| `jace/api/` | FastAPI REST server with WebSocket |
| `jace/config/` | YAML config loader with env var expansion |

### LLM Tools

The agent has access to these tools during conversations and health check analysis:

- `run_command` — execute any Junos operational command
- `get_config` — retrieve device configuration (full or filtered, text/set/xml)
- `get_device_facts` — device model, version, serial, uptime
- `list_devices` — all managed devices and their status
- `get_findings` — current and historical findings with filtering
- `run_health_check` — trigger a health check category on demand
- `compare_config` — diff current config against a rollback

### Health Check Categories

| Category | Checks | Default Interval |
|---|---|---|
| `chassis` | Alarms, environment, FPC status, PFE exceptions | 5 min |
| `interfaces` | Link status, error counters | 2 min |
| `routing` | BGP peers, OSPF neighbors, route table summary | 3 min |
| `system` | CPU/memory, storage, top processes | 5 min |
| `config` | Security audit, best-practice audit | 1 hour |

## Development

```bash
# Set up dev environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

## License

MIT
