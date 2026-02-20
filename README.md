# JACE: Autonomous Control Engine

An autonomous AI agent that connects to Junos MX series network devices via SSH and NETCONF, proactively monitors health, audits configurations, and provides an interactive troubleshooting interface.

JACE works in the background — it decides what to check and when, surfaces issues proactively, and lets you ask natural language questions about your network.

## Features

- **Autonomous monitoring** — scheduled health checks across chassis, interfaces, routing, system resources, and configuration, analyzed by an LLM
- **Anomaly detection** — Z-score statistical anomaly detection on time-series metrics with configurable thresholds
- **Anomaly correlation** — simultaneous anomalies on the same device are batched over a configurable time window and investigated holistically, so the LLM can identify common root causes instead of treating each category in isolation
- **Persistent memory** — the agent remembers device baselines, operator preferences, and incident history across sessions, and consults memory during investigations
- **Heartbeat monitoring** — user-programmable periodic checks defined in natural language (e.g. "verify BGP peers are up on all devices")
- **Textual TUI** — interactive terminal interface with live sidebar showing device status and findings, real-time log footer, and chat-style interaction
- **Metric watches** — lightweight background metric collection via regex extraction from device commands, with zero LLM cost per sample
- **SSH config support** — honors `~/.ssh/config` by default (proxy jumps, identity files, custom ports), with per-device override
- **Dual transport** — PyEZ (NETCONF) as the primary driver with automatic Netmiko (SSH) fallback
- **Pluggable LLM backend** — supports Anthropic (Claude) and any OpenAI-compatible API (OpenAI, Ollama, vLLM, LiteLLM)
- **Findings tracking** — deduplicated findings with severity levels, persisted to SQLite, with automatic resolution detection
- **Context compaction** — long conversations are automatically summarized and compacted to stay within context limits, with important observations flushed to persistent memory first
- **LLM logging** — optional audit trail of all LLM requests and responses in text or JSONL format
- **REST API** — optional FastAPI server with WebSocket support for real-time findings
- **Config auditing** — security and best-practice checks against device configurations

## Quick Start

```bash
# Clone and set up
git clone <repo-url> && cd jace
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
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
  # log_file: ~/.jace/llm.log  # enable LLM request/response logging
  # log_format: text           # "text" (human-readable) or "jsonl"
  # For OpenAI-compatible endpoints:
  # provider: openai
  # base_url: http://localhost:11434/v1
  # api_key: ${OPENAI_API_KEY}
  # model: gpt-4o

ssh_config: ~/.ssh/config          # global SSH config (default: ~/.ssh/config)

devices:
  - name: mx-core-01
    host: 10.0.0.1
    username: admin
    password: ${DEVICE_PASSWORD}
    ssh_key: ~/.ssh/id_rsa
    driver: auto               # auto | pyez | netmiko
    port: 830
    # ssh_config: ~/.ssh/config_lab  # per-device override

schedule:
  chassis:    300              # seconds between checks
  interfaces: 120
  routing:    180
  system:     300
  config:     86400

metrics:
  retention_days: 30
  anomaly_z_threshold: 3.0     # standard deviations to trigger anomaly
  anomaly_window_hours: 24     # lookback window for baseline
  anomaly_min_samples: 10      # minimum data points before detection activates

heartbeat:
  enabled: false
  interval: 1800               # seconds (default: 30 min)
  file: heartbeat.md           # path to instructions file

memory:
  enabled: true
  max_file_size: 8000          # max chars per memory file
  max_total_size: 24000        # max chars injected into system prompt

correlation:
  enabled: true
  window_seconds: 30.0         # batch anomalies per device over this window

storage:
  path: ~/.jace/               # findings DB, metrics, memory, logs

api:
  enabled: false
  host: 127.0.0.1
  port: 8080
```

Config is loaded from `config.yaml` in the current directory, or specify a path with `-c`.

## Usage

### Interactive TUI

Once started, JACE opens a Textual-based terminal interface with:

- **Chat panel** — type natural language questions and receive answers backed by real command output
- **Device sidebar** — live device status with connection indicators (refreshes every 3s)
- **Findings sidebar** — active findings summary grouped by severity
- **Log footer** — real-time application logs

```
user> what alarms are active on mx-core-01?
jace> No alarms currently active on mx-core-01.

user> show me the BGP peer status
user> is there anything wrong with the interfaces?
```

Commands:

| Command | Description |
|---|---|
| `/devices` | List managed devices and connection status |
| `/findings` | Show active findings summary |
| `/check <device> <category>` | Run a health check now |
| `/clear` | Clear the chat panel |
| `/help` | Show help |
| `/quit` | Exit |

Keyboard shortcuts: `Ctrl+C` quit, `Ctrl+L` clear chat, `F1` help.

Background findings appear as alert panels in the chat as they are discovered.

### Heartbeat Monitoring

Enable `heartbeat` in config and write natural language instructions in the heartbeat file:

```markdown
- Check that all BGP peers are established
- Verify no critical chassis alarms
- Confirm interface error counters are not rising
```

The agent evaluates these periodically, preferring already-collected metrics and findings before running new commands. Instructions can also be managed at runtime through the `manage_heartbeat` tool.

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
┌─────────────────────────────────────────────────────────┐
│                     JACE Agent Core                     │
│                                                         │
│  ┌───────────────┐        ┌───────────────────────┐     │
│  │  Scheduler     │───────▶│  Health Checks        │     │
│  │  (background)  │        │  chassis/interfaces/  │     │
│  └───────────────┘        │  routing/system/config │     │
│         │                  └───────────┬───────────┘     │
│         │                              │                 │
│         │                  ┌───────────▼───────────┐     │
│         │                  │  Metrics Extractors    │     │
│         │                  │  + Anomaly Detector    │     │
│         │                  └───────────┬───────────┘     │
│         │                              │                 │
│         │                  ┌───────────▼───────────┐     │
│         │                  │  Anomaly Accumulator   │     │
│         │                  │  (temporal batching)   │     │
│         │                  └───────────┬───────────┘     │
│         ▼                              ▼                 │
│  ┌───────────────┐        ┌───────────────────────┐     │
│  │  LLM Client    │◀──────│  Device Manager       │     │
│  │  (Anthropic/   │        │  (PyEZ + Netmiko)     │     │
│  │   OpenAI)      │        └───────────────────────┘     │
│  └───────┬───────┘                                      │
│          │                                               │
│          ▼                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Findings     │  │ Memory Store │  │  Metrics     │   │
│  │  Tracker      │  │ (device/user/│  │  Store       │   │
│  │  (SQLite)     │  │  incident)   │  │  (SQLite)    │   │
│  └──────┬───────┘  └──────────────┘  └──────────────┘   │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │ Notifications │                                        │
│  │ (TUI + API)   │                                        │
│  └──────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Package | Purpose |
|---|---|
| `jace/agent/` | Autonomous agent loop, scheduler, findings tracker, anomaly detection, memory store, anomaly accumulator |
| `jace/llm/` | Pluggable LLM abstraction with tool-use loop and optional request logging |
| `jace/device/` | Device connectivity — PyEZ/Netmiko drivers, connection pool |
| `jace/checks/` | Health check definitions organized by category |
| `jace/metrics/` | Metric extractors that parse command output into time-series data |
| `jace/ui/` | Textual-based TUI with sidebar, chat panel, and log footer |
| `jace/api/` | FastAPI REST server with WebSocket |
| `jace/config/` | YAML config loader with env var expansion |

### LLM Tools

The agent has access to these tools during conversations and health check analysis:

| Tool | Description |
|---|---|
| `run_command` | Execute any Junos operational command |
| `get_config` | Retrieve device configuration (full or filtered, text/set/xml) |
| `get_device_facts` | Device model, version, serial, uptime |
| `list_devices` | All managed devices and their status |
| `get_findings` | Current and historical findings with filtering |
| `run_health_check` | Trigger a health check category on demand |
| `get_metrics` | Query time-series metrics and historical trends |
| `compare_config` | Diff current config against a rollback |
| `manage_heartbeat` | List, add, remove, or replace heartbeat instructions |
| `manage_watches` | Add, remove, or list lightweight background metric watches |
| `save_memory` | Persist observations to long-term store (device/user/incident) |
| `read_memory` | Recall saved memories or list available entries |

### Health Check Categories

| Category | Checks | Default Interval |
|---|---|---|
| `chassis` | Alarms, environment, FPC status, PFE exceptions | 5 min |
| `interfaces` | Link status, error counters | 2 min |
| `routing` | BGP peers, OSPF neighbors, route table summary | 3 min |
| `system` | CPU/memory, storage, top processes | 5 min |
| `config` | Security audit, best-practice audit | 24 hours |

### Anomaly Detection Pipeline

Each health check extracts structured metrics (CPU temperatures, error counter deltas, peer counts, etc.) and stores them as time-series data. The anomaly detector computes Z-scores against a sliding window baseline — when a metric deviates beyond the configured threshold, JACE triggers an LLM investigation with the anomaly context and raw data.

When anomalies fire across multiple categories on the same device within the correlation window (default 30s), they are batched and investigated together. The LLM receives all anomaly data in a single prompt, along with related active findings from across the fleet, enabling it to identify common root causes.

### Persistent Memory

JACE maintains a persistent knowledge base across sessions, organized into three categories:

- **Device** — per-device profiles, baselines, and learned quirks
- **User** — operator preferences (alert thresholds, output formats, workflow habits)
- **Incident** — past incident records with root causes and resolutions

Memory is injected into the system prompt (within a configurable budget) and the agent is instructed to consult and update it during investigations.

## Development

```bash
# Set up dev environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

## License

MIT
