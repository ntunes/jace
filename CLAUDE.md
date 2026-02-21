# JACE: Autonomous Control Engine

## Environment
- Always use the project virtual environment (`.venv/`) — never install packages globally
- Activate with: `source .venv/bin/activate`
- Python 3.10+ required

## Project Structure
- `jace/` — main package
- `tests/` — pytest test suite
- Run tests: `source .venv/bin/activate && python -m pytest tests/ -v`

## Key Conventions
- All modules use `from __future__ import annotations` for PEP 604 union syntax compatibility
- Async-first architecture with `asyncio`; blocking PyEZ/Netmiko calls run in thread pool executors
