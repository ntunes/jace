# JACE: Autonomous Control Engine

## Environment
- Always use the project virtual environment (`.venv/`) — never install packages globally
- Activate with: `source .venv/bin/activate`
- Python 3.9+ compatible (uses `from __future__ import annotations` and `eval_type_backport`)

## Project Structure
- `jace/` — main package
- `tests/` — pytest test suite
- Run tests: `source .venv/bin/activate && python -m pytest tests/ -v`

## Key Conventions
- All modules use `from __future__ import annotations` for PEP 604 union syntax compatibility
- Pydantic models require `eval_type_backport` on Python <3.10
- Async-first architecture with `asyncio`; blocking PyEZ/Netmiko calls run in thread pool executors
