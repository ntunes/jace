"""FastAPI server for REST API and WebSocket integrations."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from jace.agent.core import AgentCore
from jace.agent.findings import Finding, FindingsTracker, Severity
from jace.device.manager import DeviceManager


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class TabRequest(BaseModel):
    tab: str


TAB_MAP = {
    "chat": "tab-chat",
    "findings": "tab-findings",
    "logs": "tab-logs",
}


def create_api_app(agent: AgentCore, device_manager: DeviceManager,
                   findings_tracker: FindingsTracker) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="JACE API",
        description="REST API for JACE: Autonomous Control Engine",
        version="0.1.0",
    )

    # WebSocket connections for real-time findings
    ws_clients: list[WebSocket] = []

    # Wire up finding notifications to WebSocket broadcast
    original_callback = agent._notify_callback

    async def _broadcast_finding(finding: Finding, is_new: bool) -> None:
        if original_callback:
            await original_callback(finding, is_new)
        data = json.dumps({
            "type": "finding",
            "is_new": is_new,
            "finding": finding.to_dict(),
        })
        for ws in list(ws_clients):
            try:
                await ws.send_text(data)
            except Exception:
                ws_clients.remove(ws)

    agent.set_notify_callback(_broadcast_finding)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        connected = device_manager.get_connected_devices()
        return {
            "status": "ok",
            "devices_connected": len(connected),
            "active_findings": findings_tracker.active_count,
            "critical_findings": findings_tracker.critical_count,
        }

    @app.get("/devices")
    async def list_devices(category: str | None = None) -> list[dict[str, Any]]:
        devices = device_manager.list_devices(category=category)
        return [
            {
                "name": d.name,
                "host": d.host,
                "category": d.category,
                "status": d.status.value,
                "model": d.model,
                "version": d.version,
                "serial": d.serial,
                "uptime": d.uptime,
                "last_check": d.last_check.isoformat() if d.last_check else None,
            }
            for d in devices
        ]

    @app.get("/findings")
    async def get_findings(
        device: str | None = None,
        severity: str | None = None,
        category: str | None = None,
        device_category: str | None = None,
        include_resolved: bool = False,
    ) -> list[dict[str, Any]]:
        if include_resolved:
            findings = await findings_tracker.get_history(
                device=device, include_resolved=True,
            )
        else:
            sev = Severity(severity) if severity else None
            findings = findings_tracker.get_active(
                device=device, severity=sev, category=category,
            )
        result = [f.to_dict() for f in findings]
        if device_category:
            cat_devices = {
                d.name
                for d in device_manager.list_devices(category=device_category)
            }
            result = [f for f in result if f.get("device") in cat_devices]
        return result

    @app.get("/inventory")
    async def get_inventory() -> dict[str, Any]:
        categories = device_manager.get_categories()
        result: dict[str, Any] = {}
        for cat in categories:
            cat_devices = device_manager.list_devices(category=cat)
            result[cat] = {
                "device_count": len(cat_devices),
                "devices": [d.name for d in cat_devices],
            }
        uncategorized = device_manager.list_devices(category="")
        if uncategorized:
            result["_uncategorized"] = {
                "device_count": len(uncategorized),
                "devices": [d.name for d in uncategorized],
            }
        return result

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        response = await agent.handle_user_input(request.message)
        return ChatResponse(response=response)

    @app.get("/screenshot")
    async def screenshot() -> dict[str, str]:
        tui = getattr(app.state, "tui", None)
        if tui is None:
            raise HTTPException(status_code=503, detail="TUI not yet available")
        svg = tui.export_screenshot()
        return {"svg": svg}

    @app.get("/logs")
    async def get_logs(lines: int = 50) -> list[dict[str, str]]:
        handler = getattr(app.state, "log_handler", None)
        if handler is None:
            return []
        return handler.get_entries(lines)

    @app.post("/tabs")
    async def switch_tab(request: TabRequest) -> dict[str, str]:
        tab_id = TAB_MAP.get(request.tab)
        if tab_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tab '{request.tab}'. "
                       f"Valid: {', '.join(TAB_MAP)}",
            )
        tui = getattr(app.state, "tui", None)
        if tui is None:
            raise HTTPException(status_code=503, detail="TUI not yet available")

        from textual.widgets import TabbedContent

        def _switch() -> None:
            tabs = tui.query_one("#tabs", TabbedContent)
            tabs.active = tab_id

        tui.call_later(_switch)
        return {"tab": request.tab}

    @app.get("/chat/history")
    async def chat_history(limit: int = 50) -> list[dict[str, str]]:
        return agent.get_chat_history(limit)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        ws_clients.append(websocket)
        try:
            while True:
                # Keep connection alive, handle incoming messages
                data = await websocket.receive_text()
                # Process as chat message
                response = await agent.handle_user_input(data)
                await websocket.send_text(json.dumps({
                    "type": "chat_response",
                    "response": response,
                }))
        except WebSocketDisconnect:
            ws_clients.remove(websocket)

    return app
