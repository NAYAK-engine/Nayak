"""
nayak/studio/server.py — NAYAK Studio FastAPI Web Server.

Provides a REST API consumed by the Studio web dashboard at
http://localhost:8000.  All endpoints are read/write proxies into the live
NAYAK runtime singletons (bus, registry, safety, platform, memory).

Endpoints::

    GET  /api/status          Runtime + module registry snapshot
    GET  /api/events          Last 50 events captured from the Event Bus
    GET  /api/memory          Last 20 memory entries from MemoryStore
    GET  /api/safety          Safety engine state
    GET  /api/skills          Loaded skills from the Developer Platform
    POST /api/safety/stop     Trigger emergency stop
    POST /api/safety/resume   Resume from emergency stop
    POST /api/run             Classify and fast-answer a goal
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── NAYAK runtime singletons ─────────────────────────────────────────────────
from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import registry
from nayak.core.runtime import runtime
from nayak.safety.engine import safety
from nayak.sdk.platform import platform
from nayak.cognition.fastpath import classify_query, fast_answer

# ─────────────────────────────────────────────────────────────────────────────
# Global event log (circular buffer, 50 entries)
# ─────────────────────────────────────────────────────────────────────────────

_EVENT_LOG: deque[dict[str, Any]] = deque(maxlen=50)
"""Circular buffer that stores the last 50 events emitted on the Event Bus."""

_STUDIO_START = time.time()


async def _capture_event(event: NayakEvent) -> None:
    """Append a serializable snapshot of *event* to the global event log."""
    _EVENT_LOG.append({
        "type": event.type.name,
        "source": event.source,
        "timestamp": event.timestamp,
        "payload": event.payload,
    })


# Subscribe to every EventType on import so no events are missed
for _et in EventType:
    bus.subscribe(_et, _capture_event)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NAYAK Studio",
    description="Real-time web dashboard for the NAYAK OS.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Body for POST /api/run."""
    goal: str


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Return the runtime status and full module registry snapshot.

    Returns:
        JSON containing nayak_version, runtime state, uptime, and every
        registered module with its layer, name, status, and version.
    """
    modules = [
        {
            "name": m.name,
            "layer": m.layer,
            "status": m.status.name,
            "version": m.version,
            "description": m.description,
        }
        for m in registry.list_all()
    ]
    return JSONResponse({
        "nayak_version": "0.2.0",
        "runtime_running": runtime.is_running,
        "uptime": round(runtime.uptime, 2),
        "modules": modules,
    })


@app.get("/api/events")
async def get_events() -> JSONResponse:
    """Return the last 50 events captured from the Event Bus.

    Returns:
        JSON list of event snapshots with type, source, timestamp, payload.
    """
    return JSONResponse(list(_EVENT_LOG))


@app.get("/api/memory")
async def get_memory() -> JSONResponse:
    """Return the last 20 memory entries from the default MemoryStore.

    Opens a short-lived connection to the SQLite store and closes it.

    Returns:
        JSON list of context-line strings, or an error if unavailable.
    """
    try:
        from nayak.memory.store import MemoryStore
        store = MemoryStore(agent_id="nayak-agent", session_id="__studio__")
        await store.init()
        lines = await store.get_recent(n=20)
        await store.close()
        return JSONResponse({"entries": lines})
    except Exception as exc:
        return JSONResponse({"entries": [], "error": str(exc)})


@app.get("/api/safety")
async def get_safety() -> JSONResponse:
    """Return the current Safety Engine state.

    Returns:
        JSON with is_stopped flag, violation count, and enabled capabilities.
    """
    try:
        caps = [c.value for c in safety.enabled_capabilities]
    except Exception:
        caps = []
    return JSONResponse({
        "is_stopped": safety.is_stopped,
        "violations": len(safety.violations),
        "enabled_capabilities": caps,
    })


@app.get("/api/skills")
async def get_skills() -> JSONResponse:
    """Return all skills currently loaded into the Developer Platform.

    Returns:
        JSON list of skill manifests (name, version, type, permissions).
    """
    skills = await platform.list_skills()
    return JSONResponse([
        {
            "name": s.name,
            "version": s.version,
            "skill_type": s.skill_type.name,
            "description": s.description,
            "author": s.author,
            "permissions": s.permissions,
            "skill_id": s.skill_id,
        }
        for s in skills
    ])


@app.post("/api/safety/stop")
async def safety_stop() -> JSONResponse:
    """Trigger the Safety Engine emergency stop.

    Immediately locks all agent actions. Requires a manual resume.

    Returns:
        JSON ``{"status": "stopped"}``.
    """
    await safety.emergency_stop()
    return JSONResponse({"status": "stopped"})


@app.post("/api/safety/resume")
async def safety_resume() -> JSONResponse:
    """Resume from a Safety Engine emergency stop.

    Re-enables agent action execution.

    Returns:
        JSON ``{"status": "resumed"}``.
    """
    await safety.resume()
    return JSONResponse({"status": "resumed"})


@app.post("/api/run")
async def run_goal(body: RunRequest) -> JSONResponse:
    """Classify a goal and attempt a fast-path answer.

    Uses :func:`~nayak.cognition.fastpath.classify_query` to determine
    complexity.  For INSTANT/SIMPLE queries, attempts a direct answer via
    the configured cognition backend.

    Args:
        body: Request body containing the ``goal`` string.

    Returns:
        JSON with ``complexity`` value and ``answer`` (or ``null``).
    """
    goal = body.goal.strip()
    if not goal:
        return JSONResponse({"error": "empty goal"}, status_code=400)

    complexity = await classify_query(goal)
    answer: str | None = None

    if complexity.name != "COMPLEX":
        try:
            import os
            provider = os.environ.get("NAYAK_PROVIDER", "ollama").lower()
            if provider == "gemini":
                from nayak.cognition.gemini import gemini_cognition as cog
            else:
                from nayak.cognition.ollama import ollama_cognition as cog
            answer = await fast_answer(goal, cog)
        except Exception:
            answer = None

    return JSONResponse({
        "complexity": complexity.name,
        "answer": answer,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

__all__ = ["app"]
