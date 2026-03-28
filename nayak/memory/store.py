"""
memory/store.py — Persistent episodic memory backed by SQLite (via aiosqlite).

Database file: nayak_memory.db (in ~/.nayak/ by default).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH = Path.home() / ".nayak" / "nayak_memory.db"

_DDL = """\
CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    ts          TEXT    NOT NULL,
    goal        TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    result      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id);
"""


@dataclass
class MemoryEntry:
    agent_id: str
    session_id: str
    step: int
    ts: str
    goal: str
    action: str
    result: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "MemoryEntry":
        return cls(
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            step=row["step"],
            ts=row["ts"],
            goal=row["goal"],
            action=row["action"],
            result=row["result"],
        )

    def to_context_line(self) -> str:
        """One-line summary for the brain prompt."""
        return f"[Step {self.step} | {self.ts[:19]}] {self.action} → {self.result}"


class MemoryStore:
    """
    Async SQLite memory store.

    Usage::

        store = MemoryStore(agent_id="my-robot", session_id="run-001")
        await store.init()
        await store.save(step=1, action="[NAVIGATE] url='...'", result="HTTP 200", goal="...")
        lines = await store.get_recent(n=10)
        await store.close()
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        db_path: Path | str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Open database connection and ensure schema exists."""
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        logger.debug("MemoryStore ready at %s", self._db_path)

    async def save(
        self,
        step: int,
        action: str,
        result: str,
        goal: str,
    ) -> None:
        """Persist one step's action+result to the database."""
        if self._conn is None:
            raise RuntimeError("MemoryStore.init() must be called first")
        ts = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            """
            INSERT INTO memory (agent_id, session_id, step, ts, goal, action, result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (self.agent_id, self.session_id, step, ts, goal, action, result),
        )
        await self._conn.commit()

    async def get_recent(self, n: int = 10) -> list[str]:
        """Return the last *n* memory entries as context-line strings."""
        if self._conn is None:
            raise RuntimeError("MemoryStore.init() must be called first")
        async with self._conn.execute(
            """
            SELECT * FROM memory
            WHERE agent_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.agent_id, n),
        ) as cursor:
            rows = await cursor.fetchall()
        entries = [MemoryEntry.from_row(r) for r in reversed(rows)]
        return [e.to_context_line() for e in entries]

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for every session of this agent."""
        if self._conn is None:
            raise RuntimeError("MemoryStore.init() must be called first")
        async with self._conn.execute(
            """
            SELECT session_id,
                   MIN(ts)  AS started_at,
                   MAX(ts)  AS last_ts,
                   COUNT(*) AS steps,
                   goal
            FROM memory
            WHERE agent_id = ?
            GROUP BY session_id
            ORDER BY started_at DESC
            """,
            (self.agent_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
