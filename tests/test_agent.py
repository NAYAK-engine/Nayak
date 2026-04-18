"""
tests/test_agent.py — Production pytest test suite for the NAYAK engine.

Tests cover: MemoryStore, Action/ActionType parsing, Agent config.
Brain provider tests use mocks so no real API key is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nayak.agent import Agent, AgentConfig
from nayak.brain import Action, ActionType
from nayak.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# MemoryStore tests
# ---------------------------------------------------------------------------

class TestMemoryStore:
    @pytest.fixture()
    async def store(self, tmp_path: Path):
        s = MemoryStore(
            agent_id="test-agent",
            session_id="test-session-001",
            db_path=tmp_path / "nayak_memory.db",
        )
        await s.init()
        yield s
        await s.close()

    async def test_init_creates_database(self, tmp_path: Path):
        db = tmp_path / "nayak_memory.db"
        s = MemoryStore("a", "s", db_path=db)
        await s.init()
        assert db.exists()
        await s.close()

    async def test_save_and_get_recent(self, store: MemoryStore):
        await store.save(
            step=1,
            action="[NAVIGATE] url='https://example.com'",
            result="Navigated OK",
            goal="test",
        )
        entries = await store.get_recent(n=5)
        assert len(entries) == 1
        assert "NAVIGATE" in entries[0]
        assert "Navigated OK" in entries[0]

    async def test_get_visited_urls(self, store: MemoryStore):
        await store.save(
            step=1,
            action="[NAVIGATE] url='https://example.com'",
            result="Navigated to https://example.com — HTTP 200",
            goal="test",
        )
        urls = await store.get_visited_urls()
        assert len(urls) == 1
        assert "example.com" in urls[0]


# ---------------------------------------------------------------------------
# Action / ActionType tests
# ---------------------------------------------------------------------------

class TestAction:
    def test_action_from_dict_navigate(self):
        data = {"type": "navigate", "params": {"url": "https://a.com"}}
        action = Action.from_dict(data)
        assert action.type == ActionType.NAVIGATE
        assert action.url == "https://a.com"

    def test_action_from_dict_click_selector(self):
        data = {"type": "click", "params": {"selector": "#b"}}
        action = Action.from_dict(data)
        assert action.type == ActionType.CLICK
        assert action.selector == "#b"

    def test_action_from_dict_click_coords(self):
        data = {"type": "click", "params": {"x": 100, "y": 200}}
        action = Action.from_dict(data)
        assert action.coordinates == (100, 200)

    def test_action_from_dict_type_text(self):
        data = {"type": "type_text", "params": {"selector": "input", "text": "hi"}}
        action = Action.from_dict(data)
        assert action.type == ActionType.TYPE_TEXT
        assert action.text == "hi"

    def test_all_action_types_parseable(self):
        for t in ActionType:
            data = {"type": t.value, "params": {}}
            action = Action.from_dict(data)
            assert action.type == t

    def test_unknown_type_returns_finish(self):
        data = {"type": "teleport", "params": {}}
        action = Action.from_dict(data)
        assert action.type == ActionType.FINISH

    def test_non_dict_returns_finish(self):
        action = Action.from_dict("not a dict")
        assert action.type == ActionType.FINISH


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------

class TestAgentConfig:
    def test_defaults_filled(self):
        config = AgentConfig(goal="test goal")
        assert config.agent_id == "nayak-agent"
        assert len(config.session_id) == 36   # UUID4

    def test_custom_values(self):
        config = AgentConfig(
            goal="g",
            agent_id="my-bot",
            max_steps=50,
            headless=False,
        )
        assert config.agent_id == "my-bot"
        assert config.max_steps == 50
        assert config.headless is False
