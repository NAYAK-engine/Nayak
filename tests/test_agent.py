"""
tests/test_agent.py — Production pytest test suite for the NAYAK engine.

Tests for Groq API integration (OpenAI-compatible).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nayak.agent import Agent, AgentConfig
from nayak.brain.groq import Action, ActionType, Brain
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


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------

class TestAction:
    def test_action_from_dict_navigate(self):
        data = {"type": "navigate", "params": {"url": "https://a.com"}, "reason": "r"}
        action = Action.from_dict(data)
        assert action.type == ActionType.NAVIGATE
        assert action.url == "https://a.com"

    def test_action_from_dict_click_selector(self):
        data = {"type": "click", "params": {"selector": "#b"}, "reason": "r"}
        action = Action.from_dict(data)
        assert action.type == ActionType.CLICK
        assert action.selector == "#b"

    def test_action_from_dict_click_coords(self):
        data = {"type": "click", "params": {"x": 100, "y": 200}, "reason": "r"}
        action = Action.from_dict(data)
        assert action.coordinates == (100, 200)

    def test_action_from_dict_type_text(self):
        data = {"type": "type_text", "params": {"selector": "input", "text": "hi"}, "reason": "r"}
        action = Action.from_dict(data)
        assert action.type == ActionType.TYPE_TEXT
        assert action.text == "hi"

    def test_all_action_types_parseable(self):
        for t in ActionType:
            data = {"type": t.value, "params": {}, "reason": "r"}
            action = Action.from_dict(data)
            assert action.type == t


# ---------------------------------------------------------------------------
# Brain tests (Mocked OpenAI)
# ---------------------------------------------------------------------------

class TestBrain:
    def _make_mock_response(self, content: str) -> MagicMock:
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        response.choices = [choice]
        return response

    @patch("nayak.brain.groq.OpenAI")
    def test_decide_returns_action(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '{"type": "navigate", "params": {"url": "https://x.com"}, "reason": "go"}'
        )

        brain = Brain(api_key="fake-key")
        action = brain.decide("goal", 1, "url", "title", "text", "mem")
        assert action.type == ActionType.NAVIGATE
        assert action.url == "https://x.com"

    @patch("nayak.brain.groq.OpenAI")
    def test_decide_retries_on_error(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        good_response = self._make_mock_response('{"type": "finish", "reason": "done"}')
        mock_client.chat.completions.create.side_effect = [
            Exception("429 Rate Limit"),
            good_response,
        ]

        brain = Brain(api_key="fake-key", retry_delay=0.0)
        with patch("time.sleep"):
            action = brain.decide("goal", 1, "url", "title", "text", "mem")
        assert action.type == ActionType.FINISH


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestAgent:
    def _make_agent(self, tmp_path: Path) -> Agent:
        config = AgentConfig(
            goal="test",
            agent_id="test",
            groq_api_key="fake",
            db_path=str(tmp_path / "test.db"),
        )
        return Agent.__new__(Agent) # Partial mock

    async def test_agent_config(self):
        config = AgentConfig(goal="g", groq_api_key="k")
        assert config.agent_id == "nayak-agent"
        assert len(config.session_id) == 36
