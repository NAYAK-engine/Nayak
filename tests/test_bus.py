"""
tests/test_bus.py — Unit tests for the NAYAK Event Bus.

Validates subscribe, emit, unsubscribe, handler isolation, and error
emission on a fresh EventBus instance per test.
"""

import pytest
import asyncio

from nayak.core.bus import EventBus, EventType, NayakEvent


class TestEventBus:
    """Tests for the core async event bus."""

    def setup_method(self) -> None:
        """Create a fresh EventBus for every test."""
        self.bus = EventBus()

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self) -> None:
        """A subscribed handler should receive the emitted event."""
        received: list[NayakEvent] = []

        async def handler(event: NayakEvent) -> None:
            received.append(event)

        self.bus.subscribe(EventType.AGENT_STARTED, handler)

        event = NayakEvent(
            type=EventType.AGENT_STARTED,
            payload={"goal": "test-goal"},
            source="test",
        )
        await self.bus.emit(event)

        assert len(received) == 1
        assert received[0].payload == {"goal": "test-goal"}
        assert received[0].source == "test"

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_multiple_handlers(self) -> None:
        """All subscribed handlers should be called on a single emit."""
        call_counts: list[int] = [0, 0, 0]

        async def handler_a(event: NayakEvent) -> None:
            call_counts[0] += 1

        async def handler_b(event: NayakEvent) -> None:
            call_counts[1] += 1

        async def handler_c(event: NayakEvent) -> None:
            call_counts[2] += 1

        self.bus.subscribe(EventType.STEP_STARTED, handler_a)
        self.bus.subscribe(EventType.STEP_STARTED, handler_b)
        self.bus.subscribe(EventType.STEP_STARTED, handler_c)

        await self.bus.emit(NayakEvent(
            type=EventType.STEP_STARTED,
            payload={"step": 1},
            source="test",
        ))

        assert all(c == 1 for c in call_counts), f"Expected all 1, got {call_counts}"

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        """After unsubscribing, the handler must NOT be called."""
        called = False

        async def handler(event: NayakEvent) -> None:
            nonlocal called
            called = True

        self.bus.subscribe(EventType.ACTION_TAKEN, handler)
        self.bus.unsubscribe(EventType.ACTION_TAKEN, handler)

        await self.bus.emit(NayakEvent(
            type=EventType.ACTION_TAKEN,
            payload={},
            source="test",
        ))

        assert not called

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash_bus(self) -> None:
        """A failing handler must not prevent other handlers from running."""
        second_called = False

        async def bad_handler(event: NayakEvent) -> None:
            raise RuntimeError("boom")

        async def good_handler(event: NayakEvent) -> None:
            nonlocal second_called
            second_called = True

        self.bus.subscribe(EventType.MEMORY_SAVED, bad_handler)
        self.bus.subscribe(EventType.MEMORY_SAVED, good_handler)

        # Should not raise
        await self.bus.emit(NayakEvent(
            type=EventType.MEMORY_SAVED,
            payload={},
            source="test",
        ))

        assert second_called

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_emit_error(self) -> None:
        """emit_error() should broadcast an ERROR_OCCURRED event."""
        received: list[NayakEvent] = []

        async def handler(event: NayakEvent) -> None:
            received.append(event)

        self.bus.subscribe(EventType.ERROR_OCCURRED, handler)

        await self.bus.emit_error(
            source="test",
            error=ValueError("something broke"),
        )

        assert len(received) == 1
        assert received[0].payload["error_type"] == "ValueError"
        assert received[0].payload["error_message"] == "something broke"
        assert received[0].source == "test"
