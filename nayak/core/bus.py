"""
nayak/core/bus.py — Lightweight async event bus for NAYAK.

Provides a publish/subscribe mechanism that decouples agent components.
Any module can emit events onto the bus; any subscriber will be called
asynchronously without the emitter needing to know who is listening.

Usage::

    from nayak.core.bus import bus, EventType, NayakEvent

    # Subscribe
    async def on_step(event: NayakEvent) -> None:
        print(f"Step started: {event.payload}")

    bus.subscribe(EventType.STEP_STARTED, on_step)

    # Emit
    await bus.emit(NayakEvent(
        type=EventType.STEP_STARTED,
        payload={"step": 1},
        source="agent",
    ))

    # Unsubscribe
    bus.unsubscribe(EventType.STEP_STARTED, on_step)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EventType
# ─────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    """All recognised NAYAK event types.

    Extend this enum to introduce new bus events — existing subscribers
    are unaffected.
    """

    AGENT_STARTED        = auto()  # Agent.run() entered
    AGENT_STOPPED        = auto()  # Agent fully shut down (finally block)
    STEP_STARTED         = auto()  # A new perceive-think-act-remember cycle began
    STEP_COMPLETED       = auto()  # A step finished successfully
    STEP_FAILED          = auto()  # A step timed out or raised an error
    ACTION_TAKEN         = auto()  # An action was executed by Computer
    MEMORY_SAVED         = auto()  # A step was persisted to the memory store
    GOAL_COMPLETED       = auto()  # ActionType.FINISH was returned
    ERROR_OCCURRED       = auto()  # Any unexpected error in any component
    PERCEPTION_READY     = auto()  # _perceive() finished; PageState available
    COGNITION_READY      = auto()  # _think() finished; Action chosen
    MEMORY_READY         = auto()  # Memory backend initialized and ready 
    MODULE_REGISTERED    = auto()  # A NayakModule was added to the ModuleRegistry
    MODULE_UNREGISTERED  = auto()  # A NayakModule was removed from the ModuleRegistry
    RUNTIME_STARTING     = auto()  # NayakRuntime.start() entered
    RUNTIME_READY        = auto()  # All modules initialised; runtime is operational
    RUNTIME_STOPPING     = auto()  # NayakRuntime.stop() entered
    RUNTIME_STOPPED      = auto()  # Runtime fully shut down


# ─────────────────────────────────────────────────────────────────────────────
# NayakEvent
# ─────────────────────────────────────────────────────────────────────────────

Handler = Callable[["NayakEvent"], Awaitable[None]]


@dataclass
class NayakEvent:
    """An immutable event that travels over the EventBus.

    Attributes:
        type:      The EventType that categorises this event.
        payload:   Arbitrary dict of data attached by the emitter.
        timestamp: Unix timestamp (seconds) set automatically on creation.
        source:    Human-readable name of the module that emitted this event.
    """

    type: EventType
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


# ─────────────────────────────────────────────────────────────────────────────
# EventBus
# ─────────────────────────────────────────────────────────────────────────────

class EventBus:
    """Async publish/subscribe event bus.

    All handlers are called with ``await`` in the order they subscribed.
    Handler exceptions are caught, logged, and never propagate back to the
    emitter — the bus never crashes due to a bad handler.

    Thread/task safety: the ``handlers`` dict is mutated only from the
    subscribe/unsubscribe methods, which are synchronous and fast.  ``emit``
    is async and iterates a snapshot of the current handler list, so late
    subscription changes during emission have no effect on the in-progress
    emit call.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register *handler* to be called whenever *event_type* is emitted.

        The same handler can safely be registered multiple times; it will be
        called once per registration.

        Args:
            event_type: The :class:`EventType` to listen for.
            handler:    An ``async`` callable that accepts a single
                        :class:`NayakEvent` argument.
        """
        self._handlers[event_type].append(handler)
        logger.debug(
            "EventBus: subscribed %s to %s (total=%d)",
            getattr(handler, "__qualname__", repr(handler)),
            event_type.name,
            len(self._handlers[event_type]),
        )

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove *handler* from the subscriber list for *event_type*.

        If *handler* is not currently registered, this is a no-op.

        Args:
            event_type: The :class:`EventType` to stop listening to.
            handler:    The exact callable reference previously passed to
                        :meth:`subscribe`.
        """
        try:
            self._handlers[event_type].remove(handler)
            logger.debug(
                "EventBus: unsubscribed %s from %s",
                getattr(handler, "__qualname__", repr(handler)),
                event_type.name,
            )
        except ValueError:
            logger.debug(
                "EventBus: unsubscribe called for unregistered handler %s on %s",
                getattr(handler, "__qualname__", repr(handler)),
                event_type.name,
            )

    async def emit(self, event: NayakEvent) -> None:
        """Broadcast *event* to all handlers subscribed to its type.

        Handlers are awaited sequentially in subscription order.  If a handler
        raises any exception, it is caught and logged at ERROR level; remaining
        handlers are still called.

        Args:
            event: The :class:`NayakEvent` to broadcast.
        """
        handlers = list(self._handlers.get(event.type, []))  # snapshot
        if not handlers:
            logger.debug("EventBus: emit %s — no subscribers", event.type.name)
            return

        logger.debug(
            "EventBus: emit %s from '%s' to %d subscriber(s)",
            event.type.name,
            event.source,
            len(handlers),
        )
        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "EventBus: handler %s raised on event %s: %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event.type.name,
                    exc,
                    exc_info=True,
                )

    async def emit_error(self, source: str, error: Exception) -> None:
        """Convenience shortcut to emit an :attr:`EventType.ERROR_OCCURRED` event.

        Args:
            source: Name of the component where the error originated.
            error:  The exception that was raised.
        """
        await self.emit(
            NayakEvent(
                type=EventType.ERROR_OCCURRED,
                payload={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                source=source,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

bus: EventBus = EventBus()
"""The process-wide NAYAK event bus.

Import and use this singleton directly::

    from nayak.core.bus import bus
"""

__all__ = ["EventType", "NayakEvent", "EventBus", "bus"]
