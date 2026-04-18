"""
nayak/memory/base.py — Abstract memory interface for NAYAK.

All memory backends must subclass :class:`MemoryBase` and implement the
abstract async methods. The concrete lifecycle hook (:meth:`register`)
handles registry integration and bus signalling automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryBase
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBase(ABC):
    """Abstract base class for all NAYAK memory backends.

    Provides:
    * A standard **layer** declaration (layer 5 — the memory layer).
    * Abstract async methods for memory operations.
    * A concrete :meth:`register` hook to register with the runtime.
    """

    #: NAYAK OS layer for all memory backends.
    layer: int = 5

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier for this memory backend.

        Should follow the dot-namespaced convention, e.g.
        ``"memory.sqlite"`` or ``"memory.redis"``.
        """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def init(self) -> None:
        """Initialize the memory backend (e.g. database connections)."""

    @abstractmethod
    async def close(self) -> None:
        """Close the memory backend gracefully."""

    @abstractmethod
    async def save(self, step: int, action: str, result: str, goal: str) -> None:
        """Save a discrete step to memory.

        Args:
            step:   The step number.
            action: The string representation of the action taken.
            result: The string result of the action.
            goal:   The current high-level goal.
        """

    @abstractmethod
    async def get_recent(self, n: int = 10) -> list[str]:
        """Fetch the *n* most recent memories.

        Args:
            n: Number of recent items to return.

        Returns:
            A list of formatted memory strings.
        """

    @abstractmethod
    async def search(self, query: str) -> list[str]:
        """Search past memories for *query*.

        Args:
            query: The text to search for.

        Returns:
            A list of matching memory strings.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories associated with the current session/agent."""

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """Register this backend with the NAYAK module registry.

        Called to integrate the memory backend into the runtime.

        Side-effects:
            * Adds a :class:`~nayak.core.registry.NayakModule` entry.
            * Sets the module status to ``READY``.
            * Emits :attr:`~nayak.core.bus.EventType.MEMORY_READY`.
        """
        await registry.register(NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Memory backend",
            instance=self,
        ))

        registry.set_status(self.name, ModuleStatus.READY)
        logger.info("MemoryBase: '%s' registered on layer %d", self.name, self.layer)

        await bus.emit(NayakEvent(
            type=EventType.MEMORY_READY,
            payload={"name": self.name, "layer": self.layer},
            source=self.name,
        ))

__all__ = ["MemoryBase"]
