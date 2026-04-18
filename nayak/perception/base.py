"""
nayak/perception/base.py — Abstract perception interface for NAYAK.

All perception backends (e.g. Playwright browser vision, USB cameras, etc.)
must subclass :class:`PerceptionBase` and implement its abstract methods.
The concrete lifecycle hook (:meth:`register`) handles module registry and
bus integration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PerceptionBase
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionBase(ABC):
    """Abstract base class for all NAYAK perception backends.

    Provides:
    * A standard **layer** declaration (layer 2 — the perception layer).
    * Abstract async methods mapping to sensor acquisition.
    * A concrete :meth:`register` hook to register with the runtime.
    """

    #: NAYAK OS layer for all perception backends.
    layer: int = 2

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier for this perception backend.

        Should follow the dot-namespaced convention, e.g.
        ``"perception.browser"`` or ``"perception.camera"``.
        """

    # ------------------------------------------------------------------
    # Abstract interface - Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def init(self) -> None:
        """Initialize the perception backend (e.g. acquire devices)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop cleanly and release all system resources."""

    @abstractmethod
    async def start(self) -> None:
        """Start actively reading sensors/state (if applicable)."""

    # ------------------------------------------------------------------
    # Abstract interface - Senses
    # ------------------------------------------------------------------

    @abstractmethod
    async def see(self) -> Any:
        """Capture and return the current state of the world.

        Returns:
            An object representing the current perception state (e.g. a
            ``PageState`` or raw camera frame).
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks 
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """Register this backend with the NAYAK module registry.

        Side-effects:
            * Adds a :class:`~nayak.core.registry.NayakModule` entry.
            * Sets the module status to ``READY``.
            * Emits :attr:`~nayak.core.bus.EventType.PERCEPTION_READY`.
        """
        await registry.register(NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Perception backend",
            instance=self,
        ))

        registry.set_status(self.name, ModuleStatus.READY)
        logger.info("PerceptionBase: '%s' registered on layer %d", self.name, self.layer)

        await bus.emit(NayakEvent(
            type=EventType.PERCEPTION_READY,
            payload={"name": self.name, "layer": self.layer},
            source=self.name,
        ))

__all__ = ["PerceptionBase"]
