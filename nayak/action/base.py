"""
nayak/action/base.py — Abstract action interface for NAYAK.

All action backends (e.g. Playwright computer control) must subclass
:class:`ActionBase` and implement its abstract methods. The concrete
lifecycle hook (:meth:`register`) handles module registry and bus
integration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ActionBase
# ─────────────────────────────────────────────────────────────────────────────

class ActionBase(ABC):
    """Abstract base class for all NAYAK action backends.

    Provides:
    * A standard **layer** declaration (layer 4 — the action/actuation layer).
    * Abstract async methods mapping to primitive real-world actions.
    * A concrete :meth:`register` hook to register with the runtime.
    """

    #: NAYAK OS layer for all action backends.
    layer: int = 4

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier for this action backend.

        Should follow the dot-namespaced convention, e.g.
        ``"action.playwright"`` or ``"action.pyautogui"``.
        """

    # ------------------------------------------------------------------
    # Abstract interface - Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def init(self) -> None:
        """Initialize the action backend (e.g. acquire resources)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop cleanly and release all system resources."""

    # ------------------------------------------------------------------
    # Abstract interface - Primitive Actions
    # ------------------------------------------------------------------
    
    @abstractmethod
    async def navigate(self, url: str) -> str:
        """Navigate to the specified URL."""

    @abstractmethod
    async def click(self, selector: str) -> str:
        """Click an element matching the given selector."""

    @abstractmethod
    async def click_coordinates(self, x: int, y: int) -> str:
        """Click at absolute viewport coordinates."""

    @abstractmethod
    async def type_text(self, selector: str, text: str) -> str:
        """Type text into an element matching the selector."""

    @abstractmethod
    async def scroll(self, direction: str, amount: int) -> str:
        """Scroll the viewport."""

    @abstractmethod
    async def extract(self) -> str:
        """Extract visible text content from the current state."""

    @abstractmethod
    async def save_file(self, filename: str, content: str) -> str:
        """Save a file to the host system."""

    @abstractmethod
    async def google_search(self, query: str) -> str:
        """Execute a clean Google search."""

    @abstractmethod
    async def press_key(self, key: str) -> str:
        """Press a keyboard key."""

    # ------------------------------------------------------------------
    # Lifecycle hooks 
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """Register this backend with the NAYAK module registry.

        Side-effects:
            * Adds a :class:`~nayak.core.registry.NayakModule` entry.
            * Sets the module status to ``READY``.
            * Emits :attr:`~nayak.core.bus.EventType.ACTION_READY`.
        """
        await registry.register(NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Action backend",
            instance=self,
        ))

        registry.set_status(self.name, ModuleStatus.READY)
        logger.info("ActionBase: '%s' registered on layer %d", self.name, self.layer)

        await bus.emit(NayakEvent(
            type=EventType.ACTION_READY,
            payload={"name": self.name, "layer": self.layer},
            source=self.name,
        ))

__all__ = ["ActionBase"]
