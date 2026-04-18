"""
nayak/cognition/base.py — Abstract cognition interface for NAYAK.

All cognition backends (Ollama, Gemini, NVIDIA NIM, …) must subclass
:class:`CognitionBase` and implement the three abstract async methods.
The concrete lifecycle hooks (:meth:`init` and :meth:`stop`) handle
registry integration and bus signalling automatically — subclasses do
not need to override them.

Example::

    from nayak.cognition.base import CognitionBase
    from nayak.brain import Action

    class MyCognition(CognitionBase):
        @property
        def name(self) -> str:
            return "cognition.my_backend"

        async def plan(self, goal: str, context: str) -> list[str]:
            ...

        async def decide(self, goal, step, url, page_title,
                         page_text, screenshot_b64, memory_context) -> Action:
            ...

        async def generate(self, prompt: str) -> str:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from nayak.brain import Action
from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CognitionBase
# ─────────────────────────────────────────────────────────────────────────────

class CognitionBase(ABC):
    """Abstract base class for all NAYAK cognition backends.

    Provides:

    * A standard **layer** declaration (layer 3 — the cognition/brain layer).
    * Concrete :meth:`init` and :meth:`stop` hooks that plug the backend into
      the :mod:`nayak.core.registry` and :mod:`nayak.core.bus` without any
      extra boilerplate in the subclass.
    * Three **abstract async methods** that every backend must implement:
      :meth:`plan`, :meth:`decide`, and :meth:`generate`.

    .. note::
        Subclasses **must** implement the :attr:`name` abstract property and
        all three abstract methods.  They **should not** override :meth:`init`
        or :meth:`stop` unless they need to extend them (use ``await
        super().init()`` / ``await super().stop()`` in that case).
    """

    #: NAYAK OS layer for all cognition backends.
    layer: int = 3

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier for this cognition backend.

        Should follow the dot-namespaced convention, e.g.
        ``"cognition.ollama"`` or ``"cognition.gemini"``.
        """

    @abstractmethod
    async def plan(self, goal: str, context: str) -> list[str]:
        """Decompose *goal* into an ordered list of executable steps.

        Args:
            goal:    The high-level objective the agent must achieve.
            context: Any prior memory or environment context relevant to
                     planning (may be an empty string).

        Returns:
            An ordered ``list[str]`` where each element is a discrete,
            actionable step toward *goal*.  The list must contain at least
            one element.
        """

    @abstractmethod
    async def decide(
        self,
        goal: str,
        step: int,
        url: str,
        page_title: str,
        page_text: str,
        screenshot_b64: str,
        memory_context: str,
    ) -> Action:
        """Choose the next :class:`~nayak.brain.Action` to execute.

        This is the core perceive → think boundary: the backend receives a
        full snapshot of the current browser state and memory, and returns
        exactly one :class:`~nayak.brain.Action` for the agent to dispatch.

        Args:
            goal:           The top-level goal string.
            step:           Current step index (1-based).
            url:            URL of the currently visible page.
            page_title:     ``<title>`` of the current page.
            page_text:      Visible text content of the current page.
            screenshot_b64: Base-64-encoded PNG screenshot of the page.
            memory_context: Concatenated string of recent memory entries.

        Returns:
            A single :class:`~nayak.brain.Action` instance.
        """

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate free-form text from *prompt*.

        Used for report generation, summarisation, and any other open-ended
        text task that does not require structured :class:`~nayak.brain.Action`
        output.

        Args:
            prompt: The full prompt to send to the underlying language model.

        Returns:
            The model's text response as a plain ``str``.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks (concrete — subclasses may extend but not replace)
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Register this backend with the NAYAK module registry and signal readiness.

        Called automatically by :class:`~nayak.core.runtime.NayakRuntime`
        during startup.  If called manually, it is idempotent — registering
        the same name twice replaces the old entry with a warning (handled by
        the registry).

        Side-effects:
            * Adds a :class:`~nayak.core.registry.NayakModule` entry to
              :data:`~nayak.core.registry.registry`.
            * Sets the module status to ``READY``.
            * Emits :attr:`~nayak.core.bus.EventType.COGNITION_READY` on the
              global bus.
        """
        await registry.register(NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Cognition backend",
            instance=self,
        ))

        registry.set_status(self.name, ModuleStatus.READY)
        logger.info("CognitionBase: '%s' initialised on layer %d", self.name, self.layer)

        await bus.emit(NayakEvent(
            type=EventType.COGNITION_READY,
            payload={"name": self.name, "layer": self.layer},
            source=self.name,
        ))

    async def stop(self) -> None:
        """Mark this backend as stopped in the module registry.

        Called automatically by :class:`~nayak.core.runtime.NayakRuntime`
        during shutdown.  Safe to call even if :meth:`init` was never called
        (logs a warning and returns without raising).

        Side-effects:
            * Sets the module status to ``STOPPED`` in
              :data:`~nayak.core.registry.registry`.
        """
        module = registry.get(self.name)
        if module is None:
            logger.warning(
                "CognitionBase.stop(): module '%s' not found in registry — "
                "was init() called?",
                self.name,
            )
            return

        registry.set_status(self.name, ModuleStatus.STOPPED)
        logger.info("CognitionBase: '%s' stopped", self.name)


__all__ = ["CognitionBase"]
