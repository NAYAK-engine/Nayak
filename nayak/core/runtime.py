"""
nayak/core/runtime.py — The NAYAK Runtime Engine.

The ``NayakRuntime`` orchestrates the full module lifecycle: it starts all
registered modules in ascending layer order (1 → 9), keeps them running, and
shuts them down cleanly in reverse order (9 → 1).

It broadcasts its own state transitions on the global event bus so that any
subscriber can react without coupling to the runtime directly.

Usage::

    from nayak.core.runtime import runtime

    # With a coroutine (agent task, etc.)
    await runtime.run_until_complete(agent.run())

    # Or manually
    await runtime.start()
    # … your work …
    await runtime.stop()

    # Register a module before starting
    from nayak.core.registry import registry, NayakModule
    await registry.register(NayakModule(
        name="eyes.browser",
        version="1.0.0",
        layer=2,
        instance=browser,
    ))
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Coroutine, Any

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    """Configuration knobs for the NAYAK Runtime Engine.

    Attributes:
        name:              Human-readable runtime identifier.
        version:           Semantic version of this runtime build.
        log_level:         Python logging level applied at startup
                           (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …).
        auto_load_modules: When ``True``, :meth:`NayakRuntime.start` will call
                           ``init()`` on every registered module instance that
                           exposes it.  Set to ``False`` to manage module
                           initialisation manually.
    """

    name: str = "NAYAK"
    version: str = "0.2.0"
    log_level: str = "INFO"
    auto_load_modules: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# NayakRuntime
# ─────────────────────────────────────────────────────────────────────────────

class NayakRuntime:
    """Orchestrates the full NAYAK module lifecycle.

    On :meth:`start`, every module registered in the global
    :data:`~nayak.core.registry.registry` is initialised in layer order
    (1 → 9).  On :meth:`stop`, all ``READY`` modules are shut down in reverse
    order (9 → 1).  A single failing module never aborts the sequence — the
    exception is caught, logged, and the module is marked ``FAILED`` so the
    rest of the runtime continues normally.

    All state transitions are announced on the global
    :data:`~nayak.core.bus.bus` via :class:`~nayak.core.bus.EventType` events.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._bus = bus
        self._registry = registry
        self._running: bool = False
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the runtime and initialise all registered modules.

        Steps
        -----
        1. Emit :attr:`~EventType.RUNTIME_STARTING`.
        2. For each layer 1–9, call ``module.instance.init()`` (if it exists)
           on every module in that layer.  Modules that succeed are promoted to
           ``READY``; modules that raise are marked ``FAILED`` and the error
           is logged — the runtime continues regardless.
        3. Emit :attr:`~EventType.RUNTIME_READY` with a count of ``READY``
           modules.
        4. Log a full :meth:`~ModuleRegistry.summary` of the registry.

        Raises:
            RuntimeError: If the runtime is already running.
        """
        if self._running:
            raise RuntimeError("NayakRuntime is already running.")

        self._running = True
        self._start_time = time.time()

        logger.info(
            "NayakRuntime '%s' v%s starting …",
            self.config.name, self.config.version,
        )

        await self._bus.emit(NayakEvent(
            type=EventType.RUNTIME_STARTING,
            payload={"name": self.config.name, "version": self.config.version},
            source="runtime",
        ))

        if self.config.auto_load_modules:
            await self._init_modules_in_order()

        ready_count = sum(
            1 for m in self._registry.list_all()
            if m.status == ModuleStatus.READY
        )

        await self._bus.emit(NayakEvent(
            type=EventType.RUNTIME_READY,
            payload={"modules_loaded": ready_count},
            source="runtime",
        ))

        logger.info(
            "NayakRuntime ready — %d module(s) loaded.\n%s",
            ready_count,
            self._registry.summary(),
        )

    async def stop(self) -> None:
        """Stop the runtime and shut down all READY modules.

        Steps
        -----
        1. Emit :attr:`~EventType.RUNTIME_STOPPING`.
        2. For each layer 9–1, call ``module.instance.stop()`` (if it exists)
           on every ``READY`` module.  Exceptions are caught and logged per
           module; the shutdown sequence always completes.
        3. Set :attr:`_running` to ``False``.
        4. Emit :attr:`~EventType.RUNTIME_STOPPED` with the total uptime.

        This method is idempotent — calling it when not running logs a warning
        and returns immediately.
        """
        if not self._running:
            logger.warning("NayakRuntime.stop() called but runtime is not running.")
            return

        logger.info("NayakRuntime '%s' stopping …", self.config.name)

        await self._bus.emit(NayakEvent(
            type=EventType.RUNTIME_STOPPING,
            payload={"name": self.config.name},
            source="runtime",
        ))

        await self._stop_modules_in_order()

        uptime = time.time() - self._start_time
        self._running = False

        await self._bus.emit(NayakEvent(
            type=EventType.RUNTIME_STOPPED,
            payload={"uptime": uptime},
            source="runtime",
        ))

        logger.info(
            "NayakRuntime '%s' stopped. Uptime: %.2fs",
            self.config.name, uptime,
        )

    async def run_until_complete(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Start the runtime, await *coro*, then stop the runtime.

        :meth:`stop` is called in the ``finally`` block so the runtime always
        shuts down cleanly regardless of whether *coro* raises.

        Args:
            coro: Any awaitable (e.g. ``agent.run()``) to execute while the
                  runtime is running.

        Returns:
            The return value of *coro*.

        Raises:
            Any exception raised by *coro* after the runtime has been stopped.
        """
        await self.start()
        try:
            return await coro
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def uptime(self) -> float:
        """Seconds since the runtime was started, or ``0`` if not running."""
        return time.time() - self._start_time if self._running else 0.0

    @property
    def is_running(self) -> bool:
        """``True`` if the runtime has been started and not yet stopped."""
        return self._running

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _init_modules_in_order(self) -> None:
        """Initialise all registered modules in ascending layer order (1 → 9)."""
        for layer in range(1, 10):
            for module in self._registry.get_by_layer(layer):
                self._registry.set_status(module.name, ModuleStatus.INITIALIZING)
                logger.debug(
                    "Runtime: initialising [L%d] '%s' …", layer, module.name
                )
                try:
                    if module.instance is not None and hasattr(module.instance, "init"):
                        result = module.instance.init()
                        if asyncio.iscoroutine(result):
                            await result
                    self._registry.set_status(module.name, ModuleStatus.READY)
                    logger.info(
                        "Runtime: [L%d] '%s' — READY", layer, module.name
                    )
                except Exception as exc:  # noqa: BLE001
                    self._registry.set_status(module.name, ModuleStatus.FAILED)
                    logger.error(
                        "Runtime: [L%d] '%s' init() raised — marking FAILED: %s",
                        layer, module.name, exc, exc_info=True,
                    )

    async def _stop_modules_in_order(self) -> None:
        """Stop all READY modules in descending layer order (9 → 1)."""
        for layer in range(9, 0, -1):
            for module in self._registry.get_by_layer(layer):
                if module.status != ModuleStatus.READY:
                    continue
                logger.debug(
                    "Runtime: stopping [L%d] '%s' …", layer, module.name
                )
                try:
                    if module.instance is not None and hasattr(module.instance, "stop"):
                        result = module.instance.stop()
                        if asyncio.iscoroutine(result):
                            await result
                    self._registry.set_status(module.name, ModuleStatus.STOPPED)
                    logger.info(
                        "Runtime: [L%d] '%s' — STOPPED", layer, module.name
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Runtime: [L%d] '%s' stop() raised: %s",
                        layer, module.name, exc, exc_info=True,
                    )
                    # Still mark as stopped — we are shutting down regardless
                    self._registry.set_status(module.name, ModuleStatus.STOPPED)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

runtime: NayakRuntime = NayakRuntime(RuntimeConfig())
"""The process-wide NAYAK runtime engine.

Import and use this singleton directly::

    from nayak.core.runtime import runtime
"""

__all__ = ["RuntimeConfig", "NayakRuntime", "runtime"]
