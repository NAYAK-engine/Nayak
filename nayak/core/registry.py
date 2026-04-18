"""
nayak/core/registry.py — NAYAK Module Registry.

Tracks every runtime module that has been registered with the NAYAK engine.
Each module is associated with an OS layer (1-9), a lifecycle status, and an
optional reference to the live module instance.

The registry emits bus events on every structural change so that any subscriber
can react to modules coming on- or offline without polling.

Usage::

    from nayak.core.registry import registry, NayakModule, ModuleStatus

    # Register a module
    await registry.register(NayakModule(
        name="eyes.browser",
        version="1.0.0",
        layer=2,
        description="Playwright-backed browser perception layer",
        instance=browser_obj,
    ))

    # Promote its status
    registry.set_status("eyes.browser", ModuleStatus.READY)

    # Query
    layer2_modules = registry.get_by_layer(2)
    print(registry.summary())

    # Unregister
    await registry.unregister("eyes.browser")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from nayak.core.bus import EventType, NayakEvent, bus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ModuleStatus
# ─────────────────────────────────────────────────────────────────────────────

class ModuleStatus(Enum):
    """Lifecycle states for a registered NAYAK module.

    Transitions typically follow:
        REGISTERED → INITIALIZING → READY
                                 ↘ FAILED
        READY → STOPPED
    """

    REGISTERED   = auto()  # Module has been added to the registry
    INITIALIZING = auto()  # Module is performing its async startup
    READY        = auto()  # Module is operational
    FAILED       = auto()  # Module encountered a fatal error
    STOPPED      = auto()  # Module was gracefully shut down


# ─────────────────────────────────────────────────────────────────────────────
# NayakModule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NayakModule:
    """Descriptor for a single NAYAK runtime module.

    Attributes:
        name:          Unique identifier for this module (e.g. ``"eyes.browser"``).
        version:       Semantic version string (e.g. ``"1.2.0"``).
        layer:         NAYAK OS layer this module belongs to (1–9).
        status:        Current :class:`ModuleStatus`; defaults to ``REGISTERED``.
        description:   Short human-readable description of the module's role.
        instance:      Reference to the live Python object backing this module.
                       ``None`` until the module is instantiated.
        registered_at: Unix timestamp recorded at object creation time.
    """

    name: str
    version: str
    layer: int
    status: ModuleStatus = ModuleStatus.REGISTERED
    description: str = ""
    instance: Any = None
    registered_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 1 <= self.layer <= 9:
            raise ValueError(
                f"NayakModule '{self.name}': layer must be 1-9, got {self.layer}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ModuleRegistry
# ─────────────────────────────────────────────────────────────────────────────

class ModuleRegistry:
    """Central registry for all NAYAK runtime modules.

    Stores modules in an internal ``dict[str, NayakModule]`` keyed by module
    name.  Every structural mutation (register / unregister) emits a
    corresponding bus event so that observers receive real-time notifications.

    This class is not thread-safe by itself — all callers are expected to run
    within the same asyncio event loop (the standard NAYAK assumption).
    """

    def __init__(self) -> None:
        self._modules: dict[str, NayakModule] = {}

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    async def register(self, module: NayakModule) -> None:
        """Add *module* to the registry and emit a ``MODULE_REGISTERED`` event.

        If a module with the same name is already registered it will be
        **replaced** and a warning will be logged.

        Args:
            module: The :class:`NayakModule` descriptor to register.
        """
        if module.name in self._modules:
            logger.warning(
                "ModuleRegistry: replacing already-registered module '%s'",
                module.name,
            )

        self._modules[module.name] = module
        logger.info(
            "ModuleRegistry: registered '%s' v%s on layer %d",
            module.name, module.version, module.layer,
        )

        await bus.emit(NayakEvent(
            type=EventType.MODULE_REGISTERED,
            payload={
                "name": module.name,
                "version": module.version,
                "layer": module.layer,
                "status": module.status.name,
            },
            source="registry",
        ))

    async def unregister(self, name: str) -> None:
        """Remove the module identified by *name* and emit a ``MODULE_UNREGISTERED`` event.

        If no module with that name exists, this is a no-op and a warning is
        logged.

        Args:
            name: The unique module name to remove.
        """
        module = self._modules.pop(name, None)
        if module is None:
            logger.warning(
                "ModuleRegistry: unregister called for unknown module '%s'", name
            )
            return

        logger.info("ModuleRegistry: unregistered '%s'", name)

        await bus.emit(NayakEvent(
            type=EventType.MODULE_UNREGISTERED,
            payload={
                "name": module.name,
                "version": module.version,
                "layer": module.layer,
                "status": module.status.name,
            },
            source="registry",
        ))

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get(self, name: str) -> NayakModule | None:
        """Fetch a module by its unique name.

        Args:
            name: The module name to look up.

        Returns:
            The :class:`NayakModule` if found, otherwise ``None``.
        """
        return self._modules.get(name)

    def get_by_layer(self, layer: int) -> list[NayakModule]:
        """Return all modules registered on the given OS layer.

        Args:
            layer: NAYAK OS layer number (1–9).

        Returns:
            A list of :class:`NayakModule` instances on *layer*, sorted by name.
            Returns an empty list if no modules are registered on that layer.
        """
        return sorted(
            (m for m in self._modules.values() if m.layer == layer),
            key=lambda m: m.name,
        )

    def list_all(self) -> list[NayakModule]:
        """Return all registered modules, sorted by layer then name.

        Returns:
            A list of every :class:`NayakModule` currently in the registry.
        """
        return sorted(
            self._modules.values(),
            key=lambda m: (m.layer, m.name),
        )

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------

    def set_status(self, name: str, status: ModuleStatus) -> None:
        """Update the lifecycle status of a registered module.

        Args:
            name:   The unique module name.
            status: The new :class:`ModuleStatus` to apply.

        Raises:
            KeyError: If no module with *name* is registered.
        """
        module = self._modules.get(name)
        if module is None:
            raise KeyError(f"ModuleRegistry: no module named '{name}'")

        old = module.status
        module.status = status
        logger.debug(
            "ModuleRegistry: '%s' status %s → %s",
            name, old.name, status.name,
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted multi-line string describing all registered modules.

        Each line follows the pattern::

            [layer] name vVERSION — STATUS  (description)

        Modules are sorted by layer, then name.

        Returns:
            A human-readable registry summary, or a placeholder message when
            the registry is empty.
        """
        if not self._modules:
            return "ModuleRegistry: (empty)"

        lines: list[str] = ["ModuleRegistry — registered modules:", ""]
        for module in self.list_all():
            desc = f"  ({module.description})" if module.description else ""
            lines.append(
                f"  [L{module.layer}] {module.name} v{module.version}"
                f" — {module.status.name}{desc}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

registry: ModuleRegistry = ModuleRegistry()
"""The process-wide NAYAK module registry.

Import and use this singleton directly::

    from nayak.core.registry import registry
"""

__all__ = ["ModuleStatus", "NayakModule", "ModuleRegistry", "registry"]
