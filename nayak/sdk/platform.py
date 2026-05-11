"""
nayak/sdk/platform.py — NAYAK Developer Platform: SDK engine.

Provides the concrete implementation of the :class:`DeveloperPlatformBase`.
This engine manages the lifecycle of external skills, allowing them to be
dynamically loaded, executed, and unloaded at runtime.

All skill execution is routed through :class:`SkillSandbox` to ensure a
crashing or hanging skill can never impact the NAYAK core runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
from typing import List, Dict, Any

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, registry
from nayak.sdk.base import DeveloperPlatformBase, SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SkillSandbox
# ─────────────────────────────────────────────────────────────────────────────

class SkillSandbox:
    """Isolates skill execution from the NAYAK core runtime.

    A crashed or hanging skill must never affect HAL, Safety, or any other
    OS layer.  Every call to :meth:`execute_safe` is wrapped with:

    * A hard asyncio timeout.
    * Full exception isolation — uncaught errors inside a skill produce a
      structured error dict rather than propagating up to the caller.
    * Return-type validation — skills that return non-dict values are caught.

    On success the result dict receives an additional ``_sandbox`` key set to
    ``"ok"`` so callers can distinguish sandbox-executed results from raw ones.

    Args:
        timeout_seconds: Maximum seconds a skill's ``execute()`` may run
                         before being forcibly cancelled.  Defaults to 10.
    """

    def __init__(self, timeout_seconds: int = 10) -> None:
        self._timeout = timeout_seconds

    async def execute_safe(self, skill: SkillBase, payload: dict) -> dict:
        """Execute *skill* with full isolation guarantees.

        Attempts ``await skill.execute(payload)`` inside:
        * :func:`asyncio.wait_for` to enforce a hard deadline.
        * A ``try/except`` block so any uncaught exception is converted to
          a structured error response.

        Args:
            skill:   The :class:`SkillBase` instance to execute.
            payload: Input parameters forwarded to ``skill.execute()``.

        Returns:
            On success: the result dict with ``{"_sandbox": "ok"}`` added.

            On timeout:
            ``{"error": "skill timed out after Xs", "skill": skill_name}``

            On crash:
            ``{"error": "<exception message>", "skill": skill_name}``

            On invalid return type:
            ``{"error": "invalid result type", "skill": skill_name}``
        """
        try:
            result = await asyncio.wait_for(
                skill.execute(payload),
                timeout=self._timeout,
            )
            if not isinstance(result, dict):
                logger.error(
                    "SkillSandbox: '%s' returned non-dict type %s",
                    skill.skill_name, type(result).__name__,
                )
                return {"error": "invalid result type", "skill": skill.skill_name}
            result["_sandbox"] = "ok"
            return result

        except asyncio.TimeoutError:
            logger.error(
                "SkillSandbox: '%s' timed out after %ds",
                skill.skill_name, self._timeout,
            )
            return {
                "error": f"skill timed out after {self._timeout}s",
                "skill": skill.skill_name,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SkillSandbox: '%s' raised an uncaught exception: %s",
                skill.skill_name, exc,
            )
            return {"error": str(exc), "skill": skill.skill_name}


# ─────────────────────────────────────────────────────────────────────────────
# NayakPlatform
# ─────────────────────────────────────────────────────────────────────────────

class NayakPlatform(DeveloperPlatformBase):
    """The concrete operational developer platform for NAYAK.

    Handles dynamic skill loading, execution, and template generation for
    ecosystem expansion.  All skill executions are routed through a
    :class:`SkillSandbox` instance so the core runtime is always protected.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skills_registry: Dict[str, SkillManifest] = {}
        self._skills_dir: str = "skills"
        self._sandbox: SkillSandbox = SkillSandbox(timeout_seconds=10)
        os.makedirs(self._skills_dir, exist_ok=True)

    @property
    def name(self) -> str:
        """Registry identifier for the SDK platform."""
        return "nayak-platform"

    # ── Skill Management ──────────────────────────────────────────────────────

    async def load_skill(self, manifest: SkillManifest) -> bool:
        """Dynamically load and initialize a skill plugin.

        Imports the module specified by ``manifest.entry_point``, locates the
        first :class:`SkillBase` subclass, instantiates it, and calls
        ``on_load()``.  If ``on_load()`` raises, the skill is marked FAILED and
        the platform continues — a bad skill must never crash the OS layer.

        Args:
            manifest: Defining metadata for the skill.

        Returns:
            ``True`` on success, ``False`` on any failure.
        """
        logger.info(
            "SDK: Attempting to load skill '%s' v%s...",
            manifest.name, manifest.version,
        )

        try:
            module = importlib.import_module(manifest.entry_point)

            skill_class = None
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, SkillBase)
                    and obj is not SkillBase
                ):
                    skill_class = obj
                    break

            if not skill_class:
                logger.error(
                    "SDK: No SkillBase subclass found in module %s",
                    manifest.entry_point,
                )
                return False

            instance = skill_class()

            # ── on_load() is sandboxed — a crash must not kill the platform ──
            try:
                await instance.on_load()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "SDK: Skill '%s' on_load() raised — marking FAILED: %s",
                    manifest.name, exc,
                )
                return False

            self.loaded_skills[manifest.skill_id] = instance
            self._skills_registry[manifest.skill_id] = manifest

            await bus.emit(NayakEvent(
                type=EventType.SKILL_LOADED,
                payload={
                    "name": manifest.name,
                    "version": manifest.version,
                    "type": manifest.skill_type.name,
                    "skill_id": manifest.skill_id,
                },
                source=self.name,
            ))

            logger.info(
                "SDK: Successfully loaded skill '%s' (%s)",
                manifest.name, manifest.skill_id,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SDK: Critical failure loading skill '%s': %s",
                manifest.name, exc,
            )
            return False

    async def unload_skill(self, skill_id: str) -> bool:
        """Shut down and unregister a loaded skill safely.

        Calls ``on_unload()`` inside a try/except so a crashing shutdown hook
        never blocks the unload sequence.

        Args:
            skill_id: The unique identifier for the skill instance.

        Returns:
            ``True`` on success, ``False`` if skill not found.
        """
        if skill_id not in self.loaded_skills:
            logger.warning("SDK: Unload requested for unknown skill_id: %s", skill_id)
            return False

        skill = self.loaded_skills[skill_id]
        logger.info("SDK: Unloading skill '%s' (%s)", skill.skill_name, skill_id)

        # on_unload() failure must never block the unload sequence
        try:
            await skill.on_unload()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SDK: Skill '%s' on_unload() raised (continuing anyway): %s",
                skill.skill_name, exc,
            )

        del self.loaded_skills[skill_id]
        if skill_id in self._skills_registry:
            del self._skills_registry[skill_id]

        await bus.emit(NayakEvent(
            type=EventType.SKILL_UNLOADED,
            payload={"skill_id": skill_id},
            source=self.name,
        ))
        return True

    async def execute_skill(self, skill_id: str, payload: dict) -> dict:
        """Execute a skill through the sandbox. Core runtime is always protected.

        All execution is delegated to :class:`SkillSandbox` which enforces
        timeouts, catches exceptions, and validates return types.

        Args:
            skill_id: The unique identifier for the skill instance.
            payload:  Input parameters forwarded to the skill.

        Returns:
            Result dict from the skill (with ``_sandbox: "ok"`` added),
            or a structured error dict if the skill failed.
        """
        if skill_id not in self.loaded_skills:
            return {"error": "skill not found"}
        skill = self.loaded_skills[skill_id]
        return await self._sandbox.execute_safe(skill, payload)

    async def list_skills(self) -> List[SkillManifest]:
        """List manifests for all currently active plugins.

        Returns:
            A list of :class:`SkillManifest` objects.
        """
        return list(self._skills_registry.values())

    # ── Developer Tooling ─────────────────────────────────────────────────────

    async def create_skill_template(self, name: str, skill_type: SkillType) -> str:
        """Generate a Python boilerplate file for a new developer skill.

        Args:
            name:       Human-readable name (used as filename and class name).
            skill_type: Functional category of the skill.

        Returns:
            The absolute path to the generated file.
        """
        class_name = "".join(
            x.capitalize()
            for x in name.replace("-", " ").replace("_", " ").split()
        )
        safe_filename = name.lower().replace(" ", "_").replace("-", "_")
        file_path = os.path.join(self._skills_dir, f"{safe_filename}.py")

        template = f'''\"""
{name} — NAYAK Developer Skill.

Automatically generated by the NAYAK SDK platform.
\"""

from __future__ import annotations
import logging
from nayak.sdk.base import SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)

class {class_name}(SkillBase):
    \"""Custom skill implementation for NAYAK.\"""

    def __init__(self) -> None:
        self._manifest = SkillManifest(
            name="{name}",
            version="0.1.0",
            skill_type=SkillType.{skill_type.name},
            description="Generated by NAYAK SDK platform.",
            entry_point="skills.{safe_filename}"
        )

    @property
    def manifest(self) -> SkillManifest:
        \"""Defining metadata for the skill.\"""
        return self._manifest

    async def on_load(self) -> None:
        logger.info("{name}: booted")

    async def on_unload(self) -> None:
        logger.info("{name}: shutdown")

    async def execute(self, payload: dict) -> dict:
        # TODO: Implement your logic here
        return {{"status": "success", "echo": payload}}
'''
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template)
        logger.info("SDK: Created skill template for '%s' at %s", name, file_path)
        return file_path

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Mount the Developer Platform into the OS stack."""
        await self.register()
        logger.info("Developer platform initialized")
        logger.info("SDK: Skills directory: %s", self._skills_dir)

    async def stop(self) -> None:
        """Gracefully shut down all active plugins and the SDK platform."""
        ids = list(self.loaded_skills.keys())
        for skill_id in ids:
            await self.unload_skill(skill_id)
        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("SDK: Platform shutdown complete.")
        except KeyError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

platform: NayakPlatform = NayakPlatform()
"""Global NAYAK Developer Platform instance."""

__all__ = ["SkillSandbox", "NayakPlatform", "platform"]
