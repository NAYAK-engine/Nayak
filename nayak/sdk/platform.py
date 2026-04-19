"""
nayak/sdk/platform.py — NAYAK Developer Platform: SDK engine.

Provides the concrete implementation of the :class:`DeveloperPlatformBase`.
This engine manages the lifecycle of external skills, allowing them to be
dynamically loaded, executed, and unloaded at runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import inspect
from typing import List, Dict, Any

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, registry
from nayak.sdk.base import DeveloperPlatformBase, SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NayakPlatform
# ─────────────────────────────────────────────────────────────────────────────

class NayakPlatform(DeveloperPlatformBase):
    """The concrete operational developer platform for NAYAK.

    Handles dynamic skill loading, execution, and template generation for
    ecosystem expansion.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skills_registry: Dict[str, SkillManifest] = {}
        self._skills_dir: str = "skills"
        os.makedirs(self._skills_dir, exist_ok=True)

    @property
    def name(self) -> str:
        """Registry identifier for the SDK platform."""
        return "nayak-platform"

    # ── Skill Management ──────────────────────────────────────────────────────

    async def load_skill(self, manifest: SkillManifest) -> bool:
        """Dynamically load and initialize a skill plugin.

        Args:
            manifest: Defining metadata for the skill.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        logger.info("SDK: Attempting to load skill '%s' v%s...", manifest.name, manifest.version)
        
        try:
            # Dynamic import of the skill module
            module = importlib.import_module(manifest.entry_point)
            
            # Locate the SkillBase implementation in the module
            skill_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, SkillBase) and obj is not SkillBase:
                    skill_class = obj
                    break
            
            if not skill_class:
                logger.error("SDK: Failed to find SkillBase subclass in module %s", manifest.entry_point)
                return False

            # Instantiate and trigger on_load lifecycle hook
            instance = skill_class()
            await instance.on_load()
            
            # Store in internal registries
            self.loaded_skills[manifest.skill_id] = instance
            self._skills_registry[manifest.skill_id] = manifest
            
            await bus.emit(NayakEvent(
                type=EventType.SKILL_LOADED,
                payload={
                    "name": manifest.name,
                    "version": manifest.version,
                    "type": manifest.skill_type.name,
                    "skill_id": manifest.skill_id
                },
                source=self.name
            ))
            
            logger.info("SDK: Successfully loaded skill '%s' (%s)", manifest.name, manifest.skill_id)
            return True

        except Exception as exc:
            logger.error("SDK: Critical failure loading skill '%s': %s", manifest.name, exc)
            return False

    async def unload_skill(self, skill_id: str) -> bool:
        """Shut down and unregister a loaded skill safely.

        Args:
            skill_id: The unique identifier for the skill instance.

        Returns:
            ``True`` on success, ``False`` if skill not found or unload failed.
        """
        if skill_id not in self.loaded_skills:
            logger.warning("SDK: Unload requested for invalid skill_id: %s", skill_id)
            return False

        try:
            skill = self.loaded_skills[skill_id]
            logger.info("SDK: Unloading skill '%s' (%s)", skill.skill_name, skill_id)
            
            await skill.on_unload()
            
            del self.loaded_skills[skill_id]
            del self._skills_registry[skill_id]
            
            await bus.emit(NayakEvent(
                type=EventType.SKILL_UNLOADED,
                payload={"skill_id": skill_id},
                source=self.name
            ))
            
            return True
        except Exception as exc:
            logger.error("SDK: Error during unload of skill %s: %s", skill_id, exc)
            return False

    async def execute_skill(self, skill_id: str, payload: dict) -> dict:
        """Trigger the execution of a specific skill's logic.

        Args:
            skill_id: The unique identifier for the skill instance.
            payload: Input parameters for the skill.

        Returns:
            A dictionary containing the results or an error message.
        """
        if skill_id not in self.loaded_skills:
            return {"error": "skill not found"}
        
        try:
            skill = self.loaded_skills[skill_id]
            return await skill.execute(payload)
        except Exception as exc:
            logger.error("SDK: Skill execution error (%s): %s", skill_id, exc)
            return {"error": str(exc)}

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
            name: Human-readable name (used as filename and class name).
            skill_type: Functional category category.

        Returns:
            The absolute path to the generated file.
        """
        class_name = "".join(x.capitalize() for x in name.replace("-", " ").replace("_", " ").split())
        safe_filename = name.lower().replace(" ", "_").replace("-", "_")
        file_path = os.path.join(self._skills_dir, f"{safe_filename}.py")

        template = f'''"""
{name} — NAYAK Developer Skill.

Automatically generated by the NAYAK SDK platform.
"""

from __future__ import annotations
import logging
from nayak.sdk.base import SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)

class {class_name}(SkillBase):
    """Custom skill implementation for NAYAK."""

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
        """Defining metadata for the skill."""
        return self._manifest

    async def on_load(self) -> None:
        """Hook triggered when NAYAK boots this skill."""
        logger.info("{name}: booted")

    async def on_unload(self) -> None:
        """Hook triggered when NAYAK shuts down this skill."""
        logger.info("{name}: shutdown")

    async def execute(self, payload: dict) -> dict:
        """Primary functional logic for the skill.
        
        Args:
            payload: Input parameters dict.
            
        Returns:
            Execution results dict.
        """
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

__all__ = ["NayakPlatform", "platform"]
