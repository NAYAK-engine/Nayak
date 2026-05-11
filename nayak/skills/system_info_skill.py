"""
nayak/skills/system_info_skill.py — NAYAK System Info Skill.

First-party skill that returns runtime system diagnostics including CPU,
memory, disk usage, and platform metadata.  Built on the NAYAK SDK (Layer 9).

Usage::

    from nayak.skills.system_info_skill import SystemInfoSkill

    skill = SystemInfoSkill()
    await skill.on_load()
    result = await skill.execute({})
    print(result)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from nayak.sdk.base import SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)


class SystemInfoSkill(SkillBase):
    """Return NAYAK runtime system information.

    Reports CPU, memory, and disk usage via ``psutil``, along with platform
    metadata and the current NAYAK version.

    Requires ``MEMORY_ACCESS`` permission.
    """

    def __init__(self) -> None:
        self._manifest = SkillManifest(
            name="system-info",
            version="1.0.0",
            skill_type=SkillType.UTILITY,
            description="Returns NAYAK runtime system information",
            author="nayak-core",
            entry_point="nayak.skills.system_info_skill",
            permissions=["MEMORY_ACCESS"],
        )

    @property
    def manifest(self) -> SkillManifest:
        """The defining metadata for this skill."""
        return self._manifest

    async def on_load(self) -> None:
        """Called when the SDK boots this skill."""
        logger.info("System Info Skill loaded")

    async def on_unload(self) -> None:
        """Called when the SDK shuts down this skill."""
        logger.info("System Info Skill unloaded")

    async def execute(self, payload: dict) -> dict:
        """Collect and return system diagnostics.

        Args:
            payload: Ignored — no input parameters required.

        Returns:
            A dict with ``cpu_percent``, ``memory_percent``, ``disk_percent``,
            ``platform``, ``python_version``, and ``nayak_version`` keys on
            success, or ``{"error": ...}`` on failure.
        """
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "platform": sys.platform,
                "python_version": sys.version,
                "nayak_version": "0.2.0",
            }

        except Exception as exc:
            logger.error("SystemInfoSkill: execute() failed: %s", exc)
            return {"error": str(exc)}


__all__ = ["SystemInfoSkill"]
