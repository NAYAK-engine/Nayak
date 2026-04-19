"""
nayak/sdk/base.py — NAYAK Developer Platform: core abstractions.

Defines the canonical data types and the abstract base classes for Layer 9.
This layer provides the SDK for external developers to build skills, plugins,
and integrations on top of the NAYAK core runtime.
"""

from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SkillType
# ─────────────────────────────────────────────────────────────────────────────

class SkillType(Enum):
    """The functional category of a NAYAK skill."""
    PERCEPTION    = auto()
    COGNITION     = auto()
    ACTION        = auto()
    MEMORY        = auto()
    COMMUNICATION = auto()
    SAFETY        = auto()
    UTILITY       = auto()
    INTEGRATION   = auto()


# ─────────────────────────────────────────────────────────────────────────────
# SkillManifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillManifest:
    """A standardized manifest defining a NAYAK Skill.

    Attributes:
        skill_id:     Automatically generated UUID4 string.
        name:         Human-readable name of the skill.
        version:      Semantic version string.
        skill_type:   Functional category of the skill.
        description:  Brief summary of what this skill provides.
        author:       Developer or organization. Defaults to "community".
        entry_point:  Python module path to the SkillBase implementation.
        dependencies: External skills or modules required.
        permissions:  Sensitive OS capabilities required by the skill.
        created_at:   Timestamp of manifest creation.
        metadata:     Arbitrary payload for extra contextual data.
    """
    name: str
    version: str
    skill_type: SkillType
    description: str
    entry_point: str
    author: str           = "community"
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str]  = field(default_factory=list)
    skill_id: str         = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float     = field(default_factory=time.time)
    metadata: dict        = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SkillBase
# ─────────────────────────────────────────────────────────────────────────────

class SkillBase(abc.ABC):
    """Abstract base class for all NAYAK functional skills.

    A Skill is a granular unit of logic that can be loaded into the NAYAK
    environment to extend its capabilities across any OS layer.
    """

    @property
    @abc.abstractmethod
    def manifest(self) -> SkillManifest:
        """The defining metadata for this skill."""
        pass

    @abc.abstractmethod
    async def on_load(self) -> None:
        """Called when the SDK successfully boots the skill plugin."""
        pass

    @abc.abstractmethod
    async def on_unload(self) -> None:
        """Called immediately before the skill is removed from the runtime."""
        pass

    @abc.abstractmethod
    async def execute(self, payload: dict) -> dict:
        """Trigger the skill's primary functional logic.

        Args:
            payload: Input parameters for the skill execution.

        Returns:
            A dictionary containing the results of the execution.
        """
        pass

    @property
    def skill_id(self) -> str:
        """Unique UUID identifying this skill instance."""
        return self.manifest.skill_id

    @property
    def skill_name(self) -> str:
        """Human readable name of the skill."""
        return self.manifest.name


# ─────────────────────────────────────────────────────────────────────────────
# DeveloperPlatformBase
# ─────────────────────────────────────────────────────────────────────────────

class DeveloperPlatformBase(abc.ABC):
    """Abstract base class for the NAYAK Layer-9 SDK backend.

    Manages the loading, lifecycle, and execution orchestration of external skills.

    Class Attributes:
        layer (int): Always ``9``.
    """

    layer: int = 9

    def __init__(self) -> None:
        self.loaded_skills: Dict[str, SkillBase] = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Registry identifier for the SDK platform."""
        pass

    @abc.abstractmethod
    async def load_skill(self, manifest: SkillManifest) -> bool:
        """Instantiate and initialize a skill into the live environment."""
        pass

    @abc.abstractmethod
    async def unload_skill(self, skill_id: str) -> bool:
        """Cleanly remove a skill, triggering its shutdown hooks."""
        pass

    @abc.abstractmethod
    async def execute_skill(self, skill_id: str, payload: dict) -> dict:
        """Proxy execution to a specific loaded skill."""
        pass

    @abc.abstractmethod
    async def list_skills(self) -> List[SkillManifest]:
        """Return manifests for all currently active skills."""
        pass

    # ── Concrete Methods ──────────────────────────────────────────────────────

    async def register(self) -> None:
        """Register the SDK platform with the NAYAK module registry."""
        module = NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Developer platform",
            instance=self,
        )
        await registry.register(module)
        registry.set_status(self.name, ModuleStatus.READY)

        logger.info("DeveloperPlatformBase: '%s' registered and READY", self.name)

        await bus.emit(NayakEvent(
            type=EventType.SDK_READY,
            payload={"module": self.name, "layer": self.layer},
            source=self.name,
        ))

# ─────────────────────────────────────────────────────────────────────────────
# Public Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "SkillType",
    "SkillManifest",
    "SkillBase",
    "DeveloperPlatformBase",
]
