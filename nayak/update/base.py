"""
nayak/update/base.py — NAYAK Update Engine: core abstractions.

Defines the canonical data types and the abstract base class for NAYAK's Layer 8
Update Engine. This layer handles OTA updates, skill installations, and system
evolution.
"""

from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UpdateStatus
# ─────────────────────────────────────────────────────────────────────────────

class UpdateStatus(Enum):
    """The lifecycle status of a skill package update."""
    PENDING = auto()
    DOWNLOADING = auto()
    INSTALLING = auto()
    COMPLETE = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


# ─────────────────────────────────────────────────────────────────────────────
# SkillPackage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillPackage:
    """A standardized packet representing a NAYAK skill or module update.

    Attributes:
        package_id:   Automatically generated UUID4 string.
        name:         Human-readable name of the skill/update.
        version:      Semantic version string.
        description:  Brief summary of what this package provides.
        author:       Creator of the package. Defaults to "community".
        size_bytes:   Total size of the package in bytes.
        checksum:     SHA256 hash for integrity verification.
        install_path: Local filesystem path where the package is installed.
        status:       Current lifecycle state of the package.
        created_at:   Creation timestamp.
        metadata:     Arbitrary payload for extra contextual data.
    """
    name: str
    version: str
    description: str
    author: str = "community"
    size_bytes: int = 0
    checksum: str = ""
    install_path: str = ""
    status: UpdateStatus = UpdateStatus.PENDING
    package_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# UpdateBase
# ─────────────────────────────────────────────────────────────────────────────

class UpdateBase(abc.ABC):
    """Abstract base class for all NAYAK Layer-8 update backends.

    Manages the lifecycle of system updates and agent skills.

    Class Attributes:
        layer (int): Always ``8``.
    """

    layer: int = 8

    def __init__(self) -> None:
        self.installed_packages: Dict[str, SkillPackage] = {}
        self.pending_updates: List[SkillPackage] = []

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Registry identifier for this update engine."""
        pass

    @abc.abstractmethod
    async def check_updates(self) -> List[SkillPackage]:
        """Query for available module or skill updates."""
        pass

    @abc.abstractmethod
    async def download(self, package: SkillPackage) -> bool:
        """Fetch a specific package to the local staging area."""
        pass

    @abc.abstractmethod
    async def install(self, package: SkillPackage) -> bool:
        """Mount and initialize a pre-downloaded package into the live OS."""
        pass

    @abc.abstractmethod
    async def rollback(self, package_id: str) -> bool:
        """Revert the system state after a failed or corrupted update."""
        pass

    @abc.abstractmethod
    async def list_installed(self) -> List[SkillPackage]:
        """Return all currently active skills and module updates."""
        pass

    # ── Concrete Methods ──────────────────────────────────────────────────────

    async def register(self) -> None:
        """Register the Update engine with the NAYAK module registry."""
        module = NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Update engine",
            instance=self,
        )
        await registry.register(module)
        registry.set_status(self.name, ModuleStatus.READY)

        logger.info("UpdateBase: '%s' registered and READY", self.name)

        await bus.emit(NayakEvent(
            type=EventType.UPDATE_READY,
            payload={"module": self.name, "layer": self.layer},
            source=self.name,
        ))

    async def emit_update_event(self, package: SkillPackage) -> None:
        """Broadcast a package installation success to the global NAYAK bus.

        Args:
            package: The SkillPackage object that was processed.
        """
        await bus.emit(NayakEvent(
            type=EventType.PACKAGE_INSTALLED,
            payload={
                "name": package.name,
                "version": package.version,
                "status": package.status.name
            },
            source=self.name,
        ))

# ─────────────────────────────────────────────────────────────────────────────
# Public Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "UpdateStatus",
    "SkillPackage",
    "UpdateBase",
]
