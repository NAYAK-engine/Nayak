"""
nayak/safety/base.py — NAYAK Safety Engine: core abstractions.

Defines the canonical data types and the abstract base class that every
safety backend (Layer 7) in NAYAK must implement to moderate AI autonomy.
"""

from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Threat Levels & Capabilities
# ─────────────────────────────────────────────────────────────────────────────

class ThreatLevel(Enum):
    """Categorisation of security risks during action execution."""
    SAFE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class CapabilityFlag(Enum):
    """Discrete granular permissions that an agent can possess."""
    BROWSE_WEB = auto()
    READ_FILES = auto()
    WRITE_FILES = auto()
    EXECUTE_CODE = auto()
    CONTROL_HARDWARE = auto()
    NETWORK_ACCESS = auto()
    MEMORY_ACCESS = auto()
    INTER_AGENT_COMM = auto()


@dataclass
class SafetyViolation:
    """An immutable record of a blocked unsafe behavior.

    Attributes:
        threat_level:   Evaluated severity of the blocked action.
        source:         Module originating the violation.
        description:    Log context for why it failed.
        action_blocked: Serialized representation of the prohibited intent.
        resolved:       Whether human verification bypassed this violation.
        violation_id:   Unique sequential tracking ID.
        timestamp:      Time of incident string.
    """
    threat_level: ThreatLevel
    source: str
    description: str
    action_blocked: str
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# SafetyBase
# ─────────────────────────────────────────────────────────────────────────────

class SafetyBase(abc.ABC):
    """Abstract base class for all NAYAK Layer-7 safety backends.

    Class Attributes:
        layer (int): Always ``7``.
    """

    layer: int = 7

    def __init__(self) -> None:
        self.violations: list[SafetyViolation] = []
        self.enabled_capabilities: set[CapabilityFlag] = set(CapabilityFlag)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Registry alias for the instantiated backend."""
        pass

    @abc.abstractmethod
    async def check(self, action_type: str, payload: dict) -> ThreatLevel:
        """Inspect a payload and return its evaluated threat severity."""
        pass

    @abc.abstractmethod
    async def emergency_stop(self) -> None:
        """Actuate a global kill-switch procedure."""
        pass

    # ── Concrete Methods ──────────────────────────────────────────────────────

    def is_allowed(self, capability: CapabilityFlag) -> bool:
        """Verify if the agent possesses the capability flag."""
        return capability in self.enabled_capabilities

    def disable_capability(self, capability: CapabilityFlag) -> None:
        """Revoke a capability, restricting the agent immediately."""
        if capability in self.enabled_capabilities:
            self.enabled_capabilities.remove(capability)
            logger.warning("SafetyBase: disabled capability %s", capability.name)

    def enable_capability(self, capability: CapabilityFlag) -> None:
        """Grant a capability securely."""
        self.enabled_capabilities.add(capability)
        logger.info("SafetyBase: enabled capability %s", capability.name)

    async def record_violation(self, violation: SafetyViolation) -> None:
        """Store the violation locally and emit a real-time event.

        Args:
            violation: Struct possessing incident details.
        """
        self.violations.append(violation)
        await bus.emit(NayakEvent(
            type=EventType.SAFETY_VIOLATION,
            payload={
                "threat_level": violation.threat_level.name,
                "source": violation.source,
                "description": violation.description
            },
            source=self.name,
        ))

    async def register(self) -> None:
        """Register the Safety instance directly into the OS Layer stack."""
        module = NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Safety engine",
            instance=self,
        )
        await registry.register(module)
        registry.set_status(self.name, ModuleStatus.READY)
        
        logger.info("SafetyBase: '%s' registered and READY", self.name)

        await bus.emit(NayakEvent(
            type=EventType.SAFETY_READY,
            payload={"module": self.name, "layer": self.layer},
            source=self.name,
        ))

__all__ = [
    "ThreatLevel",
    "CapabilityFlag",
    "SafetyViolation",
    "SafetyBase"
]
