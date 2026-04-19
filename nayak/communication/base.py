"""
nayak/communication/base.py — NAYAK Communication Engine: core abstractions.

Defines the canonical data types and the abstract base class that every
communication backend in NAYAK must implement.

Concrete backends (e.g. a terminal CLI interface, a WebSocket server, or a 
Text-to-Speech voice engine) subclass :class:`CommunicationBase`, implement 
the abstract async interface, and call ``await self.register()`` once ready.
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
# MessageType
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(Enum):
    """The intent and modality of a communication payload.

    Members:
        TEXT:            Raw text interface transmission (e.g., CLI).
        VOICE:           Audio/spoken interface transmission.
        COMMAND:         Instructional directive expecting action.
        RESPONSE:        Reply to a previous command or query.
        BROADCAST:       Unicast or multicast transmission to all observers.
        ROBOT_TO_ROBOT:  Machine-level peer-to-peer swarm comms.
        HUMAN_TO_ROBOT:  Operator issuing commands to NAYAK.
        ROBOT_TO_HUMAN:  NAYAK communicating status to an operator.
    """
    TEXT           = auto()
    VOICE          = auto()
    COMMAND        = auto()
    RESPONSE       = auto()
    BROADCAST      = auto()
    ROBOT_TO_ROBOT = auto()
    HUMAN_TO_ROBOT = auto()
    ROBOT_TO_HUMAN = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Message
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A standardized packet encapsulating a communication payload.

    Used for all inbound and outbound data transmitted via the Comm Engine.

    Attributes:
        message_id: Automatically generated UUID4 string.
        type:       The intent and format category of the message.
        sender:     The name/ID of the originating entity or robot.
        receiver:   The target entity. Defaults to ``"broadcast"``.
        content:    The actual string content or transcript.
        timestamp:  Unix creation timestamp automatically generated.
        metadata:   Arbitrary payload for extra contextual data (e.g., voice tone).
    """
    type:       MessageType
    sender:     str
    content:    str
    receiver:   str   = "broadcast"
    message_id: str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:  float = field(default_factory=time.time)
    metadata:   dict  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# CommunicationBase
# ─────────────────────────────────────────────────────────────────────────────

class CommunicationBase(abc.ABC):
    """Abstract base class for all NAYAK Layer-6 communication backends.

    Concrete subclasses represent discrete communication channels. They must 
    implement ``send()``, ``receive()``, and ``broadcast()``. The base class 
    handles NAYAK OS module registration automatically.

    Class Attributes:
        layer (int): Always ``6``.
    """

    layer: int = 6

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Dot-namespaced registry identifier, e.g., ``"comm.cli"``."""
        pass

    @abc.abstractmethod
    async def send(self, message: Message) -> bool:
        """Transmit a specific message over this interface.

        Args:
            message: The Message object.

        Returns:
            ``True`` on success, ``False`` on transmission failure.
        """
        pass

    @abc.abstractmethod
    async def receive(self) -> Message | None:
        """Poll or await an incoming transmission on this interface.

        Returns:
            A new Message instance, or ``None`` if polling fails.
        """
        pass

    @abc.abstractmethod
    async def broadcast(self, content: str, sender: str) -> bool:
        """Quickly blast string content to all listeners on this interface.

        Args:
            content: The text payload to transmit.
            sender: The originator of the broadcast.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        pass


    # ── Concrete methods ──────────────────────────────────────────────────────

    async def register(self) -> None:
        """Register this backend with the NAYAK module registry.

        Completing this action promotes the internal status to READY and emits
        the COMM_READY bus event.
        """
        module = NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Communication backend",
            instance=self,
        )
        await registry.register(module)
        registry.set_status(self.name, ModuleStatus.READY)

        logger.info("CommunicationBase: '%s' registered and READY", self.name)

        await bus.emit(NayakEvent(
            type=EventType.COMM_READY,
            payload={"module": self.name, "layer": self.layer},
            source=self.name,
        ))

    async def emit_message_event(self, message: Message) -> None:
        """Broadcast a generic MESSAGE_RECEIVED event to the global NAYAK bus.

        Signals to cognition or logging layers that a verified inbound chunk
        was obtained by this subsystem.

        Args:
            message: The evaluated Message object.
        """
        await bus.emit(NayakEvent(
            type=EventType.MESSAGE_RECEIVED,
            payload={
                "sender": message.sender,
                "type": message.type.name,
                "content": message.content
            },
            source=self.name,
        ))

# ─────────────────────────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "MessageType",
    "Message",
    "CommunicationBase",
]
