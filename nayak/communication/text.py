"""
nayak/communication/text.py — NAYAK Communication: Text backend.

Provides a concrete :class:`CommunicationBase` implementation for text-based
interactions. It handles simple conversational text routing between the human 
operator, the NAYAK agent, and potentially other robots via an internal inbox 
and outbox queuing mechanism.

Usage::

    from nayak.communication.text import text_comm

    await text_comm.init()

    # NAYAK speaks cleanly to the terminal
    await text_comm.say("I have found the results.")

    # Operator injects a textual intent into NAYAK's queues
    await text_comm.hear("Stop execution immediately.")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from rich.console import Console

from nayak.core.registry import ModuleStatus, registry
from nayak.communication.base import CommunicationBase, Message, MessageType

logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# TextCommunication
# ─────────────────────────────────────────────────────────────────────────────

class TextCommunication(CommunicationBase):
    """Text communication backend for the NAYAK Communication Engine.

    Acts as an asynchronous message bus adapter bridging standard textual
    interactions into the broader NAYAK Layer 6 abstractions. Manages internal
    inbox/outbox queues for rapid processing.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._outbox: list[Message] = []
        self._handlers: dict[MessageType, list[Callable[[Message], Awaitable[None]]]] = {}

    @property
    def name(self) -> str:
        """Dot-namespaced registry identifier for this backend.

        Returns:
            ``"text-communication"``
        """
        return "text-communication"

    # ── Core async interface ──────────────────────────────────────────────────

    async def send(self, message: Message) -> bool:
        """Transmit a specific message over the text interface.

        Appends the message to the internal outbox registry and broadcasts it 
        onto the global NAYAK event bus.

        Args:
            message: The Message payload.

        Returns:
            Always ``True``.
        """
        self._outbox.append(message)
        logger.debug("TextComm: Sent message -> %s (Type: %s)", message.content, message.type.name)
        await self.emit_message_event(message)
        return True

    async def receive(self) -> Message | None:
        """Check the text inbox for waiting messages.

        Returns:
            The next waiting :class:`Message`, or ``None`` if the inbox is empty.
        """
        if self._inbox.empty():
            return None
        return await self._inbox.get()

    async def broadcast(self, content: str, sender: str) -> bool:
        """Blast text unconditionally to all observers on this channel.

        Args:
            content: The text payload to transmit.
            sender: The entity initiating the broadcast.

        Returns:
            Always ``True``.
        """
        msg = Message(
            type=MessageType.BROADCAST,
            sender=sender,
            content=content
        )
        return await self.send(msg)

    # ── Text-specific Shortcuts ───────────────────────────────────────────────

    async def say(self, content: str) -> bool:
        """Shortcut for NAYAK talking cleanly to the human operator.

        Automatically routes a ``ROBOT_TO_HUMAN`` message designated from "nayak"
        to "human" and renders the rich payload visually in the terminal.

        Args:
            content: What NAYAK is communicating.

        Returns:
            Always ``True``.
        """
        msg = Message(
            type=MessageType.ROBOT_TO_HUMAN,
            sender="nayak",
            receiver="human",
            content=content
        )
        await self.send(msg)
        console.print(f"[bold cyan][NAYAK says][/bold cyan] {content}")
        return True

    async def hear(self, content: str, sender: str = "human") -> None:
        """Shortcut for forcing NAYAK to listen to inbound text commands.

        Constructs a ``HUMAN_TO_ROBOT`` message, places it securely into the 
        waiting inbox queue, and emits the global message received event.

        Args:
            content: The payload received from the human/system.
            sender: Optional explicit sender override.
        """
        msg = Message(
            type=MessageType.HUMAN_TO_ROBOT,
            sender=sender,
            receiver="nayak",
            content=content
        )
        await self._inbox.put(msg)
        await self.emit_message_event(msg)
        logger.debug("TextComm: Heard message: '%s'", content)

    async def command(self, content: str, receiver: str) -> bool:
        """Shortcut for dispatching strict execution directives.

        Args:
            content: The command payload.
            receiver: The designated target of the command.

        Returns:
            Always ``True``.
        """
        msg = Message(
            type=MessageType.COMMAND,
            sender="nayak",
            receiver=receiver,
            content=content
        )
        return await self.send(msg)

    async def get_outbox(self) -> list[Message]:
        """Fetch a snapshot of all successfully transmitted outbound messages.

        Returns:
            A shallow copy of the outbox container.
        """
        return list(self._outbox)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Initialise the Text Communication engine and register it with NAYAK."""
        await self.register()
        logger.info("TextComm: Text communication initialized")

    async def stop(self) -> None:
        """Gracefully shut down the Text Communication engine."""
        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("TextComm: stopped cleanly")
        except KeyError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

text_comm: TextCommunication = TextCommunication()
"""Process-wide Text Communication singleton.

Import and use directly — no instantiation required::

    from nayak.communication.text import text_comm

    await text_comm.init()
    await text_comm.say("Hello World!")
    await text_comm.stop()
"""

__all__ = ["TextCommunication", "text_comm"]
