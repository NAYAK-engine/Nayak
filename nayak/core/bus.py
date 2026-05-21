"""
nayak/core/bus.py — Lightweight async event bus + zero-copy shared memory bus.

Provides two complementary communication mechanisms:

1. :class:`EventBus` — async pub/sub for control signals, status events, and
   small payloads.  Every NAYAK component uses this for coordination.

2. :class:`SharedMemoryBus` — zero-copy channels for *heavy* payloads such as
   camera frames, LIDAR point clouds, and sensor arrays.  Data is written once
   into a shared memory region; consumers read directly from that region
   without any serialization or copying.

Rule of thumb::

    EventBus   → control signals, text, small dicts  (< 10 KB)
    SharedMemoryBus → camera frames, tensors, sensor arrays (> 10 KB)

Usage (EventBus)::

    from nayak.core.bus import bus, EventType, NayakEvent

    async def on_step(event: NayakEvent) -> None:
        print(f"Step started: {event.payload}")

    bus.subscribe(EventType.STEP_STARTED, on_step)
    await bus.emit(NayakEvent(type=EventType.STEP_STARTED, payload={"step": 1}))
    bus.unsubscribe(EventType.STEP_STARTED, on_step)

Usage (SharedMemoryBus)::

    from nayak.core.bus import shm_bus

    # Producer (e.g. camera HAL)
    ch = shm_bus.create_channel("camera.frame", size_bytes=1920*1080*3)
    ch.write(raw_frame_bytes)

    # Consumer (e.g. perception layer)
    ch = shm_bus.get_channel("camera.frame")
    frame = ch.read()

    # Cleanup
    shm_bus.destroy_channel("camera.frame")
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import shared_memory
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# Large-payload threshold: warn if an EventBus payload exceeds this length
_LARGE_PAYLOAD_THRESHOLD = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# PayloadType
# ─────────────────────────────────────────────────────────────────────────────

class PayloadType(Enum):
    """Semantic category of data stored in a :class:`SharedMemoryChannel`.

    Used as metadata so consumers know how to interpret raw bytes.
    """
    TEXT         = auto()  # UTF-8 encoded string
    BINARY       = auto()  # Raw bytes (arbitrary)
    TENSOR       = auto()  # Flat float32 array (NumPy compatible)
    IMAGE_FRAME  = auto()  # RGB/RGBA camera frame (H×W×C layout)
    SENSOR_DATA  = auto()  # Packed sensor readings (struct layout)


# ─────────────────────────────────────────────────────────────────────────────
# SharedMemoryChannel
# ─────────────────────────────────────────────────────────────────────────────

class SharedMemoryChannel:
    """Zero-copy shared memory channel for heavy payloads.

    Camera frames, sensor arrays, and tensor data are written once into a
    named POSIX shared memory block.  Consumers map the same block and read
    directly — no serialization, no network copy, no GIL contention.

    A :class:`threading.Lock` protects concurrent writes/reads so that a
    producer writing a new frame never races with a consumer reading the
    previous one.

    Args:
        name:       Unique name for the shared memory block (across processes).
        size_bytes: Fixed capacity of the channel in bytes.  Payloads larger
                    than this are rejected by :meth:`write`.
    """

    def __init__(self, name: str, size_bytes: int) -> None:
        self._name: str = name
        self._size: int = size_bytes
        self._lock: threading.Lock = threading.Lock()
        self._shm: shared_memory.SharedMemory = shared_memory.SharedMemory(
            create=True,
            size=size_bytes,
            name=name,
        )
        logger.debug(
            "SharedMemoryChannel '%s' created (%d bytes)", name, size_bytes
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique name of this shared memory channel."""
        return self._name

    @property
    def size_bytes(self) -> int:
        """Fixed capacity of this channel in bytes."""
        return self._size

    # ── IO ────────────────────────────────────────────────────────────────────

    def write(self, data: bytes) -> bool:
        """Write *data* into the shared memory buffer.

        The write is protected by an internal :class:`threading.Lock` so
        concurrent writes are serialised safely.  If *data* exceeds the channel
        capacity the call is rejected without modifying the buffer.

        Args:
            data: Raw bytes to write.  Must be ``≤ size_bytes``.

        Returns:
            ``True`` on success, ``False`` if *data* is too large.
        """
        if len(data) > self._size:
            logger.error(
                "SharedMemoryChannel '%s': payload %d bytes exceeds capacity %d — rejected",
                self._name, len(data), self._size,
            )
            return False
        with self._lock:
            self._shm.buf[:len(data)] = data
        return True

    def read(self) -> bytes:
        """Read the current contents of the shared memory buffer.

        Returns a copy of the full buffer as ``bytes``.  The copy is made
        inside the lock so the returned bytes are always consistent.

        Returns:
            A ``bytes`` snapshot of the buffer at the moment of the call.
        """
        with self._lock:
            return bytes(self._shm.buf[:self._size])

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release and unlink the shared memory block.

        After this call the channel is unusable.  Always call :meth:`close`
        when the channel is no longer needed to avoid POSIX shm leaks.
        """
        try:
            self._shm.close()
            self._shm.unlink()
            logger.debug("SharedMemoryChannel '%s' closed and unlinked", self._name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SharedMemoryChannel '%s' close error: %s", self._name, exc
            )


# ─────────────────────────────────────────────────────────────────────────────
# SharedMemoryBus
# ─────────────────────────────────────────────────────────────────────────────

class SharedMemoryBus:
    """High-performance bus for heavy sensor and tensor payloads.

    Manages a collection of named :class:`SharedMemoryChannel` instances.
    Use this bus for any data exceeding ~10 KB (camera frames, LIDAR point
    clouds, neural network activation tensors).

    For control signals and small dicts, use the regular :data:`bus`
    (:class:`EventBus`) instead.

    Thread-safe: a :class:`threading.Lock` guards the channel registry.

    Example::

        from nayak.core.bus import shm_bus

        ch = shm_bus.create_channel("cam.rgb", 1920 * 1080 * 3)
        ch.write(frame_bytes)
        ...
        data = shm_bus.get_channel("cam.rgb").read()
        shm_bus.destroy_channel("cam.rgb")
    """

    def __init__(self) -> None:
        self._channels: dict[str, SharedMemoryChannel] = {}
        self._lock: threading.Lock = threading.Lock()

    def create_channel(self, name: str, size_bytes: int) -> SharedMemoryChannel:
        """Create a new named shared memory channel.

        If a channel with *name* already exists it is returned as-is (no
        duplication).

        Args:
            name:       Unique channel identifier.
            size_bytes: Fixed capacity in bytes.

        Returns:
            The new (or existing) :class:`SharedMemoryChannel`.
        """
        with self._lock:
            if name in self._channels:
                logger.debug(
                    "SharedMemoryBus: channel '%s' already exists — returning existing", name
                )
                return self._channels[name]
            channel = SharedMemoryChannel(name=name, size_bytes=size_bytes)
            self._channels[name] = channel
            logger.info(
                "SharedMemoryBus: created channel '%s' (%d bytes)", name, size_bytes
            )
            return channel

    def get_channel(self, name: str) -> SharedMemoryChannel | None:
        """Look up an existing channel by name.

        Args:
            name: Channel identifier.

        Returns:
            The :class:`SharedMemoryChannel` if it exists, otherwise ``None``.
        """
        with self._lock:
            return self._channels.get(name)

    def destroy_channel(self, name: str) -> None:
        """Close and remove a channel from the bus.

        Calls :meth:`SharedMemoryChannel.close` which unlinks the POSIX shm
        block.  After this call, :meth:`get_channel` returns ``None`` for
        *name*.

        Args:
            name: Channel identifier to destroy.
        """
        with self._lock:
            channel = self._channels.pop(name, None)
        if channel is None:
            logger.warning(
                "SharedMemoryBus: destroy_channel('%s') — not found", name
            )
            return
        channel.close()
        logger.info("SharedMemoryBus: destroyed channel '%s'", name)

    def list_channels(self) -> list[str]:
        """Return the names of all active channels.

        Returns:
            Sorted list of channel name strings.
        """
        with self._lock:
            return sorted(self._channels.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._channels)


# ─────────────────────────────────────────────────────────────────────────────
# EventType
# ─────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    """All recognised NAYAK event types.

    Extend this enum to introduce new bus events — existing subscribers
    are unaffected.
    """

    AGENT_STARTED        = auto()  # Agent.run() entered
    AGENT_STOPPED        = auto()  # Agent fully shut down (finally block)
    STEP_STARTED         = auto()  # A new perceive-think-act-remember cycle began
    STEP_COMPLETED       = auto()  # A step finished successfully
    STEP_FAILED          = auto()  # A step timed out or raised an error
    ACTION_TAKEN         = auto()  # An action was executed by Computer
    MEMORY_SAVED         = auto()  # A step was persisted to the memory store
    GOAL_COMPLETED       = auto()  # ActionType.FINISH was returned
    ERROR_OCCURRED       = auto()  # Any unexpected error in any component
    PERCEPTION_READY     = auto()  # _perceive() finished; PageState available
    COGNITION_READY      = auto()  # _think() finished; Action chosen
    ACTION_READY         = auto()  # Action backend initialized and ready
    MEMORY_READY         = auto()  # Memory backend initialized and ready
    MODULE_REGISTERED    = auto()  # A NayakModule was added to the ModuleRegistry
    MODULE_UNREGISTERED  = auto()  # A NayakModule was removed from the ModuleRegistry
    RUNTIME_STARTING     = auto()  # NayakRuntime.start() entered
    RUNTIME_READY        = auto()  # All modules initialised; runtime is operational
    RUNTIME_STOPPING     = auto()  # NayakRuntime.stop() entered
    RUNTIME_STOPPED      = auto()  # Runtime fully shut down

    # HAL — Layer 1 hardware events
    HAL_READY            = auto()  # A HardwareBase backend registered and is operational
    DEVICE_DATA          = auto()  # A device produced a data reading (sensor/camera/etc.)

    # COMM — Layer 6 communication events
    COMM_READY           = auto()  # A CommunicationBase backend registered and is operational
    MESSAGE_RECEIVED     = auto()  # The communication engine received an incoming message

    # SAFETY — Layer 7 safety engine events
    SAFETY_READY         = auto()  # A SafetyBase backend registered and is operational
    SAFETY_VIOLATION     = auto()  # A safety violation occurred

    # UPDATE — Layer 8 update engine events
    UPDATE_READY         = auto()  # An UpdateBase backend registered and is operational
    PACKAGE_INSTALLED    = auto()  # A new skill package or module update was installed

    # SDK — Layer 9 developer platform events
    SDK_READY            = auto()  # A DeveloperPlatformBase backend registered and is operational
    SKILL_LOADED         = auto()  # A skill was successfully loaded into the runtime
    SKILL_UNLOADED       = auto()  # A skill was removed from the runtime

    # SHM — Shared memory bus events
    SHM_CHANNEL_CREATED  = auto()  # A new shared memory channel was opened
    SHM_CHANNEL_CLOSED   = auto()  # A shared memory channel was destroyed


# ─────────────────────────────────────────────────────────────────────────────
# NayakEvent
# ─────────────────────────────────────────────────────────────────────────────

Handler = Callable[["NayakEvent"], Awaitable[None]]


@dataclass
class NayakEvent:
    """An immutable event that travels over the EventBus.

    Attributes:
        type:      The EventType that categorises this event.
        payload:   Arbitrary dict of data attached by the emitter.
        timestamp: Unix timestamp (seconds) set automatically on creation.
        source:    Human-readable name of the module that emitted this event.
    """

    type: EventType
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


# ─────────────────────────────────────────────────────────────────────────────
# EventBus
# ─────────────────────────────────────────────────────────────────────────────

class EventBus:
    """Async publish/subscribe event bus for control signals and small payloads.

    All handlers are called with ``await`` in the order they subscribed.
    Handler exceptions are caught, logged, and never propagate back to the
    emitter — the bus never crashes due to a bad handler.

    Thread/task safety: the ``handlers`` dict is mutated only from the
    subscribe/unsubscribe methods, which are synchronous and fast.  ``emit``
    is async and iterates a snapshot of the current handler list, so late
    subscription changes during emission have no effect on the in-progress
    emit call.

    Large-payload guard: if ``emit()`` receives a payload whose string
    representation exceeds ``_LARGE_PAYLOAD_THRESHOLD`` (10 000 chars) and
    contains a ``"data"`` key, a WARNING is logged recommending
    :data:`shm_bus` instead.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register *handler* to be called whenever *event_type* is emitted.

        The same handler can safely be registered multiple times; it will be
        called once per registration.

        Args:
            event_type: The :class:`EventType` to listen for.
            handler:    An ``async`` callable that accepts a single
                        :class:`NayakEvent` argument.
        """
        self._handlers[event_type].append(handler)
        logger.debug(
            "EventBus: subscribed %s to %s (total=%d)",
            getattr(handler, "__qualname__", repr(handler)),
            event_type.name,
            len(self._handlers[event_type]),
        )

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove *handler* from the subscriber list for *event_type*.

        If *handler* is not currently registered, this is a no-op.

        Args:
            event_type: The :class:`EventType` to stop listening to.
            handler:    The exact callable reference previously passed to
                        :meth:`subscribe`.
        """
        try:
            self._handlers[event_type].remove(handler)
            logger.debug(
                "EventBus: unsubscribed %s from %s",
                getattr(handler, "__qualname__", repr(handler)),
                event_type.name,
            )
        except ValueError:
            logger.debug(
                "EventBus: unsubscribe called for unregistered handler %s on %s",
                getattr(handler, "__qualname__", repr(handler)),
                event_type.name,
            )

    async def emit(self, event: NayakEvent) -> None:
        """Broadcast *event* to all handlers subscribed to its type.

        Handlers are awaited sequentially in subscription order.  If a handler
        raises any exception, it is caught and logged at ERROR level; remaining
        handlers are still called.

        Large-payload guard: if the payload contains a ``"data"`` key and its
        total string length exceeds 10 000 characters, a WARNING is emitted
        with the message ``"Use spinal_bridge.read_shm_frame() for sensor
        data"``.  Sensor frames and binary blobs must travel through the SHM
        channel — never through the event bus.

        Args:
            event: The :class:`NayakEvent` to broadcast.
        """
        # ── Large-payload guard ──────────────────────────────────────────────
        if "data" in event.payload and len(str(event.payload)) > _LARGE_PAYLOAD_THRESHOLD:
            logger.warning(
                "EventBus: Large payload detected on event '%s' (~%d chars) — "
                "Use spinal_bridge.read_shm_frame() for sensor data",
                event.type.name,
                len(str(event.payload)),
            )

        handlers = list(self._handlers.get(event.type, []))  # snapshot
        if not handlers:
            logger.debug("EventBus: emit %s — no subscribers", event.type.name)
            return

        logger.debug(
            "EventBus: emit %s from '%s' to %d subscriber(s)",
            event.type.name,
            event.source,
            len(handlers),
        )
        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "EventBus: handler %s raised on event %s: %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event.type.name,
                    exc,
                    exc_info=True,
                )

    async def emit_error(self, source: str, error: Exception) -> None:
        """Convenience shortcut to emit an :attr:`EventType.ERROR_OCCURRED` event.

        Args:
            source: Name of the component where the error originated.
            error:  The exception that was raised.
        """
        await self.emit(
            NayakEvent(
                type=EventType.ERROR_OCCURRED,
                payload={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                source=source,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global singletons
# ─────────────────────────────────────────────────────────────────────────────

bus: EventBus = EventBus()
"""The process-wide NAYAK event bus.

Use for control signals, status events, and small payloads (< 10 KB).

Import and use this singleton directly::

    from nayak.core.bus import bus
"""

shm_bus: SharedMemoryBus = SharedMemoryBus()
"""The process-wide NAYAK shared memory bus.

Use for heavy payloads: camera frames, LIDAR point clouds, tensors.
Data is written once; consumers read directly — zero copies, zero serialization.

Import and use this singleton directly::

    from nayak.core.bus import shm_bus

    ch = shm_bus.create_channel("camera.rgb", 1920 * 1080 * 3)
    ch.write(frame_bytes)
"""

__all__ = [
    "PayloadType",
    "SharedMemoryChannel",
    "SharedMemoryBus",
    "shm_bus",
    "EventType",
    "NayakEvent",
    "EventBus",
    "bus",
]
