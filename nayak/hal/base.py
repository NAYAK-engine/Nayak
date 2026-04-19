"""
nayak/hal/base.py — NAYAK Hardware Abstraction Layer: core abstractions.

Defines the canonical data types and the abstract base class that every
hardware backend in NAYAK must implement.

Concrete backends (e.g. a ROS2 motor adapter, a V4L2 camera driver, a
simulated IMU) subclass :class:`HardwareBase`, implement the abstract async
interface, and call ``await self.register()`` once ready.  From that point
on the rest of the NAYAK stack can interact with any hardware device through
the common interface without knowing which physical backend is in use.

Usage::

    from nayak.hal.base import HardwareBase, DeviceType, DeviceStatus, DeviceInfo

    class MyMotorBackend(HardwareBase):
        name = "hal.motor.acme"

        async def connect(self, device_id: str) -> bool: ...
        async def disconnect(self, device_id: str) -> bool: ...
        async def read(self, device_id: str) -> Any: ...
        async def write(self, device_id: str, data: Any) -> bool: ...
        async def list_devices(self) -> list[DeviceInfo]: ...

    backend = MyMotorBackend()
    await backend.register()
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, NayakModule, registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DeviceType
# ─────────────────────────────────────────────────────────────────────────────

class DeviceType(Enum):
    """The broad functional category of a hardware device.

    Use this enum to tag :class:`DeviceInfo` entries so that higher-level
    NAYAK layers can filter or route device interactions by capability class.

    Members:
        CAMERA:     Image or video capture devices (USB, CSI, IP cameras).
        MOTOR:      Rotary actuators — servos, DC motors, stepper motors.
        SENSOR:     Generic transducers — temperature, pressure, proximity, etc.
        ARM:        Articulated robotic arm assemblies.
        LEG:        Articulated leg or walking-mechanism assemblies.
        WHEEL:      Driven or passive wheel units on a mobile platform.
        MICROPHONE: Audio input devices.
        SPEAKER:    Audio output devices.
        LIDAR:      Light-detection-and-ranging distance sensors.
        GPS:        Global-positioning / GNSS receivers.
        IMU:        Inertial measurement units (accelerometer + gyroscope ±\u00a0mag).
        GENERIC:    Any device that does not fit an established category.
    """

    CAMERA     = auto()
    MOTOR      = auto()
    SENSOR     = auto()
    ARM        = auto()
    LEG        = auto()
    WHEEL      = auto()
    MICROPHONE = auto()
    SPEAKER    = auto()
    LIDAR      = auto()
    GPS        = auto()
    IMU        = auto()
    GENERIC    = auto()


# ─────────────────────────────────────────────────────────────────────────────
# DeviceStatus
# ─────────────────────────────────────────────────────────────────────────────

class DeviceStatus(Enum):
    """Lifecycle status of an individual hardware device.

    A typical happy-path transition is::

        DISCONNECTED → CONNECTING → CONNECTED → ACTIVE

    From any state the device may fall into ERROR or be explicitly DISABLED.

    Members:
        DISCONNECTED: Device is known but no active connection exists.
        CONNECTING:   A connection attempt is in progress.
        CONNECTED:    Transport-layer link established; device not yet streaming.
        ACTIVE:       Device is fully operational and producing / accepting data.
        ERROR:        Device has encountered a fault and requires attention.
        DISABLED:     Device has been intentionally taken offline by the operator.
    """

    DISCONNECTED = auto()
    CONNECTING   = auto()
    CONNECTED    = auto()
    ACTIVE       = auto()
    ERROR        = auto()
    DISABLED     = auto()


# ─────────────────────────────────────────────────────────────────────────────
# DeviceInfo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceInfo:
    """Descriptor for a single hardware device managed by a HAL backend.

    :class:`HardwareBase` implementations store one :class:`DeviceInfo` per
    device in their ``devices`` dict, keyed by ``device_id``.

    Attributes:
        device_id:    Unique identifier for this device within the backend
                      (e.g. ``"/dev/video0"``, ``"motor_left_hip"``).
        device_type:  The functional category of this device (:class:`DeviceType`).
        name:         A human-readable label (e.g. ``"Front-facing USB Camera"``).
        status:       Current lifecycle state (:class:`DeviceStatus`).
                      Defaults to ``DISCONNECTED``.
        metadata:     Arbitrary key/value pairs for vendor-specific information
                      (firmware version, serial number, calibration data, …).
                      Defaults to an empty dict.
        connected_at: Unix timestamp (seconds) of the last successful connection.
                      ``0.0`` indicates the device has never been connected.
    """

    device_id:    str
    device_type:  DeviceType
    name:         str
    status:       DeviceStatus = DeviceStatus.DISCONNECTED
    metadata:     dict         = field(default_factory=dict)
    connected_at: float        = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HardwareBase
# ─────────────────────────────────────────────────────────────────────────────

class HardwareBase(abc.ABC):
    """Abstract base class for all NAYAK Layer-1 hardware backends.

    Every concrete hardware driver (motor controller, camera adapter, sensor
    hub, …) must subclass :class:`HardwareBase` and implement the five
    abstract async methods that form the canonical HAL contract.

    The class handles self-registration with the NAYAK module registry and
    provides convenience helpers for device lookup and event emission so that
    backends only need to focus on their hardware-specific logic.

    Class Attributes:
        layer (int):                Always ``1`` — HAL is always NAYAK Layer 1.
        devices (dict[str, DeviceInfo]):
            Mutable mapping of ``device_id → DeviceInfo`` maintained by the
            concrete subclass.  Populated during :meth:`connect` and cleared
            (or updated) during :meth:`disconnect`.

    Abstract Properties:
        name (str): Dot-namespaced module identifier used in the registry,
                    e.g. ``"hal.motor.acme"`` or ``"hal.camera.realsense"``.
    """

    # ── Class-level constants ─────────────────────────────────────────────────

    layer: int = 1
    """NAYAK OS layer. Always 1 for HAL backends."""

    # Instance-level device registry — each subclass gets its own dict via
    # __init_subclass__ so that sibling subclasses never share state.
    devices: dict[str, DeviceInfo]

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        # Give every concrete subclass its own independent devices dict so
        # class-level mutations on one backend never bleed into another.
        if "devices" not in cls.__dict__:
            cls.devices = {}

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Dot-namespaced registry identifier for this backend.

        Returns:
            A unique string such as ``"hal.motor.acme"`` that the NAYAK module
            registry will use as the primary lookup key.
        """

    @abc.abstractmethod
    async def connect(self, device_id: str) -> bool:
        """Establish a connection to the device identified by *device_id*.

        Implementations should update ``self.devices[device_id].status`` to
        :attr:`DeviceStatus.CONNECTED` (or :attr:`DeviceStatus.ERROR`) and
        record ``self.devices[device_id].connected_at = time.time()`` on
        success.

        Args:
            device_id: The unique identifier of the device to connect to.

        Returns:
            ``True`` if the connection was established successfully, ``False``
            otherwise.
        """

    @abc.abstractmethod
    async def disconnect(self, device_id: str) -> bool:
        """Tear down the connection to *device_id*.

        Implementations should update the device's status to
        :attr:`DeviceStatus.DISCONNECTED` upon success.

        Args:
            device_id: The unique identifier of the device to disconnect.

        Returns:
            ``True`` if the device was disconnected cleanly, ``False`` on error.
        """

    @abc.abstractmethod
    async def read(self, device_id: str) -> Any:
        """Read the latest data from *device_id*.

        The shape of the returned value is device-type-specific (e.g. a
        ``numpy`` array for cameras, a ``float`` for a single-axis sensor).
        Callers should check :attr:`DeviceInfo.device_type` to interpret the
        return value correctly.

        Args:
            device_id: The unique identifier of the device to read from.

        Returns:
            Device-specific data, or ``None`` if no data is available.
        """

    @abc.abstractmethod
    async def write(self, device_id: str, data: Any) -> bool:
        """Send *data* to *device_id* (e.g. a motor speed command).

        Args:
            device_id: The unique identifier of the device to write to.
            data:      Device-specific command payload.

        Returns:
            ``True`` if the write was acknowledged, ``False`` on error.
        """

    @abc.abstractmethod
    async def list_devices(self) -> list[DeviceInfo]:
        """Return all devices currently managed by this backend.

        Implementations may perform a live hardware scan or simply return the
        current contents of ``self.devices``.

        Returns:
            A list of :class:`DeviceInfo` descriptors, one per managed device.
        """

    # ── Concrete methods ──────────────────────────────────────────────────────

    async def register(self) -> None:
        """Register this backend with the NAYAK module registry.

        Creates a :class:`~nayak.core.registry.NayakModule` descriptor for
        this backend and submits it to the global :data:`~nayak.core.registry.registry`.
        After registration the module status is promoted to
        :attr:`~nayak.core.registry.ModuleStatus.READY` and a
        :attr:`~nayak.core.bus.EventType.HAL_READY` event is emitted on the
        global :data:`~nayak.core.bus.bus`.

        This method is idempotent — calling it more than once replaces the
        existing registry entry (the registry logs a warning).

        Example::

            backend = MyMotorBackend()
            await backend.register()
        """
        module = NayakModule(
            name=self.name,
            version="0.2.0",
            layer=self.layer,
            description="Hardware abstraction backend",
            instance=self,
        )
        await registry.register(module)
        registry.set_status(self.name, ModuleStatus.READY)

        logger.info("HardwareBase: '%s' registered and READY", self.name)

        await bus.emit(NayakEvent(
            type=EventType.HAL_READY,
            payload={"module": self.name, "layer": self.layer},
            source=self.name,
        ))

    def get_device(self, device_id: str) -> DeviceInfo | None:
        """Look up a device by its identifier.

        Args:
            device_id: The unique identifier of the device to find.

        Returns:
            The :class:`DeviceInfo` for *device_id* if it is tracked by this
            backend, or ``None`` if it is unknown.
        """
        return self.devices.get(device_id)

    async def emit_device_event(self, device_id: str, data: Any) -> None:
        """Broadcast a :attr:`~nayak.core.bus.EventType.DEVICE_DATA` event.

        Convenience helper for backends to publish a new reading from a device
        to any NAYAK subscriber (e.g. a perception layer or a logging service)
        without coupling the hardware driver to its consumers.

        Args:
            device_id: The unique identifier of the source device.
            data:      The device reading to broadcast.  The shape is entirely
                       device-specific; consumers must inspect
                       :attr:`DeviceInfo.device_type` to interpret it.

        Example::

            raw = await self._driver.poll()
            await self.emit_device_event("imu_0", raw)
        """
        await bus.emit(NayakEvent(
            type=EventType.DEVICE_DATA,
            payload={"device_id": device_id, "data": data},
            source=self.name,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "DeviceType",
    "DeviceStatus",
    "DeviceInfo",
    "HardwareBase",
]
