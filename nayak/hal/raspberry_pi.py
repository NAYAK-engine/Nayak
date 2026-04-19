"""
nayak/hal/raspberry_pi.py — NAYAK HAL: Raspberry Pi hardware backend.

Provides a concrete :class:`HardwareBase` implementation targeting the
Raspberry Pi GPIO ecosystem.  When ``RPi.GPIO`` is not available (i.e. the
process is running on a development machine rather than actual Pi hardware)
the backend transparently falls back to *simulated mode* — every operation
succeeds and returns deterministic placeholder data.  This lets the full
NAYAK stack run, test, and integrate on any machine without physical hardware.

Usage::

    from nayak.hal.raspberry_pi import raspberry_pi

    await raspberry_pi.init()

    ok = await raspberry_pi.connect("sensor_0")
    data = await raspberry_pi.read("sensor_0")
    await raspberry_pi.stop()

The module exposes a process-wide singleton :data:`raspberry_pi` so that all
NAYAK layers share exactly one GPIO resource manager.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nayak.core.registry import ModuleStatus, registry
from nayak.hal.base import DeviceInfo, DeviceStatus, DeviceType, HardwareBase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RaspberryPiHAL
# ─────────────────────────────────────────────────────────────────────────────

class RaspberryPiHAL(HardwareBase):
    """Raspberry Pi hardware backend for the NAYAK Hardware Abstraction Layer.

    This backend abstracts GPIO, sensor buses (I²C, SPI, UART), and camera
    interfaces available on Raspberry Pi hardware.  On non-Pi hosts the backend
    enters *simulated mode* automatically: all reads return structured dummy
    payloads and writes are logged but otherwise no-op.  Simulated mode is
    indicated by ``DeviceInfo.metadata["simulated"] == True``.

    Simulated mode is designed to be fully deterministic and stable so that
    integration tests, CI pipelines, and developer machines can exercise the
    entire NAYAK software stack without requiring physical hardware.

    Attributes:
        devices (dict[str, DeviceInfo]):
            Registry of every device managed by this backend, keyed by
            ``device_id``.  Populated on :meth:`connect` and cleared on
            :meth:`stop`.
    """

    # ── HardwareBase contract ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Dot-namespaced registry identifier for this backend.

        Returns:
            ``"raspberry-pi-hal"``
        """
        return "raspberry-pi-hal"

    # ── Core async interface ──────────────────────────────────────────────────

    async def connect(self, device_id: str) -> bool:
        """Connect to the device identified by *device_id*.

        If *device_id* is not yet tracked, a new :class:`~nayak.hal.base.DeviceInfo`
        entry is created with a generic type of :attr:`~nayak.hal.base.DeviceType.GENERIC`.
        The method then attempts to import ``RPi.GPIO``:

        * **On a real Raspberry Pi** — GPIO is available; the device is
          registered as ``CONNECTED`` with real hardware metadata.
        * **On any other machine** — GPIO import fails; the device is still
          registered as ``CONNECTED`` with ``metadata["simulated"] = True``
          so that the rest of the stack can operate normally.

        Args:
            device_id: Unique identifier for the device to connect
                       (e.g. ``"gpio_17"``, ``"camera_0"``, ``"imu_main"``).

        Returns:
            ``True`` on success (including simulated mode), ``False`` if an
            unexpected error prevents registration.
        """
        try:
            # ── Ensure a DeviceInfo entry exists ──────────────────────────────
            if device_id not in self.devices:
                self.devices[device_id] = DeviceInfo(
                    device_id=device_id,
                    device_type=DeviceType.GENERIC,
                    name=f"Pi Device [{device_id}]",
                    status=DeviceStatus.CONNECTING,
                )
            else:
                self.devices[device_id].status = DeviceStatus.CONNECTING

            device = self.devices[device_id]

            # ── Attempt real GPIO import ──────────────────────────────────────
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]  # noqa: F401
                device.status = DeviceStatus.CONNECTED
                device.metadata.setdefault("simulated", False)
                device.metadata["gpio_available"] = True
                logger.info(
                    "RaspberryPiHAL: '%s' connected via RPi.GPIO (real hardware)",
                    device_id,
                )
            except ImportError:
                # Not on a Pi — engage simulated mode transparently.
                device.status = DeviceStatus.CONNECTED
                device.metadata["simulated"] = True
                device.metadata["gpio_available"] = False
                logger.info(
                    "RaspberryPiHAL: '%s' connected in simulated mode "
                    "(RPi.GPIO not available)",
                    device_id,
                )

            device.connected_at = time.time()
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RaspberryPiHAL: unexpected error connecting '%s': %s",
                device_id, exc, exc_info=True,
            )
            if device_id in self.devices:
                self.devices[device_id].status = DeviceStatus.ERROR
            return False

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect the device identified by *device_id*.

        Updates the device's status to :attr:`~nayak.hal.base.DeviceStatus.DISCONNECTED`.
        If *device_id* is not tracked this is a safe no-op.

        Args:
            device_id: Unique identifier for the device to disconnect.

        Returns:
            Always ``True`` (disconnection is best-effort).
        """
        try:
            device = self.devices.get(device_id)
            if device is None:
                logger.warning(
                    "RaspberryPiHAL: disconnect called for unknown device '%s'",
                    device_id,
                )
                return True

            device.status = DeviceStatus.DISCONNECTED
            logger.info(
                "RaspberryPiHAL: '%s' disconnected", device_id
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RaspberryPiHAL: error disconnecting '%s': %s",
                device_id, exc, exc_info=True,
            )
            return True  # Disconnection failures should never block shutdown.

    async def read(self, device_id: str) -> Any:
        """Read the latest data from *device_id*.

        If the device is not in a connected/active state ``None`` is returned
        immediately.  In simulated mode the return value is a deterministic
        dict shaped to match the device's :class:`~nayak.hal.base.DeviceType`:

        +-----------------+-------------------------------------------------------+
        | DeviceType      | Simulated payload                                     |
        +=================+=======================================================+
        | CAMERA          | ``{"frame": "simulated_frame", "resolution": "640x480"}`` |
        +-----------------+-------------------------------------------------------+
        | SENSOR          | ``{"value": 0.0, "unit": "unknown"}``                 |
        +-----------------+-------------------------------------------------------+
        | MOTOR           | ``{"speed": 0, "direction": "stopped"}``              |
        +-----------------+-------------------------------------------------------+
        | IMU             | ``{"ax": 0.0, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0}`` |
        +-----------------+-------------------------------------------------------+
        | *anything else* | ``{"raw": None}``                                     |
        +-----------------+-------------------------------------------------------+

        A :attr:`~nayak.core.bus.EventType.DEVICE_DATA` bus event is emitted
        after a successful read so that subscribers (perception, logging) are
        notified without polling.

        Args:
            device_id: Unique identifier of the device to read from.

        Returns:
            A device-specific data dict, or ``None`` if the device is not
            connected / an error occurred.
        """
        try:
            device = self.devices.get(device_id)
            if device is None or device.status not in (
                DeviceStatus.CONNECTED,
                DeviceStatus.ACTIVE,
            ):
                logger.debug(
                    "RaspberryPiHAL: read skipped for '%s' — not connected "
                    "(status=%s)",
                    device_id,
                    device.status.name if device else "UNKNOWN",
                )
                return None

            # ── Simulated data path ───────────────────────────────────────────
            if device.metadata.get("simulated", False):
                data = self._simulated_payload(device)
            else:
                # Real GPIO read would happen here.
                # Subclasses or future revisions extend this branch.
                data = {"raw": None}

            await self.emit_device_event(device_id, data)
            return data

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RaspberryPiHAL: error reading '%s': %s",
                device_id, exc, exc_info=True,
            )
            return None

    async def write(self, device_id: str, data: Any) -> bool:
        """Send *data* to the device identified by *device_id*.

        In simulated mode the command is logged at DEBUG level and the method
        returns ``True`` immediately without any physical I/O.  On real hardware
        this is the extension point where GPIO writes, PWM duty-cycle updates,
        or serial commands would be dispatched.

        Args:
            device_id: Unique identifier of the target device.
            data:      Device-specific command payload (e.g. motor speed ``int``,
                       GPIO pin state ``bool``, or an arbitrary ``dict``).

        Returns:
            ``True`` on success or in simulated mode, ``False`` if the device
            is not connected or an unexpected error occurred.
        """
        try:
            device = self.devices.get(device_id)
            if device is None or device.status not in (
                DeviceStatus.CONNECTED,
                DeviceStatus.ACTIVE,
            ):
                logger.warning(
                    "RaspberryPiHAL: write rejected for '%s' — not connected",
                    device_id,
                )
                return False

            if device.metadata.get("simulated", False):
                logger.debug(
                    "RaspberryPiHAL [sim]: write to '%s' — data=%r",
                    device_id, data,
                )
                return True

            # Real GPIO write would dispatch here.
            logger.debug(
                "RaspberryPiHAL: write to '%s' — data=%r",
                device_id, data,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RaspberryPiHAL: error writing to '%s': %s",
                device_id, exc, exc_info=True,
            )
            return False

    async def list_devices(self) -> list[DeviceInfo]:
        """Return all devices currently tracked by this backend.

        Returns:
            A snapshot list of :class:`~nayak.hal.base.DeviceInfo` objects for
            every device that has been connected via :meth:`connect`.
        """
        return list(self.devices.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Initialise the Raspberry Pi HAL backend and register it with NAYAK.

        Calls :meth:`~nayak.hal.base.HardwareBase.register` to publish this
        backend to the module registry and emit the
        :attr:`~nayak.core.bus.EventType.HAL_READY` bus event.

        This method **must** be awaited before any :meth:`connect` / :meth:`read`
        / :meth:`write` calls.

        Example::

            from nayak.hal.raspberry_pi import raspberry_pi
            await raspberry_pi.init()
        """
        await self.register()
        logger.info("RaspberryPiHAL: Raspberry Pi HAL initialized")

    async def stop(self) -> None:
        """Gracefully shutdown the backend.

        Disconnects every tracked device in order to release hardware resources,
        then promotes the module status to
        :attr:`~nayak.core.registry.ModuleStatus.STOPPED` in the registry.

        Safe to call even if no devices have been connected.

        Example::

            await raspberry_pi.stop()
        """
        device_ids = list(self.devices.keys())
        for device_id in device_ids:
            await self.disconnect(device_id)

        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("RaspberryPiHAL: stopped and all devices disconnected")
        except KeyError:
            # Backend was never fully registered (e.g. stop() before init()).
            logger.debug(
                "RaspberryPiHAL: stop() called before registration — "
                "skipping registry status update"
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _simulated_payload(device: DeviceInfo) -> dict[str, Any]:
        """Return a deterministic simulated data payload for *device*.

        The payload shape is chosen based on :attr:`~nayak.hal.base.DeviceInfo.device_type`
        so that downstream consumers receive structurally valid data even in
        simulated mode.

        Args:
            device: The :class:`~nayak.hal.base.DeviceInfo` whose type drives
                    the payload shape.

        Returns:
            A ``dict`` whose keys match the schema expected by real hardware of
            the same :class:`~nayak.hal.base.DeviceType`.
        """
        match device.device_type:
            case DeviceType.CAMERA:
                return {"frame": "simulated_frame", "resolution": "640x480"}
            case DeviceType.SENSOR:
                return {"value": 0.0, "unit": "unknown"}
            case DeviceType.MOTOR:
                return {"speed": 0, "direction": "stopped"}
            case DeviceType.IMU:
                return {
                    "ax": 0.0, "ay": 0.0, "az": 9.8,  # gravity on z-axis
                    "gx": 0.0, "gy": 0.0, "gz": 0.0,
                }
            case _:
                return {"raw": None}


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

raspberry_pi: RaspberryPiHAL = RaspberryPiHAL()
"""Process-wide Raspberry Pi HAL singleton.

Import and use directly — no object instantiation required::

    from nayak.hal.raspberry_pi import raspberry_pi

    await raspberry_pi.init()
    await raspberry_pi.connect("imu_main")
    data = await raspberry_pi.read("imu_main")
    await raspberry_pi.stop()
"""

__all__ = ["RaspberryPiHAL", "raspberry_pi"]
