"""
nayak/hal/raspberry_pi.py — NAYAK HAL: Raspberry Pi hardware backend.

Provides a concrete :class:`HardwareBase` implementation targeting the
Raspberry Pi GPIO ecosystem.  Auto-detects real hardware at instantiation:

* **Real hardware mode** — ``RPi.GPIO`` imported successfully; GPIO reads
  sample physical BCM lines, writes drive real outputs.
* **Simulated mode** — ``RPi.GPIO`` not available; all operations succeed
  with deterministic placeholder data, zero errors.

The backend probes camera availability and generates a comprehensive
:meth:`hardware_report` at boot so operators always know exactly what
hardware is available.

Usage::

    from nayak.hal.raspberry_pi import raspberry_pi

    await raspberry_pi.init()

    ok = await raspberry_pi.connect("sensor_0")
    data = await raspberry_pi.read("sensor_0")
    report = raspberry_pi.hardware_report()
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
    payloads and writes are logged but otherwise no-op.

    At instantiation the class probes for:

    * ``RPi.GPIO`` — sets ``_on_pi`` and ``_gpio_available``.
    * Camera hardware — probes picamera2 then OpenCV to determine
      ``_camera_type`` (``"picamera2"`` | ``"opencv"`` | ``"none"``).

    The :meth:`hardware_report` method returns a dict summarising the detected
    hardware, and this report is logged at INFO level during :meth:`init`.

    Attributes:
        _on_pi (bool):
            ``True`` when running on a real Raspberry Pi with GPIO access.
        _gpio (module | None):
            Reference to the ``RPi.GPIO`` module, or ``None`` in simulation.
        _gpio_available (bool):
            Whether ``RPi.GPIO`` was imported successfully.
        _camera_type (str):
            Detected camera backend: ``"picamera2"``, ``"opencv"``, or
            ``"none"``.
        devices (dict[str, DeviceInfo]):
            Registry of every device managed by this backend, keyed by
            ``device_id``.  Populated on :meth:`connect` and cleared on
            :meth:`stop`.
    """

    def __init__(self) -> None:
        """Initialise the Raspberry Pi HAL backend and auto-detect hardware.

        Probes for ``RPi.GPIO`` and camera hardware.  This constructor
        **never raises** — hardware absence is a normal operating condition.
        """
        # ── GPIO auto-detection ───────────────────────────────────────────────
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            self._on_pi = True
            self._gpio = GPIO
            self._gpio_available = True
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            logger.info(
                "RaspberryPiHAL: RPi.GPIO detected — REAL hardware mode"
            )
        except (ImportError, RuntimeError):
            self._on_pi = False
            self._gpio = None
            self._gpio_available = False
            logger.info(
                "RaspberryPiHAL: RPi.GPIO not available — SIMULATION mode"
            )

        # ── Camera auto-detection ─────────────────────────────────────────────
        self._camera_type = self._detect_camera_type()

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
        The device is registered as ``CONNECTED`` with metadata reflecting
        the detected hardware mode.

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

            device.status = DeviceStatus.CONNECTED
            device.metadata["simulated"] = not self._on_pi
            device.metadata["gpio_available"] = self._gpio_available
            device.connected_at = time.time()

            logger.info(
                "RaspberryPiHAL: '%s' connected (mode=%s)",
                device_id,
                "real" if self._on_pi else "simulated",
            )
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
        immediately.

        **Real mode**: reads the physical GPIO pin value via ``RPi.GPIO``.
        **Simulated mode**: returns a deterministic dict shaped to match the
        device's :class:`~nayak.hal.base.DeviceType`.

        A :attr:`~nayak.core.bus.EventType.DEVICE_DATA` bus event is emitted
        after a successful read.

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

            # ── Real hardware path ────────────────────────────────────────────
            if self._on_pi and self._gpio is not None:
                pin = device.metadata.get("pin")
                if pin is not None:
                    try:
                        value = self._gpio.input(pin)
                        data: dict[str, Any] = {
                            "pin": pin,
                            "value": value,
                            "simulated": False,
                            "timestamp": time.time(),
                        }
                        await self.emit_device_event(device_id, data)
                        return data
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "RaspberryPiHAL: GPIO.input(%s) failed: %s",
                            pin, exc,
                        )
                        # Fall through to simulated payload.

            # ── Simulated data path ───────────────────────────────────────────
            data = self._simulated_payload(device)
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

        **Real mode**: if the data dict contains ``"pin"`` and ``"value"``
        keys, drives the physical GPIO line via ``RPi.GPIO``.
        **Simulated mode**: logs the command at DEBUG level and returns ``True``.

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

            # ── Real hardware path ────────────────────────────────────────────
            if self._on_pi and self._gpio is not None and isinstance(data, dict):
                pin = data.get("pin")
                value = data.get("value")
                if pin is not None and value is not None:
                    try:
                        self._gpio.setup(pin, self._gpio.OUT)
                        self._gpio.output(pin, value)
                        logger.debug(
                            "RaspberryPiHAL [REAL]: pin %s = %s", pin, value,
                        )
                        device.metadata["pin"] = pin
                        return True
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "RaspberryPiHAL: GPIO.output(%s, %s) failed: %s",
                            pin, value, exc,
                        )
                        return False

            # ── Simulated path ────────────────────────────────────────────────
            logger.debug(
                "RaspberryPiHAL [SIM]: write to '%s' — data=%r",
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
        backend to the module registry, logs the full :meth:`hardware_report`,
        and emits the :attr:`~nayak.core.bus.EventType.HAL_READY` bus event.

        This method **must** be awaited before any :meth:`connect` / :meth:`read`
        / :meth:`write` calls.

        Example::

            from nayak.hal.raspberry_pi import raspberry_pi
            await raspberry_pi.init()
        """
        await self.register()

        report = self.hardware_report()
        logger.info(
            "RaspberryPiHAL: ══════════════════════════════════════════"
        )
        logger.info(
            "RaspberryPiHAL: HARDWARE REPORT"
        )
        for key, value in report.items():
            logger.info(
                "RaspberryPiHAL:   %-20s = %s", key, value,
            )
        logger.info(
            "RaspberryPiHAL: ══════════════════════════════════════════"
        )

    async def stop(self) -> None:
        """Gracefully shutdown the backend.

        Disconnects every tracked device, cleans up GPIO resources, and
        promotes the module status to
        :attr:`~nayak.core.registry.ModuleStatus.STOPPED` in the registry.

        Safe to call even if no devices have been connected.

        Example::

            await raspberry_pi.stop()
        """
        device_ids = list(self.devices.keys())
        for device_id in device_ids:
            await self.disconnect(device_id)

        self.cleanup()

        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("RaspberryPiHAL: stopped and all devices disconnected")
        except KeyError:
            # Backend was never fully registered (e.g. stop() before init()).
            logger.debug(
                "RaspberryPiHAL: stop() called before registration — "
                "skipping registry status update"
            )

    # ── Hardware report ───────────────────────────────────────────────────────

    def hardware_report(self) -> dict[str, Any]:
        """Generate a comprehensive hardware status report.

        Returns a dict summarising the detected hardware environment.  This
        report is logged at INFO level during :meth:`init` so operators always
        know exactly what hardware is available at boot.

        Returns:
            A dict with the following keys:

            * ``"on_pi"`` (bool) — whether real Pi hardware was detected.
            * ``"gpio_available"`` (bool) — whether ``RPi.GPIO`` is importable.
            * ``"camera_type"`` (str) — ``"picamera2"``, ``"opencv"``, or ``"none"``.
            * ``"simulation_mode"`` (bool) — ``True`` if no real Pi hardware.
            * ``"device_count"`` (int) — number of currently tracked devices.
            * ``"timestamp"`` (float) — Unix timestamp of the report.

        Example::

            report = raspberry_pi.hardware_report()
            # {
            #     "on_pi": False,
            #     "gpio_available": False,
            #     "camera_type": "none",
            #     "simulation_mode": True,
            #     "device_count": 0,
            #     "timestamp": 1716300000.0,
            # }
        """
        return {
            "on_pi": self._on_pi,
            "gpio_available": self._gpio_available,
            "camera_type": self._camera_type,
            "simulation_mode": not self._on_pi,
            "device_count": len(self.devices),
            "timestamp": time.time(),
        }

    def cleanup(self) -> None:
        """Release all real GPIO resources.

        On a real Raspberry Pi this calls ``GPIO.cleanup()`` to reset all
        pin configurations.  In simulated mode this is a safe no-op.
        """
        if self._on_pi and self._gpio is not None:
            try:
                self._gpio.cleanup()
                logger.info("RaspberryPiHAL: GPIO.cleanup() completed")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RaspberryPiHAL: GPIO.cleanup() error: %s", exc,
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_camera_type() -> str:
        """Probe for available camera hardware.

        Tries to import ``picamera2`` (Pi CSI camera), then ``cv2`` (OpenCV).
        Returns the string identifier of the first successful import, or
        ``"none"`` if neither is available.

        Returns:
            ``"picamera2"``, ``"opencv"``, or ``"none"``.
        """
        try:
            from picamera2 import Picamera2  # type: ignore[import]  # noqa: F401
            return "picamera2"
        except (ImportError, RuntimeError, OSError):
            pass

        try:
            import cv2  # type: ignore[import]  # noqa: F401
            return "opencv"
        except ImportError:
            pass

        return "none"

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
    report = raspberry_pi.hardware_report()
    await raspberry_pi.stop()
"""

__all__ = ["RaspberryPiHAL", "raspberry_pi"]
