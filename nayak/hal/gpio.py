"""
nayak/hal/gpio.py — NAYAK HAL: GPIO hardware backend.

Provides a concrete :class:`HardwareBase` implementation dedicated to GPIO
pin control.  At instantiation the module probes for ``RPi.GPIO``:

* **Real hardware mode** — ``RPi.GPIO`` imported successfully; all reads and
  writes are dispatched to the physical BCM GPIO lines.
* **Simulated mode** — ``RPi.GPIO`` not available (e.g. running on a dev
  laptop); reads return values from an in-process shadow register and writes
  update that register with a debug log.  Zero errors, fully deterministic.

The auto-detection is **completely transparent** — callers never need to check
which mode is active.  The boot log always reports the detected mode so the
operator knows exactly what hardware is available.

Usage::

    from nayak.hal.gpio import gpio_hal

    await gpio_hal.init()

    ok = await gpio_hal.connect("gpio_17")
    await gpio_hal.write("gpio_17", {"pin": 17, "value": 1})
    data = await gpio_hal.read("gpio_17")
    # data["value"] == 1

    await gpio_hal.stop()

The module exposes a process-wide singleton :data:`gpio_hal` so that all
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
# GpioHAL
# ─────────────────────────────────────────────────────────────────────────────

class GpioHAL(HardwareBase):
    """GPIO hardware backend for the NAYAK Hardware Abstraction Layer.

    Wraps ``RPi.GPIO`` and exposes it through the standard HAL contract.
    Operates in two modes selected **automatically** at instantiation time:

    **Real hardware mode** (``RPi.GPIO`` importable and functional)
        :meth:`read` samples the physical BCM GPIO line.
        :meth:`write` drives the line high or low.
        :meth:`cleanup` calls ``GPIO.cleanup()`` to release all resources.

    **Simulated mode** (``RPi.GPIO`` not available)
        :meth:`read` returns the last value stored in the in-process shadow
        register (``self._pin_values``).  :meth:`write` updates the shadow
        register and logs the operation at DEBUG level.  :meth:`cleanup` is a
        safe no-op.

    Attributes:
        _on_pi (bool):
            ``True`` when running on a real Raspberry Pi with GPIO access.
        _gpio (module | None):
            Reference to the ``RPi.GPIO`` module, or ``None`` in simulation.
        _pin_values (dict[int, int]):
            Shadow register mapping BCM pin numbers to their last known
            logical values.  Used as the authoritative store in simulated mode
            and as a cache/mirror in real mode.
        _pin_directions (dict[int, str]):
            Tracks the configured direction (``"in"`` or ``"out"``) for each
            pin so that redundant ``GPIO.setup`` calls are avoided.
    """

    def __init__(self) -> None:
        """Initialise the GPIO backend and auto-detect real hardware.

        Attempts to import ``RPi.GPIO`` and configure BCM numbering.  If the
        import or setup fails (e.g. not running on a Pi, or insufficient
        permissions) the backend silently enters simulated mode.

        This constructor **never raises** — hardware absence is a normal
        operating condition, not an error.
        """
        self._pin_values: dict[int, int] = {}
        self._pin_directions: dict[int, str] = {}

        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            self._on_pi = True
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            logger.info(
                "GpioHAL: RPi.GPIO detected — running in REAL hardware mode"
            )
        except (ImportError, RuntimeError) as exc:
            self._on_pi = False
            self._gpio = None
            logger.info(
                "GpioHAL: no Pi detected (%s) — running in SIMULATION mode",
                type(exc).__name__,
            )

    # ── HardwareBase contract ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Dot-namespaced registry identifier for this backend.

        Returns:
            ``"gpio-hal"``
        """
        return "gpio-hal"

    # ── Core async interface ──────────────────────────────────────────────────

    async def connect(self, device_id: str) -> bool:
        """Register a GPIO device identified by *device_id*.

        Creates a :class:`~nayak.hal.base.DeviceInfo` entry with
        ``device_type=SENSOR`` and marks it as ``CONNECTED``.  The device
        metadata records whether the backend is operating in real or simulated
        mode.

        Args:
            device_id: Unique identifier for this GPIO device (e.g.
                       ``"gpio_17"``, ``"led_status"``).

        Returns:
            Always ``True`` — GPIO devices always connect (real or simulated).
        """
        try:
            self.devices[device_id] = DeviceInfo(
                device_id=device_id,
                device_type=DeviceType.SENSOR,
                name=f"GPIO [{device_id}]",
                status=DeviceStatus.CONNECTED,
                metadata={
                    "simulated": not self._on_pi,
                    "gpio_available": self._on_pi,
                },
            )
            self.devices[device_id].connected_at = time.time()
            logger.info(
                "GpioHAL: '%s' connected (mode=%s)",
                device_id,
                "real" if self._on_pi else "simulated",
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "GpioHAL: unexpected error connecting '%s': %s",
                device_id, exc, exc_info=True,
            )
            return False

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect the GPIO device identified by *device_id*.

        Updates the device status to ``DISCONNECTED``.  Safe to call even if
        *device_id* is not tracked.

        Args:
            device_id: Unique identifier of the device to disconnect.

        Returns:
            Always ``True`` — disconnection failures must never block shutdown.
        """
        try:
            device = self.devices.get(device_id)
            if device is None:
                logger.warning(
                    "GpioHAL: disconnect called for unknown device '%s'",
                    device_id,
                )
                return True

            device.status = DeviceStatus.DISCONNECTED
            logger.info("GpioHAL: '%s' disconnected", device_id)
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "GpioHAL: error disconnecting '%s': %s",
                device_id, exc, exc_info=True,
            )
            return True

    async def read(self, device_id: str) -> Any:
        """Read the current value of the GPIO pin associated with *device_id*.

        Extracts the BCM pin number from the device metadata (key ``"pin"``)
        or from *device_id* itself (e.g. ``"gpio_17"`` → pin 17).

        **Real mode**: calls ``GPIO.input(pin)`` and returns the physical
        line state.

        **Simulated mode**: returns the last value from the shadow register,
        defaulting to ``0`` for pins that have never been written.

        The returned dict always has the shape::

            {
                "pin":       <int>,
                "value":     <int>,   # 0 or 1
                "simulated": <bool>,
                "timestamp": <float>,
            }

        Args:
            device_id: Unique identifier of the GPIO device to read.

        Returns:
            A dict with pin state, or ``None`` if the device is not connected.
        """
        try:
            device = self.devices.get(device_id)
            if device is None or device.status not in (
                DeviceStatus.CONNECTED,
                DeviceStatus.ACTIVE,
            ):
                logger.debug(
                    "GpioHAL: read skipped for '%s' — not connected",
                    device_id,
                )
                return None

            pin = device.metadata.get("pin")
            if pin is None:
                pin = self._extract_pin_number(device_id)

            if self._on_pi and self._gpio is not None:
                # ── Real hardware path ────────────────────────────────────
                try:
                    self._ensure_pin_setup(pin, "in")
                    value = self._gpio.input(pin)
                    self._pin_values[pin] = value
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "GpioHAL: real GPIO.input(%d) failed: %s", pin, exc,
                    )
                    value = self._pin_values.get(pin, 0)
            else:
                # ── Simulated path ────────────────────────────────────────
                value = self._pin_values.get(pin, 0)

            data: dict[str, Any] = {
                "pin": pin,
                "value": value,
                "simulated": not self._on_pi,
                "timestamp": time.time(),
            }
            await self.emit_device_event(device_id, data)
            return data

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "GpioHAL: error reading '%s': %s",
                device_id, exc, exc_info=True,
            )
            return None

    async def write(self, device_id: str, data: Any) -> bool:
        """Write a value to the GPIO pin associated with *device_id*.

        Expects *data* to be a dict with keys ``"pin"`` (int) and ``"value"``
        (int, 0 or 1).  If ``"pin"`` is absent, the pin number is inferred
        from *device_id*.

        **Real mode**: configures the pin as OUTPUT (if not already) and calls
        ``GPIO.output(pin, value)``.

        **Simulated mode**: stores the value in the shadow register and logs
        the operation at DEBUG level.

        Args:
            device_id: Unique identifier of the target GPIO device.
            data:      Dict with ``"pin"`` and ``"value"`` keys.

        Returns:
            ``True`` on success, ``False`` if the device is not connected or
            an error occurred.
        """
        try:
            device = self.devices.get(device_id)
            if device is None or device.status not in (
                DeviceStatus.CONNECTED,
                DeviceStatus.ACTIVE,
            ):
                logger.warning(
                    "GpioHAL: write rejected for '%s' — not connected",
                    device_id,
                )
                return False

            if not isinstance(data, dict):
                logger.error(
                    "GpioHAL: write data must be a dict, got %s", type(data),
                )
                return False

            pin = data.get("pin")
            if pin is None:
                pin = device.metadata.get("pin")
            if pin is None:
                pin = self._extract_pin_number(device_id)

            value = int(data.get("value", 0))

            if self._on_pi and self._gpio is not None:
                # ── Real hardware path ────────────────────────────────────
                try:
                    self._ensure_pin_setup(pin, "out")
                    self._gpio.output(pin, value)
                    self._pin_values[pin] = value
                    logger.debug(
                        "GpioHAL [REAL]: pin %d = %d", pin, value,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "GpioHAL: real GPIO.output(%d, %d) failed: %s",
                        pin, value, exc,
                    )
                    return False
            else:
                # ── Simulated path ────────────────────────────────────────
                self._pin_values[pin] = value
                logger.debug(
                    "GpioHAL [SIM]: pin %d = %d", pin, value,
                )

            # Update device metadata with current pin binding.
            device.metadata["pin"] = pin
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "GpioHAL: error writing to '%s': %s",
                device_id, exc, exc_info=True,
            )
            return False

    async def list_devices(self) -> list[DeviceInfo]:
        """Return all GPIO devices currently tracked by this backend.

        Returns:
            A snapshot list of :class:`~nayak.hal.base.DeviceInfo` objects.
        """
        return list(self.devices.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Initialise the GPIO HAL backend and register it with NAYAK.

        Calls :meth:`~nayak.hal.base.HardwareBase.register` to publish this
        backend to the module registry and emit the HAL_READY bus event.

        This method **must** be awaited before any :meth:`connect` /
        :meth:`read` / :meth:`write` calls.

        Example::

            from nayak.hal.gpio import gpio_hal
            await gpio_hal.init()
        """
        await self.register()
        logger.info(
            "GpioHAL: initialised (on_pi=%s, pin_count=%d)",
            self._on_pi, len(self._pin_values),
        )

    async def stop(self) -> None:
        """Gracefully shut down the GPIO HAL backend.

        Disconnects all tracked devices, calls :meth:`cleanup` to release
        real GPIO resources, and updates the module registry status to
        ``STOPPED``.

        Safe to call even if no devices have been connected.

        Example::

            await gpio_hal.stop()
        """
        for device_id in list(self.devices.keys()):
            await self.disconnect(device_id)

        self.cleanup()

        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("GpioHAL: stopped and all GPIO resources released")
        except KeyError:
            logger.debug(
                "GpioHAL: stop() called before registration — "
                "skipping registry status update"
            )

    def cleanup(self) -> None:
        """Release all real GPIO resources.

        On a real Raspberry Pi this calls ``GPIO.cleanup()`` to reset all
        pin configurations.  In simulated mode this is a safe no-op.

        This method is called automatically by :meth:`stop` but may also be
        called explicitly during shutdown or error recovery.
        """
        if self._on_pi and self._gpio is not None:
            try:
                self._gpio.cleanup()
                logger.info("GpioHAL: GPIO.cleanup() completed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("GpioHAL: GPIO.cleanup() error: %s", exc)
        self._pin_values.clear()
        self._pin_directions.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_pin_setup(self, pin: int, direction: str) -> None:
        """Configure a GPIO pin if it hasn't been set up yet.

        Avoids redundant ``GPIO.setup()`` calls by tracking the configured
        direction for each pin in ``self._pin_directions``.

        Args:
            pin:       BCM pin number.
            direction: ``"in"`` for INPUT, ``"out"`` for OUTPUT.

        Raises:
            RuntimeError: If the GPIO library encounters a hardware error
                         (propagated to the caller for logging).
        """
        if self._gpio is None:
            return

        current = self._pin_directions.get(pin)
        if current == direction:
            return

        if direction == "out":
            self._gpio.setup(pin, self._gpio.OUT)
        else:
            self._gpio.setup(pin, self._gpio.IN)

        self._pin_directions[pin] = direction
        logger.debug("GpioHAL: pin %d configured as %s", pin, direction.upper())

    @staticmethod
    def _extract_pin_number(device_id: str) -> int:
        """Extract a BCM pin number from a device identifier string.

        Parses identifiers like ``"gpio_17"`` or ``"pin_4"`` by taking the
        last numeric segment.  Falls back to ``0`` if no number is found.

        Args:
            device_id: The device identifier to parse.

        Returns:
            The extracted pin number, or ``0`` if parsing fails.
        """
        import re

        match = re.search(r"(\d+)", device_id)
        if match:
            return int(match.group(1))
        logger.warning(
            "GpioHAL: could not extract pin number from '%s', defaulting to 0",
            device_id,
        )
        return 0

    @property
    def on_pi(self) -> bool:
        """Whether the backend detected real Raspberry Pi GPIO hardware.

        Returns:
            ``True`` if ``RPi.GPIO`` was imported successfully.
        """
        return self._on_pi


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

gpio_hal: GpioHAL = GpioHAL()
"""Process-wide GPIO HAL singleton.

Import and use directly — no instantiation required::

    from nayak.hal.gpio import gpio_hal

    await gpio_hal.init()
    await gpio_hal.connect("gpio_17")
    await gpio_hal.write("gpio_17", {"pin": 17, "value": 1})
    data = await gpio_hal.read("gpio_17")
    await gpio_hal.stop()
"""

__all__ = ["GpioHAL", "gpio_hal"]
