"""
nayak/hal/camera.py — NAYAK HAL: Camera hardware backend.

Provides a concrete :class:`HardwareBase` implementation for camera devices.
When ``cv2`` (OpenCV) is available *and* a physical camera can be opened the
backend operates in **real hardware mode** — frames are captured directly from
the device.  When OpenCV is not installed, or when no camera is present, the
backend falls back to **simulated mode**: every read returns a deterministic
metadata dict with ``"frame": None`` so the full NAYAK perception stack can
run on any machine without hardware.

Usage::

    from nayak.hal.camera import camera

    await camera.init()

    ok = await camera.connect("cam_0")
    frame_data = await camera.read("cam_0")
    # frame_data["frame"] is a numpy array (real) or None (simulated)

    await camera.stop()

The module exposes a process-wide singleton :data:`camera` so that all NAYAK
layers share exactly one set of camera capture contexts.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nayak.core.registry import ModuleStatus, registry
from nayak.hal.base import DeviceInfo, DeviceStatus, DeviceType, HardwareBase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CameraHAL
# ─────────────────────────────────────────────────────────────────────────────

class CameraHAL(HardwareBase):
    """Camera hardware backend for the NAYAK Hardware Abstraction Layer.

    Wraps OpenCV's ``VideoCapture`` interface and exposes it through the
    standard HAL contract.  Operates in two modes:

    **Real hardware mode** (``cv2`` available, camera index openable)
        :meth:`read` returns ``{"frame": <numpy.ndarray>, "simulated": False,
        "timestamp": <float>}``.  The ``frame`` value is a raw BGR image array
        as returned by ``cv2.VideoCapture.read()``.

    **Simulated mode** (``cv2`` absent *or* ``VideoCapture`` fails to open)
        :meth:`read` returns ``{"frame": None, "simulated": True,
        "resolution": "640x480", "fps": 30, "timestamp": <float>}``.
        No OpenCV dependency is required for this path — safe on any machine.

    Cameras are **read-only** devices; :meth:`write` always returns ``False``
    with a logged warning.

    Attributes:
        devices (dict[str, DeviceInfo]):
            Registry of every camera device managed by this backend, keyed by
            ``device_id``.  ``DeviceInfo.metadata`` stores the ``cv2``
            ``VideoCapture`` object under the key ``"cap"`` when running in
            real hardware mode.
    """

    # ── HardwareBase contract ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Dot-namespaced registry identifier for this backend.

        Returns:
            ``"camera-hal"``
        """
        return "camera-hal"

    # ── Core async interface ──────────────────────────────────────────────────

    async def connect(self, device_id: str) -> bool:
        """Connect to the camera identified by *device_id*.

        Creates (or resets) a :class:`~nayak.hal.base.DeviceInfo` entry with
        ``device_type=CAMERA`` and attempts to open a ``cv2.VideoCapture``
        handle for device index ``0`` (default camera).  The method succeeds in
        three scenarios:

        1. **OpenCV available + camera opens** — real hardware mode;
           the ``VideoCapture`` object is stored in
           ``DeviceInfo.metadata["cap"]``.
        2. **OpenCV available but camera fails to open** — status set to
           :attr:`~nayak.hal.base.DeviceStatus.ERROR`; returns ``False``.
        3. **OpenCV not installed** — simulated mode;
           ``DeviceInfo.metadata["simulated"] = True``; returns ``True``.

        Args:
            device_id: Unique identifier for this camera connection
                       (e.g. ``"cam_0"``, ``"front_camera"``).

        Returns:
            ``True`` on success (real or simulated), ``False`` if a physical
            camera could not be opened or an unexpected error occurred.
        """
        try:
            # ── Create / reset DeviceInfo entry ───────────────────────────────
            self.devices[device_id] = DeviceInfo(
                device_id=device_id,
                device_type=DeviceType.CAMERA,
                name=f"Camera [{device_id}]",
                status=DeviceStatus.CONNECTING,
            )
            device = self.devices[device_id]

            # ── Attempt OpenCV import ─────────────────────────────────────────
            try:
                import cv2  # type: ignore[import]

                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    device.status = DeviceStatus.CONNECTED
                    device.metadata["simulated"] = False
                    device.metadata["cap"] = cap
                    device.metadata["cv2_available"] = True
                    device.connected_at = time.time()
                    logger.info(
                        "CameraHAL: '%s' connected via OpenCV (real hardware)",
                        device_id,
                    )
                    return True
                else:
                    # OpenCV present but no camera accessible at index 0.
                    cap.release()
                    device.status = DeviceStatus.ERROR
                    logger.error(
                        "CameraHAL: '%s' — OpenCV available but "
                        "VideoCapture(0) failed to open",
                        device_id,
                    )
                    return False

            except ImportError:
                # cv2 not installed — fall through to simulated mode.
                device.status = DeviceStatus.CONNECTED
                device.metadata["simulated"] = True
                device.metadata["cv2_available"] = False
                device.connected_at = time.time()
                logger.info(
                    "CameraHAL: '%s' connected in simulated mode "
                    "(cv2 / OpenCV not available)",
                    device_id,
                )
                return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CameraHAL: unexpected error connecting '%s': %s",
                device_id, exc, exc_info=True,
            )
            if device_id in self.devices:
                self.devices[device_id].status = DeviceStatus.ERROR
            return False

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect the camera identified by *device_id*.

        Releases the underlying ``cv2.VideoCapture`` handle if one is present,
        then sets the device status to
        :attr:`~nayak.hal.base.DeviceStatus.DISCONNECTED`.
        Safe to call even if *device_id* is not tracked.

        Args:
            device_id: Unique identifier for the camera to disconnect.

        Returns:
            Always ``True`` — disconnection failures must never block shutdown.
        """
        try:
            device = self.devices.get(device_id)
            if device is None:
                logger.warning(
                    "CameraHAL: disconnect called for unknown device '%s'",
                    device_id,
                )
                return True

            cap = device.metadata.get("cap")
            if cap is not None:
                try:
                    cap.release()
                    logger.debug(
                        "CameraHAL: VideoCapture released for '%s'", device_id
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "CameraHAL: error releasing VideoCapture for '%s': %s",
                        device_id, exc,
                    )
                device.metadata.pop("cap", None)

            device.status = DeviceStatus.DISCONNECTED
            logger.info("CameraHAL: '%s' disconnected", device_id)
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CameraHAL: error disconnecting '%s': %s",
                device_id, exc, exc_info=True,
            )
            return True  # Never block shutdown on release failure.

    async def read(self, device_id: str) -> Any:
        """Read the latest frame from the camera identified by *device_id*.

        **Simulated mode** returns a metadata-only dict — no real image data::

            {
                "frame":      None,
                "simulated":  True,
                "resolution": "640x480",
                "fps":        30,
                "timestamp":  <float>,
            }

        **Real hardware mode** captures a live frame via ``VideoCapture.read()``::

            {
                "frame":      <numpy.ndarray>,   # BGR image, shape (H, W, 3)
                "simulated":  False,
                "timestamp":  <float>,
            }

        If the capture fails (``ret=False``), ``None`` is returned and the
        error is logged.  A :attr:`~nayak.core.bus.EventType.DEVICE_DATA` bus
        event is emitted after every successful read so that perception backends
        can subscribe without polling.

        Args:
            device_id: Unique identifier of the camera to read from.

        Returns:
            A frame dict on success, or ``None`` if the device is not connected
            / the capture failed.
        """
        try:
            device = self.devices.get(device_id)
            if device is None or device.status not in (
                DeviceStatus.CONNECTED,
                DeviceStatus.ACTIVE,
            ):
                logger.debug(
                    "CameraHAL: read skipped for '%s' — not connected "
                    "(status=%s)",
                    device_id,
                    device.status.name if device else "UNKNOWN",
                )
                return None

            # ── Simulated path ────────────────────────────────────────────────
            if device.metadata.get("simulated", False):
                data: dict[str, Any] = {
                    "frame":      None,
                    "simulated":  True,
                    "resolution": "640x480",
                    "fps":        30,
                    "timestamp":  time.time(),
                }
                await self.emit_device_event(device_id, data)
                return data

            # ── Real hardware path ────────────────────────────────────────────
            cap = device.metadata.get("cap")
            if cap is None:
                logger.error(
                    "CameraHAL: no VideoCapture handle for '%s'", device_id
                )
                return None

            ret, frame = cap.read()
            if ret:
                data = {
                    "frame":     frame,
                    "simulated": False,
                    "timestamp": time.time(),
                }
                await self.emit_device_event(device_id, data)
                return data

            logger.warning(
                "CameraHAL: VideoCapture.read() returned False for '%s'",
                device_id,
            )
            return None

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CameraHAL: error reading from '%s': %s",
                device_id, exc, exc_info=True,
            )
            return None

    async def write(self, device_id: str, data: Any) -> bool:
        """Cameras are read-only — write operations are not supported.

        This method always returns ``False`` and emits a ``WARNING``-level log
        so that callers know immediately that they have sent a command to a
        device that cannot act on it.

        Args:
            device_id: Identifier of the target camera (logged but unused).
            data:      Command payload (logged but ignored).

        Returns:
            Always ``False``.
        """
        logger.warning(
            "CameraHAL: write() called on read-only camera '%s' — "
            "operation ignored (data=%r)",
            device_id, data,
        )
        return False

    async def list_devices(self) -> list[DeviceInfo]:
        """Return all cameras currently tracked by this backend.

        Returns:
            A snapshot list of :class:`~nayak.hal.base.DeviceInfo` objects for
            every camera that has been connected via :meth:`connect`.
        """
        return list(self.devices.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Initialise the Camera HAL backend and register it with NAYAK.

        Calls :meth:`~nayak.hal.base.HardwareBase.register` to publish this
        backend to the module registry and emit the
        :attr:`~nayak.core.bus.EventType.HAL_READY` bus event.

        This method **must** be awaited before any :meth:`connect` / :meth:`read`
        calls are made.

        Example::

            from nayak.hal.camera import camera
            await camera.init()
        """
        await self.register()
        logger.info("CameraHAL: Camera HAL initialized")

    async def stop(self) -> None:
        """Gracefully shut down the Camera HAL backend.

        Releases all ``VideoCapture`` handles by calling :meth:`disconnect` on
        every tracked device, then promotes the module status to
        :attr:`~nayak.core.registry.ModuleStatus.STOPPED` in the registry.

        Safe to call even if no devices have been connected.

        Example::

            await camera.stop()
        """
        for device_id in list(self.devices.keys()):
            await self.disconnect(device_id)

        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info(
                "CameraHAL: stopped and all VideoCapture handles released"
            )
        except KeyError:
            # stop() called before init() — registry entry does not exist yet.
            logger.debug(
                "CameraHAL: stop() called before registration — "
                "skipping registry status update"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

camera: CameraHAL = CameraHAL()
"""Process-wide Camera HAL singleton.

Import and use directly — no instantiation required::

    from nayak.hal.camera import camera

    await camera.init()
    await camera.connect("cam_0")
    frame_data = await camera.read("cam_0")
    # frame_data["frame"] → numpy array (real) or None (simulated)
    await camera.stop()
"""

__all__ = ["CameraHAL", "camera"]
