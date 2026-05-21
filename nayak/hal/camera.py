"""
nayak/hal/camera.py — NAYAK HAL: Camera hardware backend.

Provides a concrete :class:`HardwareBase` implementation for camera devices
with **three-tier automatic hardware detection**:

1. **picamera2** (Raspberry Pi CSI camera) — highest priority.  Native Pi
   camera library providing direct access to the Broadcom ISP.
2. **OpenCV** (``cv2.VideoCapture``) — fallback for USB webcams and generic
   V4L2 devices.  Works on Pi, Linux, macOS, and Windows.
3. **Simulated** — zero-dependency mode.  Returns metadata-only dicts so the
   full NAYAK perception stack can run on any machine without hardware.

Detection runs **at connect time** per device.  The selected backend is stored
in ``DeviceInfo.metadata["camera_type"]`` (``"picamera2"`` | ``"opencv"`` |
``"simulated"``) and every :meth:`read` result carries a ``"source"`` key for
debugging provenance.

Usage::

    from nayak.hal.camera import camera

    await camera.init()

    ok = await camera.connect("cam_0")
    frame_data = await camera.read("cam_0")
    # frame_data["frame"] is a numpy array (real) or None (simulated)
    # frame_data["source"] in {"picamera2", "opencv", "simulated"}

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

    Implements three-tier automatic camera detection at connect time:

    **Tier 1 — picamera2** (``camera_type="picamera2"``)
        Uses the ``picamera2`` library for native Raspberry Pi CSI camera
        access.  :meth:`read` returns ``{"frame": <numpy.ndarray>,
        "source": "picamera2", "simulated": False, ...}``.

    **Tier 2 — OpenCV** (``camera_type="opencv"``)
        Falls back to ``cv2.VideoCapture(0)`` for USB webcams and generic
        capture devices.  :meth:`read` returns ``{"frame": <numpy.ndarray>,
        "source": "opencv", "simulated": False, ...}``.

    **Tier 3 — Simulated** (``camera_type="simulated"``)
        No real camera available.  :meth:`read` returns ``{"frame": None,
        "source": "simulated", "simulated": True, "resolution": "640x480",
        "fps": 30, ...}``.

    Cameras are **read-only** devices; :meth:`write` always returns ``False``
    with a logged warning.

    Attributes:
        devices (dict[str, DeviceInfo]):
            Registry of every camera device managed by this backend, keyed by
            ``device_id``.
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

        Runs the three-tier auto-detection sequence:

        1. **picamera2** — ``from picamera2 import Picamera2``.  If successful,
           creates a ``Picamera2`` instance, configures a preview at 640×480,
           and starts it.
        2. **OpenCV** — ``cv2.VideoCapture(0)``.  If the capture opens
           successfully, the handle is stored in device metadata.
        3. **Simulated** — both real backends failed; the device is registered
           in simulation mode.

        Args:
            device_id: Unique identifier for this camera connection
                       (e.g. ``"cam_0"``, ``"front_camera"``).

        Returns:
            ``True`` on success (any tier), ``False`` only on unexpected error.
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

            # ── Tier 1: picamera2 (Pi CSI camera) ─────────────────────────────
            try:
                from picamera2 import Picamera2  # type: ignore[import]

                picam = Picamera2()
                config = picam.create_preview_configuration(
                    main={"size": (640, 480)},
                )
                picam.configure(config)
                picam.start()

                device.status = DeviceStatus.CONNECTED
                device.metadata["simulated"] = False
                device.metadata["camera_type"] = "picamera2"
                device.metadata["picam"] = picam
                device.connected_at = time.time()
                logger.info(
                    "CameraHAL: '%s' connected via picamera2 "
                    "(Raspberry Pi CSI camera)",
                    device_id,
                )
                return True

            except (ImportError, RuntimeError, OSError) as exc:
                logger.debug(
                    "CameraHAL: picamera2 not available for '%s': %s",
                    device_id, exc,
                )

            # ── Tier 2: OpenCV (USB webcam / generic V4L2) ────────────────────
            try:
                import cv2  # type: ignore[import]

                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    device.status = DeviceStatus.CONNECTED
                    device.metadata["simulated"] = False
                    device.metadata["camera_type"] = "opencv"
                    device.metadata["cap"] = cap
                    device.metadata["cv2_available"] = True
                    device.connected_at = time.time()
                    logger.info(
                        "CameraHAL: '%s' connected via OpenCV "
                        "(USB / generic camera)",
                        device_id,
                    )
                    return True
                else:
                    cap.release()
                    logger.debug(
                        "CameraHAL: OpenCV available but VideoCapture(0) "
                        "failed for '%s'",
                        device_id,
                    )

            except ImportError:
                logger.debug(
                    "CameraHAL: cv2 / OpenCV not available for '%s'",
                    device_id,
                )

            # ── Tier 3: Simulated mode ────────────────────────────────────────
            device.status = DeviceStatus.CONNECTED
            device.metadata["simulated"] = True
            device.metadata["camera_type"] = "simulated"
            device.metadata["cv2_available"] = False
            device.connected_at = time.time()
            logger.info(
                "CameraHAL: '%s' connected in SIMULATED mode "
                "(no camera hardware detected)",
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

        Releases the underlying capture handle based on the detected camera
        type:

        * **picamera2** — calls ``picam.stop()`` then ``picam.close()``.
        * **OpenCV** — calls ``cap.release()``.
        * **Simulated** — no resources to release.

        Then sets the device status to ``DISCONNECTED``.  Safe to call even if
        *device_id* is not tracked.

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

            camera_type = device.metadata.get("camera_type", "simulated")

            # ── Release picamera2 ─────────────────────────────────────────────
            if camera_type == "picamera2":
                picam = device.metadata.pop("picam", None)
                if picam is not None:
                    try:
                        picam.stop()
                        picam.close()
                        logger.debug(
                            "CameraHAL: picamera2 released for '%s'",
                            device_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "CameraHAL: error releasing picamera2 for '%s': %s",
                            device_id, exc,
                        )

            # ── Release OpenCV ────────────────────────────────────────────────
            elif camera_type == "opencv":
                cap = device.metadata.pop("cap", None)
                if cap is not None:
                    try:
                        cap.release()
                        logger.debug(
                            "CameraHAL: VideoCapture released for '%s'",
                            device_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "CameraHAL: error releasing VideoCapture for '%s': %s",
                            device_id, exc,
                        )

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

        Routes to the correct capture backend based on
        ``DeviceInfo.metadata["camera_type"]``:

        **picamera2**::

            {
                "frame":      <numpy.ndarray>,  # BGR image, shape (H, W, 3)
                "source":     "picamera2",
                "simulated":  False,
                "timestamp":  <float>,
            }

        **OpenCV**::

            {
                "frame":      <numpy.ndarray>,  # BGR image, shape (H, W, 3)
                "source":     "opencv",
                "simulated":  False,
                "timestamp":  <float>,
            }

        **Simulated**::

            {
                "frame":      None,
                "source":     "simulated",
                "simulated":  True,
                "resolution": "640x480",
                "fps":        30,
                "timestamp":  <float>,
            }

        A :attr:`~nayak.core.bus.EventType.DEVICE_DATA` bus event is emitted
        after every successful read.

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

            camera_type = device.metadata.get("camera_type", "simulated")

            # ── picamera2 path ────────────────────────────────────────────────
            if camera_type == "picamera2":
                picam = device.metadata.get("picam")
                if picam is None:
                    logger.error(
                        "CameraHAL: no Picamera2 instance for '%s'", device_id,
                    )
                    return None
                try:
                    frame = picam.capture_array()
                    data: dict[str, Any] = {
                        "frame": frame,
                        "source": "picamera2",
                        "simulated": False,
                        "timestamp": time.time(),
                    }
                    await self.emit_device_event(device_id, data)
                    return data
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "CameraHAL: picamera2 capture failed for '%s': %s",
                        device_id, exc,
                    )
                    return None

            # ── OpenCV path ───────────────────────────────────────────────────
            if camera_type == "opencv":
                cap = device.metadata.get("cap")
                if cap is None:
                    logger.error(
                        "CameraHAL: no VideoCapture handle for '%s'", device_id,
                    )
                    return None

                ret, frame = cap.read()
                if ret:
                    data = {
                        "frame": frame,
                        "source": "opencv",
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

            # ── Simulated path ────────────────────────────────────────────────
            data = {
                "frame": None,
                "source": "simulated",
                "simulated": True,
                "resolution": "640x480",
                "fps": 30,
                "timestamp": time.time(),
            }
            await self.emit_device_event(device_id, data)
            return data

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

        Releases all capture handles (picamera2, OpenCV) by calling
        :meth:`disconnect` on every tracked device, then promotes the module
        status to :attr:`~nayak.core.registry.ModuleStatus.STOPPED` in the
        registry.

        Safe to call even if no devices have been connected.

        Example::

            await camera.stop()
        """
        for device_id in list(self.devices.keys()):
            await self.disconnect(device_id)

        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info(
                "CameraHAL: stopped and all capture handles released"
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
    # frame_data["source"] → "picamera2" | "opencv" | "simulated"
    await camera.stop()
"""

__all__ = ["CameraHAL", "camera"]
