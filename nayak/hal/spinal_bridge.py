"""
nayak/hal/spinal_bridge.py — Python bridge to the Rust Spinal Cord.

Manages the lifecycle of the Rust ``nayak_spinal`` subprocess and provides
a clean async API for sending hardware commands from the Python brain.

This module owns **two independent IPC channels**:

1. **JSON over stdin/stdout** (control signals)
   Small, text payloads such as GPIO commands and heartbeat ACKs.
   Written/read via :meth:`send_command` / :meth:`_read_responses`.

2. **Shared memory via mmap** (sensor data)
   Large, binary payloads: camera frames, LIDAR scans, tensor arrays.
   Written by the Rust spinal cord into a backing file every ~100 ms;
   read by Python via :meth:`read_shm_frame` / :meth:`start_frame_reader`
   with **zero JSON overhead** and **zero extra copies**.

Rule: **never mix them**.  Sensor data never goes into JSON IPC; control
signals never go through shared memory.

Data flow::

    ┌───────────────────────────────────────────────────────────────┐
    │ Python (SpinalBridge.send_command)                            │
    │     │                                                         │
    │     │  JSON line  ──►  spinal process stdin                    │
    │     │                                                         │
    │ Rust (nayak_spinal — 1 000 Hz control loop)                  │
    │     │ heartbeat   ──►  spinal process stdout  ──►  bus emit  │
    │     │ sensor frm  ──►  /tmp/nayak_sensor_frame (mmap)        │
    │     │                                                         │
    │ Python (start_frame_reader) polls mmap every 10 ms            │
    └───────────────────────────────────────────────────────────────┘

The binary is expected at ``./nayak_core/target/release/nayak_spinal``
(built via ``cd nayak_core && cargo build --release``).  If the binary
is not found, :meth:`SpinalBridge.start` returns ``False`` and logs a
helpful build instruction — the rest of the NAYAK stack continues normally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mmap
import os
import time
from typing import Optional

from nayak.core.bus import bus, EventType, NayakEvent

logger = logging.getLogger(__name__)

# Path to the compiled Rust binary (relative to project root).
_BINARY_PATH = "./nayak_core/target/release/nayak_spinal"

# Path to the shared memory backing file written by the Rust spinal cord.
# Must match the path used by `write_shm_frame()` in main.rs.
# On Windows this falls back to the OS temp directory automatically because
# the Rust side uses `std::env::temp_dir()` when `cfg!(windows)` is true.
_SHM_FRAME_PATH: str = (
    os.path.join(os.environ.get("TEMP", "/tmp"), "nayak_sensor_frame")
    if os.name == "nt"
    else "/tmp/nayak_sensor_frame"
)

# How often to poll the SHM file for a new sensor frame (seconds).
_SHM_POLL_INTERVAL_S: float = 0.010  # 10 ms → 100 Hz max read rate


class SpinalBridge:
    """Async Python bridge to the Rust Spinal Cord subprocess.

    Starts the ``nayak_spinal`` binary as a child process and maintains three
    async tasks:

    * ``_read_responses`` — continuously reads JSON lines from the binary's
      stdout and emits each payload as a :attr:`~EventType.DEVICE_DATA` event
      on the global event bus.  **Control signals only.**

    * :meth:`start_frame_reader` — polls the shared memory backing file every
      10 ms and emits a :attr:`~EventType.DEVICE_DATA` event for each sensor
      frame.  **Sensor / binary data only.**

    * Callers use :meth:`send_command` to write JSON commands to the binary's
      stdin from anywhere in the Python codebase.

    The two IPC channels are strictly separated:

    +---------------------------+-------------------------------------------+
    | Channel                   | Content                                   |
    +===========================+===========================================+
    | JSON over stdin/stdout    | Control signals (gpio_write, heartbeat …) |
    +---------------------------+-------------------------------------------+
    | mmap shared memory file   | Sensor frames (camera, LIDAR, tensors …)  |
    +---------------------------+-------------------------------------------+

    The bridge is intentionally fault-tolerant: if the binary is not found
    (e.g. Rust is not installed on the development machine) it logs a clear
    build instruction and returns ``False`` from :meth:`start` without raising.
    The Python stack continues operating in simulation mode.

    Attributes:
        _process:      The running asyncio subprocess, or ``None`` if not started.
        _running:      ``True`` while the bridge is active.
    """

    def __init__(self) -> None:
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running: bool = False
        self._response_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._frame_reader_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Start the Rust spinal cord subprocess and both IPC readers.

        Attempts to launch the pre-compiled ``nayak_spinal`` binary.  If the
        binary does not exist, logs a build instruction and returns ``False``
        so the rest of NAYAK continues in simulation mode.

        Launches two background tasks on success:

        * ``_read_responses`` — JSON IPC reader (control signals).
        * ``start_frame_reader`` — SHM mmap reader (sensor frames, 10 ms poll).

        Returns:
            ``True`` if the process started successfully, ``False`` otherwise.
        """
        if self._running:
            logger.warning("SpinalBridge: already running — ignoring duplicate start()")
            return True

        try:
            self._process = await asyncio.create_subprocess_exec(
                _BINARY_PATH,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._running = True
            self._response_task = asyncio.create_task(
                self._read_responses(),
                name="spinal-bridge-json-reader",
            )
            self._frame_reader_task = asyncio.create_task(
                self.start_frame_reader(),
                name="spinal-bridge-shm-reader",
            )
            logger.info(
                "SpinalBridge: Rust spinal cord started — PID %d "
                "| JSON IPC reader running "
                "| SHM frame reader running (poll=%dms)",
                self._process.pid,
                int(_SHM_POLL_INTERVAL_S * 1000),
            )
            return True

        except FileNotFoundError:
            logger.warning(
                "SpinalBridge: binary not found at '%s'. "
                "Build it with: cd nayak_core && cargo build --release",
                _BINARY_PATH,
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("SpinalBridge: failed to start spinal cord: %s", exc)
            return False

    async def stop(self) -> None:
        """Terminate the Rust spinal cord process and clean up both IPC tasks.

        Cancels the JSON reader task and the SHM frame reader task, then
        terminates the subprocess.  Safe to call even if the bridge was never
        started.
        """
        self._running = False

        # Cancel both reader tasks
        for task in (self._response_task, self._frame_reader_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
                logger.info("SpinalBridge: Rust spinal cord stopped")
            except Exception as exc:  # noqa: BLE001
                logger.warning("SpinalBridge: error during stop: %s", exc)
            finally:
                self._process = None

    # ── Command I/O ────────────────────────────────────────────────────────────

    async def send_command(
        self,
        command_type: str,
        device_id: str,
        payload: dict,
    ) -> bool:
        """Send a hardware command to the Rust spinal cord.

        Serializes the command as a newline-terminated JSON string and writes
        it to the spinal process stdin.  The response will arrive asynchronously
        via the :attr:`~EventType.DEVICE_DATA` event on the global event bus.

        Args:
            command_type: Opcode string (e.g. ``"gpio_write"``, ``"ping"``).
            device_id:    Target device identifier (e.g. ``"gpio.17"``).
            payload:      Dict of command-specific parameters.

        Returns:
            ``True`` if the command was written successfully, ``False`` if the
            bridge is not running or the write failed.
        """
        if not self._running or self._process is None:
            logger.debug(
                "SpinalBridge: send_command('%s') skipped — bridge not running",
                command_type,
            )
            return False

        cmd = {
            "command_type": command_type,
            "device_id": device_id,
            "payload": payload,
            "timestamp_ms": int(time.time() * 1000),
        }
        try:
            line = (json.dumps(cmd) + "\n").encode()
            self._process.stdin.write(line)
            await self._process.stdin.drain()
            logger.debug("SpinalBridge: sent command '%s' to '%s'", command_type, device_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("SpinalBridge: write error: %s", exc)
            return False

    # ── Shared memory frame reader ────────────────────────────────────────────

    def read_shm_frame(self, shm_path: str = _SHM_FRAME_PATH) -> Optional[bytes]:
        """Read the latest sensor frame from the shared memory backing file.

        Opens the file at *shm_path* and maps it read-only with Python's
        built-in :mod:`mmap` module.  Returns the raw bytes directly — **no
        JSON parsing, no deserialisation** — exactly as the Rust spinal cord
        wrote them.

        This method is **synchronous** and is designed to be called from the
        async :meth:`start_frame_reader` loop inside
        ``asyncio.get_event_loop().run_in_executor`` if blocking I/O is a
        concern.  For the 10 ms polling cadence of the frame reader the
        blocking time is negligible.

        Args:
            shm_path: Path to the shared memory backing file.  Defaults to
                      the platform-appropriate path written by the Rust side
                      (``/tmp/nayak_sensor_frame`` on Linux/macOS, the OS
                      temp directory on Windows).

        Returns:
            Raw ``bytes`` of the sensor frame on success.
            ``None`` if the file does not exist yet (Rust has not written the
            first frame), or on any I/O error.
        """
        try:
            with open(shm_path, "rb") as f:
                file_size = os.fstat(f.fileno()).st_size
                if file_size == 0:
                    # File exists but Rust hasn't written data yet.
                    return None
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    frame = bytes(mm[:file_size])
            logger.debug(
                "SpinalBridge [SHM]: received frame %d bytes from '%s'",
                len(frame), shm_path,
            )
            return frame
        except FileNotFoundError:
            # Normal at startup — the Rust side hasn't produced the first
            # frame yet.  Suppress: the reader will retry on the next poll.
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SpinalBridge [SHM]: error reading '%s': %s", shm_path, exc
            )
            return None

    async def start_frame_reader(
        self,
        shm_path: str = _SHM_FRAME_PATH,
        poll_interval_s: float = _SHM_POLL_INTERVAL_S,
    ) -> None:
        """Async loop that polls the shared memory file and emits sensor events.

        Runs **independently** of the JSON IPC reader (:meth:`_read_responses`).
        Every *poll_interval_s* seconds (default: 10 ms) it calls
        :meth:`read_shm_frame`.  When a frame is received it emits a
        :attr:`~EventType.DEVICE_DATA` event on the global event bus with a
        lightweight metadata payload::

            {"source": "spinal-shm", "size": <frame size in bytes>}

        The raw frame bytes are **never placed inside the event payload** —
        that would defeat the purpose of shared memory.  Consumers who need
        the actual frame data must call :meth:`read_shm_frame` directly or
        use :data:`~nayak.core.bus.shm_bus`.

        Args:
            shm_path:        Path to the backing file.  Defaults to the
                             platform-appropriate path.
            poll_interval_s: Polling interval in seconds.  Default is 0.010
                             (10 ms, giving up to 100 Hz read rate).

        Note:
            This coroutine is started automatically by :meth:`start` as a
            named asyncio task.  You only need to call it directly if you
            want to run the SHM reader without starting the full subprocess
            (e.g. in simulation mode where another process writes the file).
        """
        logger.info(
            "SpinalBridge [SHM]: frame reader started — polling '%s' every %d ms",
            shm_path, int(poll_interval_s * 1000),
        )

        while self._running:
            try:
                await asyncio.sleep(poll_interval_s)

                frame = self.read_shm_frame(shm_path)
                if frame is None:
                    # File not ready yet or transient read error; try next poll.
                    continue

                logger.debug(
                    "SpinalBridge [SHM]: frame received — %d bytes", len(frame)
                )

                # Emit a lightweight metadata event.
                # The raw bytes stay in shared memory — never in the event payload.
                await bus.emit(NayakEvent(
                    type=EventType.DEVICE_DATA,
                    payload={"source": "spinal-shm", "size": len(frame)},
                    source="spinal-shm",
                ))

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if self._running:
                    logger.error("SpinalBridge [SHM]: frame reader error: %s", exc)
                break

        logger.info("SpinalBridge [SHM]: frame reader stopped")

    # ── Internal JSON response reader ─────────────────────────────────────────

    async def _read_responses(self) -> None:
        """Continuously read JSON responses from the Rust process stdout.

        Each line is parsed as JSON and emitted as a
        :attr:`~EventType.DEVICE_DATA` event on the global NAYAK event bus so
        any subscriber (e.g. HAL modules, the Studio dashboard) receives the
        hardware data in real time.

        This coroutine handles **control signals only** (heartbeat, GPIO ACKs,
        etc.).  Sensor data never appears here — it arrives via
        :meth:`start_frame_reader` through the SHM channel.

        This coroutine runs for the lifetime of the bridge and exits when the
        subprocess closes stdout or the bridge is stopped.
        """
        if self._process is None or self._process.stdout is None:
            return

        while self._running:
            try:
                raw = await self._process.stdout.readline()
                if not raw:
                    logger.info("SpinalBridge: stdout closed — reader exiting")
                    break

                line = raw.decode().strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("SpinalBridge: invalid JSON from spinal cord: %s | %s", exc, line[:80])
                    continue

                await bus.emit(NayakEvent(
                    type=EventType.DEVICE_DATA,
                    payload=data,
                    source="spinal-cord",
                ))

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if self._running:
                    logger.error("SpinalBridge: read error: %s", exc)
                break

    # ── Convenience helpers ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """``True`` if the Rust process is active."""
        return self._running and self._process is not None

    async def ping(self) -> bool:
        """Send a ping command and return whether the write succeeded."""
        return await self.send_command("ping", "system", {})


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

spinal_bridge: SpinalBridge = SpinalBridge()
"""Process-wide bridge to the Rust Spinal Cord.

Usage::

    from nayak.hal.spinal_bridge import spinal_bridge

    await spinal_bridge.start()
    await spinal_bridge.send_command("gpio_write", "gpio.17", {"pin": 17, "value": 1})
    await spinal_bridge.stop()
"""

__all__ = ["SpinalBridge", "spinal_bridge", "read_shm_frame"]
