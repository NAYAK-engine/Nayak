"""
nayak/hal/spinal_bridge.py — Python bridge to the Rust Spinal Cord.

Manages the lifecycle of the Rust ``nayak_spinal`` subprocess and provides
a clean async API for sending hardware commands from the Python brain.

Data flow::

    Python (SpinalBridge.send_command)
        │
        │  JSON line  ──►  spinal process stdin
        │
    Rust (nayak_spinal — 1 000 Hz control loop)
        │
        │  JSON line  ──►  spinal process stdout
        │
    Python (_read_responses → bus.emit DEVICE_DATA)

The binary is expected at ``./nayak_core/target/release/nayak_spinal``
(built via ``cd nayak_core && cargo build --release``).  If the binary
is not found, :meth:`SpinalBridge.start` returns ``False`` and logs a
helpful build instruction — the rest of the NAYAK stack continues normally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from nayak.core.bus import bus, EventType, NayakEvent

logger = logging.getLogger(__name__)

# Path to the compiled Rust binary (relative to project root).
_BINARY_PATH = "./nayak_core/target/release/nayak_spinal"


class SpinalBridge:
    """Async Python bridge to the Rust Spinal Cord subprocess.

    Starts the ``nayak_spinal`` binary as a child process and maintains two
    async tasks:

    * ``_read_responses`` — continuously reads JSON lines from the binary's
      stdout and emits each payload as a :attr:`~EventType.DEVICE_DATA` event
      on the global event bus.

    * Callers use :meth:`send_command` to write JSON commands to the binary's
      stdin from anywhere in the Python codebase.

    The bridge is intentionally fault-tolerant: if the binary is not found
    (e.g. Rust is not installed on the development machine) it logs a clear
    build instruction and returns ``False`` from :meth:`start` without raising.
    The Python stack continues operating in simulation mode.

    Attributes:
        _process: The running asyncio subprocess, or ``None`` if not started.
        _running: ``True`` while the bridge is active.
    """

    def __init__(self) -> None:
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running: bool = False
        self._response_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Start the Rust spinal cord subprocess.

        Attempts to launch the pre-compiled ``nayak_spinal`` binary.  If the
        binary does not exist, logs a build instruction and returns ``False``
        so the rest of NAYAK continues in simulation mode.

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
                name="spinal-bridge-reader",
            )
            logger.info(
                "SpinalBridge: Rust spinal cord started — PID %d",
                self._process.pid,
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
        """Terminate the Rust spinal cord process and clean up.

        Safe to call even if the bridge was never started.
        """
        self._running = False

        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
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

    # ── Internal response reader ───────────────────────────────────────────────

    async def _read_responses(self) -> None:
        """Continuously read JSON responses from the Rust process stdout.

        Each line is parsed as JSON and emitted as a
        :attr:`~EventType.DEVICE_DATA` event on the global NAYAK event bus so
        any subscriber (e.g. HAL modules, the Studio dashboard) receives the
        hardware data in real time.

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

__all__ = ["SpinalBridge", "spinal_bridge"]
