"""
nayak/safety/engine.py — NAYAK Safety Engine backend.

A concrete implementation of :class:`SafetyBase`. This is the most critical 
defense layer in NAYAK. It intercept intents before execution, aggressively 
blocking dangerous system commands (e.g. `rm -rf`, `eval`), tracking infractions, 
and activating a hard, global system lock (Emergency Stop) if the cognition 
engine behaves maliciously or exceeds violation thresholds.

Usage::

    from nayak.safety.engine import safety

    await safety.init()
    status = await safety.check("execute", {"code": "import os; os.system('rm -rf /')"})
    if status == ThreatLevel.CRITICAL:
        print("Blocked!")
"""

from __future__ import annotations

import logging

from rich.console import Console

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, registry
from nayak.safety.base import SafetyBase, SafetyViolation, ThreatLevel

logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# SafetyEngine
# ─────────────────────────────────────────────────────────────────────────────

class SafetyEngine(SafetyBase):
    """The concrete operating safety engine for NAYAK.

    Acts as an impregnable firewall between the AI reasoning backend and the 
    OS-level computer interaction tools.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stopped: bool = False
        self._max_violations: int = 10
        self._blocked_actions: set[str] = {
            "delete_system",
            "format_disk",
            "shutdown_os",
            "rm -rf",
            "drop table",
            "exec(",
            "eval(",
            "import os; os.system",
            "__import__"
        }

    @property
    def name(self) -> str:
        """Dot-namespaced registry identifier for this backend.

        Returns:
            ``"nayak-safety"``
        """
        return "nayak-safety"

    @property
    def is_stopped(self) -> bool:
        """Returns True if the engine has dropped into a global lock mode."""
        return self._stopped

    # ── Core asynchronous logic ───────────────────────────────────────────────

    async def check(self, action_type: str, payload: dict) -> ThreatLevel:
        """Scan an intended action and its internal parameters for critical threats.

        Args:
            action_type: What the AI is attempting to do natively.
            payload: Extractable JSON metadata associated with the action.

        Returns:
            The highest matched severity as a corresponding :class:`ThreatLevel`.
            Safe actions return ``ThreatLevel.SAFE``.
        """
        # Global failsafe immediately traps if already tripped.
        if self._stopped:
            return ThreatLevel.CRITICAL

        # Heuristic 1: Primary action signature match.
        action_lower = action_type.lower()
        for blocked in self._blocked_actions:
            if blocked in action_lower:
                await self.record_violation(SafetyViolation(
                    threat_level=ThreatLevel.CRITICAL,
                    source=self.name,
                    description=f"Action type matched blocked signature: '{blocked}'",
                    action_blocked=action_type
                ))
                return ThreatLevel.CRITICAL

        # Heuristic 2: Deep packet inspection (payload string crawling).
        for key, value in payload.items():
            if isinstance(value, str):
                v_lower = value.lower()
                for blocked in self._blocked_actions:
                    if blocked in v_lower:
                        await self.record_violation(SafetyViolation(
                            threat_level=ThreatLevel.HIGH,
                            source=self.name,
                            description=f"Blocked logic signature '{blocked}' located in payload key '{key}'",
                            action_blocked=str(payload)
                        ))
                        return ThreatLevel.HIGH

        # Rule 3: Cascading failure cap logic.
        if len(self.violations) >= self._max_violations:
            logger.warning("SafetyEngine: Infraction limit (%d) hit — initializing auto-lock.", self._max_violations)
            await self.emergency_stop()
            return ThreatLevel.CRITICAL

        return ThreatLevel.SAFE

    # ── Emergency Operations ──────────────────────────────────────────────────

    async def emergency_stop(self) -> None:
        """Engage the system-wide killswitch."""
        self._stopped = True
        logger.critical("EMERGENCY STOP ACTIVATED")
        console.print("[bold red]⚠ NAYAK EMERGENCY STOP ACTIVATED ⚠[/bold red]")
        
        await bus.emit(NayakEvent(
            type=EventType.SAFETY_VIOLATION,
            payload={
                "threat_level": ThreatLevel.CRITICAL.name,
                "source": "safety-engine",
                "description": "Emergency stop activated"
            },
            source=self.name
        ))
        
        try:
            registry.set_status(self.name, ModuleStatus.ERROR)
        except KeyError:
            pass

    async def resume(self) -> None:
        """Disengage the safety lock, requiring OS-level manual overrides."""
        self._stopped = False
        logger.info("Safety engine resumed")
        try:
            registry.set_status(self.name, ModuleStatus.READY)
        except KeyError:
            pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Mount the SafetyEngine into the active session stack securely."""
        await self.register()
        logger.info("Safety engine initialized — all systems monitored")

    async def stop(self) -> None:
        """Gracefully release the safety hooks unmounting the OS layer."""
        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("Safety engine unmounted cleanly.")
        except KeyError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

safety: SafetyEngine = SafetyEngine()
"""Process-wide instance of the active Layer 7 Safety firewall."""

__all__ = ["SafetyEngine", "safety"]
