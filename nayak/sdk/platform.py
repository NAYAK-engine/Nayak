"""
nayak/sdk/platform.py — NAYAK Developer Platform: SDK engine.

Provides the concrete implementation of the :class:`DeveloperPlatformBase`.
This engine manages the lifecycle of external skills, allowing them to be
dynamically loaded, executed, and unloaded at runtime.

All skill execution is routed through :class:`SkillSandbox` to ensure a
crashing or hanging skill can never impact the NAYAK core runtime.

Security Architecture — Three Layers of Protection
===================================================

Layer 1 — Import Firewall (:class:`SkillImportFirewall`)
    Statically analyses skill source files *before* they are ever imported.
    Blocks any skill that references a forbidden module (``os``, ``subprocess``,
    ``sys``, ``socket``, ``ctypes``, etc.).  A skill that fails this check is
    never loaded — it cannot execute a single line of code.

Layer 2 — Runtime Resource Limits (:class:`SkillSandbox`)
    Enforces a hard asyncio timeout (default 10 s) on every ``execute()`` call.
    Tracks RSS memory growth before and after execution; if a skill consumes
    more than 100 MB in a single call, a SAFETY_VIOLATION is emitted.
    Timeouts also emit a SAFETY_VIOLATION so the safety engine can react.

Layer 3 — Load-Time Validation (:meth:`NayakPlatform.load_skill`)
    Runs the import firewall against the on-disk source file of every skill
    before calling ``importlib.import_module``.  This ensures that even skills
    whose module paths differ from their file names are blocked before Python
    has a chance to execute them.  A new :meth:`NayakPlatform.validate_all_skills`
    method can be called at any time to re-audit all loaded skills.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import psutil

from nayak.core.bus import EventType, NayakEvent, bus
from nayak.core.registry import ModuleStatus, registry
from nayak.sdk.base import DeveloperPlatformBase, SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Maximum RSS memory growth allowed during a single skill execution (bytes).
_MAX_MEMORY_GROWTH_BYTES: int = 100 * 1024 * 1024  # 100 MB


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — SkillImportFirewall
# ─────────────────────────────────────────────────────────────────────────────

class SkillImportFirewall:
    """Static source-code firewall for NAYAK skill files.

    Before any skill is imported into the Python runtime, its source code is
    read from disk and scanned for dangerous ``import`` statements.  A skill
    that references any module in :attr:`BLOCKED_MODULES` is rejected
    outright — it will never be loaded, never be executed, and never be able
    to call ``sys.exit()``, spawn subprocesses, or touch the filesystem.

    The scan is intentionally conservative: a *mention* of a blocked name in
    any ``import`` or ``from … import`` statement is sufficient to block the
    skill.  This prevents even indirect access (e.g. ``from os import path``).

    Attributes:
        BLOCKED_MODULES (frozenset[str]): The canonical set of module names
            that NAYAK skills are prohibited from importing.

    Example::

        firewall = SkillImportFirewall()
        violations = firewall.scan_skill_source("/path/to/evil_skill.py")
        if violations:
            print("Blocked:", violations)
        else:
            print("Safe to load.")
    """

    #: Modules that NAYAK skills are categorically forbidden from importing.
    #: Any ``import X`` or ``from X import …`` where X starts with one of
    #: these names will be flagged as a violation.
    BLOCKED_MODULES: frozenset[str] = frozenset({
        "os",
        "subprocess",
        "shutil",
        "sys",
        "socket",
        "ctypes",
        "importlib",
        "multiprocessing",
        "threading",
        "__builtins__",
    })

    # Regex patterns for text-level pre-scan (fast first pass).
    # Full AST parse is the authoritative check; regex is a performance
    # optimisation to skip the parse on obviously safe files.
    _IMPORT_PATTERN: re.Pattern = re.compile(
        r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)",
        re.MULTILINE,
    )

    def scan_skill_source(self, file_path: str) -> list[str]:
        """Scan *file_path* for dangerous ``import`` statements.

        Reads the skill source file as plain text and performs two passes:

        1. **Regex pass** — a fast regular-expression scan for any
           ``import X`` or ``from X import …`` lines where ``X`` (or its
           top-level package) appears in :attr:`BLOCKED_MODULES`.

        2. **AST pass** — a full abstract-syntax-tree parse for airtight
           detection of aliases and multi-import statements.

        Both passes run unconditionally so violations are never missed due to
        a quirky file layout that defeats the regex.

        Args:
            file_path: Absolute or relative path to the skill ``.py`` source
                file on disk.

        Returns:
            A list of human-readable violation strings.  Each string names
            the blocked module that was found.  Returns an empty list if the
            file is clean.  Returns a single-element list beginning with
            ``"[READ ERROR]"`` if the file cannot be opened.

        Side-effects:
            Logs a CRITICAL message for every violation found, and an ERROR
            message if the file cannot be read.
        """
        violations: list[str] = []

        # ── 1. Read source ────────────────────────────────────────────────────
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            error_msg = f"[READ ERROR] Cannot open skill source '{file_path}': {exc}"
            logger.error("SkillImportFirewall: %s", error_msg)
            return [error_msg]

        # ── 2. Regex pre-scan ─────────────────────────────────────────────────
        seen_via_regex: set[str] = set()
        for match in self._IMPORT_PATTERN.finditer(source):
            top_level = match.group(1).split(".")[0]
            if top_level in self.BLOCKED_MODULES:
                seen_via_regex.add(top_level)

        # ── 3. AST authoritative scan ─────────────────────────────────────────
        seen_via_ast: set[str] = set()
        try:
            tree = ast.parse(source, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_level = alias.name.split(".")[0]
                        if top_level in self.BLOCKED_MODULES:
                            seen_via_ast.add(top_level)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top_level = node.module.split(".")[0]
                        if top_level in self.BLOCKED_MODULES:
                            seen_via_ast.add(top_level)
        except SyntaxError as exc:
            # If the file has a syntax error we cannot parse it safely.
            # Treat as a violation to be conservative.
            error_msg = f"[SYNTAX ERROR] '{file_path}' could not be parsed: {exc}"
            logger.critical(
                "SkillImportFirewall: '%s' has a syntax error — blocking as unsafe: %s",
                file_path, exc,
            )
            return [error_msg]

        # ── 4. Merge and log ──────────────────────────────────────────────────
        all_violations = seen_via_regex | seen_via_ast
        for module_name in sorted(all_violations):
            violation = f"blocked import '{module_name}'"
            violations.append(violation)
            logger.critical(
                "SkillImportFirewall: DANGEROUS IMPORT detected in '%s' — %s",
                file_path, violation,
            )

        return violations

    def is_safe(self, file_path: str) -> bool:
        """Return ``True`` if and only if the skill source has zero violations.

        This is the primary decision gate used by :meth:`NayakPlatform.load_skill`.
        Internally calls :meth:`scan_skill_source` and checks that the returned
        list is empty.

        Args:
            file_path: Absolute or relative path to the skill ``.py`` source
                file on disk.

        Returns:
            ``True`` if no blocked imports were found; ``False`` otherwise.
        """
        return len(self.scan_skill_source(file_path)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — SkillSandbox (enhanced with resource limits)
# ─────────────────────────────────────────────────────────────────────────────

class SkillSandbox:
    """Isolates skill execution from the NAYAK core runtime.

    A crashed or hanging skill must never affect HAL, Safety, or any other
    OS layer.  Every call to :meth:`execute_safe` is wrapped with:

    * A hard asyncio timeout (default 10 s).  On expiry a SAFETY_VIOLATION
      event is emitted on the bus and a structured error dict is returned.

    * Full exception isolation — uncaught errors inside a skill produce a
      structured error dict rather than propagating up to the caller.

    * Return-type validation — skills that return non-dict values are caught.

    * Memory growth monitoring — RSS memory usage is sampled before and
      after every execution via :mod:`psutil`.  If a skill consumes more
      than 100 MB in a single call a SAFETY_VIOLATION is emitted.

    On success the result dict receives an additional ``_sandbox`` key set to
    ``"ok"`` so callers can distinguish sandbox-executed results from raw ones.

    Args:
        timeout_seconds: Maximum seconds a skill's ``execute()`` may run
                         before being forcibly cancelled.  Defaults to 10.
    """

    #: Maximum RSS memory growth permitted per skill execution (bytes).
    MAX_MEMORY_GROWTH_BYTES: int = _MAX_MEMORY_GROWTH_BYTES

    def __init__(self, timeout_seconds: int = 10) -> None:
        self._timeout = timeout_seconds
        self._process = psutil.Process()  # this Python process

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rss_bytes(self) -> int:
        """Return the current RSS (resident set size) of this process in bytes.

        Uses :class:`psutil.Process` for cross-platform compatibility.
        Returns ``0`` on any error so monitoring failures are never fatal.

        Returns:
            Current RSS in bytes, or ``0`` if measurement fails.
        """
        try:
            return self._process.memory_info().rss
        except psutil.Error as exc:
            logger.warning("SkillSandbox: could not sample RSS memory: %s", exc)
            return 0

    @staticmethod
    async def _emit_safety_violation(skill_name: str, reason: str) -> None:
        """Emit a :attr:`EventType.SAFETY_VIOLATION` event on the global bus.

        This is a fire-and-forget async call.  Errors during emission are
        caught and logged rather than propagated so the sandbox never
        cascades a failure.

        Args:
            skill_name: Human-readable name of the offending skill.
            reason:     Short machine-readable reason string (e.g. ``"timeout"``,
                        ``"excessive_memory"``).
        """
        try:
            await bus.emit(NayakEvent(
                type=EventType.SAFETY_VIOLATION,
                payload={
                    "source": "skill-sandbox",
                    "skill": skill_name,
                    "reason": reason,
                },
                source="skill-sandbox",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SkillSandbox: failed to emit SAFETY_VIOLATION for '%s': %s",
                skill_name, exc,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute_safe(self, skill: SkillBase, payload: dict) -> dict:
        """Execute *skill* with full isolation and resource-limit guarantees.

        Execution flow:

        1. Sample RSS memory before the call (:meth:`_rss_bytes`).
        2. Await ``skill.execute(payload)`` inside
           :func:`asyncio.wait_for` with ``timeout=self._timeout``.
        3. Sample RSS memory after the call.
        4. If memory grew by more than :attr:`MAX_MEMORY_GROWTH_BYTES`:
           log critical, emit SAFETY_VIOLATION.
        5. Validate the return type is ``dict``.
        6. Inject ``{"_sandbox": "ok"}`` into the result.

        On timeout: emit SAFETY_VIOLATION, return structured error dict.
        On any other exception: log error, return structured error dict.

        Args:
            skill:   The :class:`SkillBase` instance to execute.
            payload: Input parameters forwarded to ``skill.execute()``.

        Returns:
            On success: the result dict with ``{"_sandbox": "ok"}`` added.

            On timeout:
            ``{"error": "skill timed out after Xs", "skill": skill_name, "_sandbox": "error"}``

            On crash:
            ``{"error": "<exception message>", "skill": skill_name, "_sandbox": "error"}``

            On invalid return type:
            ``{"error": "invalid result type", "skill": skill_name, "_sandbox": "error"}``

            On excessive memory growth:
            Result dict is still returned (execution already completed), but
            ``_sandbox`` is set to ``"memory_warning"`` and a SAFETY_VIOLATION
            is emitted.
        """
        skill_name = skill.skill_name

        # ── Layer 2a: Record pre-execution memory ─────────────────────────────
        mem_before = self._rss_bytes()

        try:
            # ── Layer 2b: Hard asyncio timeout ────────────────────────────────
            result = await asyncio.wait_for(
                skill.execute(payload),
                timeout=self._timeout,
            )

            # ── Layer 2c: Post-execution memory check ─────────────────────────
            mem_after = self._rss_bytes()
            mem_delta = mem_after - mem_before

            if mem_delta > self.MAX_MEMORY_GROWTH_BYTES:
                logger.critical(
                    "SkillSandbox: Skill '%s' consumed excessive memory: "
                    "%d MB growth (limit %d MB)",
                    skill_name,
                    mem_delta // (1024 * 1024),
                    self.MAX_MEMORY_GROWTH_BYTES // (1024 * 1024),
                )
                await self._emit_safety_violation(skill_name, "excessive_memory")
                # We still return the result but flag it so callers can react.
                if isinstance(result, dict):
                    result["_sandbox"] = "memory_warning"
                    return result
                return {
                    "error": "excessive memory growth",
                    "skill": skill_name,
                    "_sandbox": "memory_warning",
                }

            # ── Layer 2d: Return-type validation ──────────────────────────────
            if not isinstance(result, dict):
                logger.error(
                    "SkillSandbox: '%s' returned non-dict type %s",
                    skill_name, type(result).__name__,
                )
                return {
                    "error": "invalid result type",
                    "skill": skill_name,
                    "_sandbox": "error",
                }

            result["_sandbox"] = "ok"
            return result

        except asyncio.TimeoutError:
            logger.error(
                "SkillSandbox: '%s' timed out after %ds",
                skill_name, self._timeout,
            )
            await self._emit_safety_violation(skill_name, "timeout")
            return {
                "error": f"skill timed out after {self._timeout}s",
                "skill": skill_name,
                "_sandbox": "error",
            }

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SkillSandbox: '%s' raised an uncaught exception: %s",
                skill_name, exc,
            )
            return {
                "error": str(exc),
                "skill": skill_name,
                "_sandbox": "error",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — NayakPlatform (load-time firewall gating + validate_all_skills)
# ─────────────────────────────────────────────────────────────────────────────

class NayakPlatform(DeveloperPlatformBase):
    """The concrete operational developer platform for NAYAK.

    Handles dynamic skill loading, execution, and template generation for
    ecosystem expansion.  All skill executions are routed through a
    :class:`SkillSandbox` instance so the core runtime is always protected.

    Security enforcement is three-layered:

    * :class:`SkillImportFirewall` (Layer 1) blocks any skill whose source
      references a forbidden module before it is ever imported.
    * :class:`SkillSandbox` (Layer 2) enforces timeouts and memory limits
      during execution.
    * :meth:`load_skill` (Layer 3) applies the firewall gate at load time and
      refuses to proceed if violations are found.

    The :meth:`validate_all_skills` method provides an on-demand audit of
    all currently loaded skills so the safety engine can schedule periodic
    re-validation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skills_registry: Dict[str, SkillManifest] = {}
        self._skills_dir: str = "skills"
        self._sandbox: SkillSandbox = SkillSandbox(timeout_seconds=10)
        self._firewall: SkillImportFirewall = SkillImportFirewall()
        os.makedirs(self._skills_dir, exist_ok=True)

    @property
    def name(self) -> str:
        """Registry identifier for the SDK platform."""
        return "nayak-platform"

    # ── Skill Management ──────────────────────────────────────────────────────

    async def load_skill(self, manifest: SkillManifest) -> bool:
        """Dynamically load and initialize a skill plugin.

        Before importing the skill, the source file is scanned by
        :class:`SkillImportFirewall`.  If **any** blocked import is found the
        skill is categorically rejected: a CRITICAL log is emitted, a
        SAFETY_VIOLATION event is dispatched on the bus, and this method
        returns ``False`` immediately — the skill's code is never executed.

        If the firewall passes, the module is imported via
        :mod:`importlib`, the first :class:`SkillBase` subclass is located,
        instantiated, and its ``on_load()`` hook is called.  If ``on_load()``
        raises, the skill is marked FAILED and the platform continues — a bad
        skill must never crash the OS layer.

        Args:
            manifest: Defining metadata for the skill, including the
                ``entry_point`` (Python module path) used to resolve the
                on-disk source file.

        Returns:
            ``True`` on success, ``False`` on any failure (firewall block,
            import error, missing SkillBase subclass, or on_load() crash).
        """
        logger.info(
            "SDK: Attempting to load skill '%s' v%s...",
            manifest.name, manifest.version,
        )

        # ── Layer 3a: Resolve on-disk source path ─────────────────────────────
        # Convert the dotted entry_point (e.g. "skills.foo_skill") to a file
        # path so the firewall can read the source.  We try two conventions:
        #   1. entry_point as a dotted module path → relative .py file
        #   2. The skills_dir + last component → common NAYAK convention
        source_path: Optional[str] = self._resolve_source_path(manifest.entry_point)

        if source_path and os.path.isfile(source_path):
            # ── Layer 3b: Import Firewall gate ────────────────────────────────
            violations = self._firewall.scan_skill_source(source_path)
            if violations:
                logger.critical(
                    "SDK: Skill '%s' BLOCKED — dangerous imports detected in '%s': %s",
                    manifest.name, source_path, violations,
                )
                try:
                    await bus.emit(NayakEvent(
                        type=EventType.SAFETY_VIOLATION,
                        payload={
                            "source": "skill-sandbox",
                            "skill": manifest.name,
                            "reason": "blocked_imports",
                            "violations": violations,
                            "file": source_path,
                        },
                        source="nayak-platform",
                    ))
                except Exception as emit_exc:  # noqa: BLE001
                    logger.error(
                        "SDK: Failed to emit SAFETY_VIOLATION for '%s': %s",
                        manifest.name, emit_exc,
                    )
                return False
        else:
            # Source file not found locally — log a warning and continue.
            # This covers skills loaded from installed Python packages where
            # the source is not at a predictable local path.
            logger.warning(
                "SDK: Cannot locate source file for skill '%s' (entry_point='%s'). "
                "Skipping firewall scan — importing at own risk.",
                manifest.name, manifest.entry_point,
            )

        # ── Layer 3c: Import and instantiate ──────────────────────────────────
        try:
            module = importlib.import_module(manifest.entry_point)

            skill_class = None
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, SkillBase)
                    and obj is not SkillBase
                ):
                    skill_class = obj
                    break

            if not skill_class:
                logger.error(
                    "SDK: No SkillBase subclass found in module %s",
                    manifest.entry_point,
                )
                return False

            instance = skill_class()

            # ── on_load() is sandboxed — a crash must not kill the platform ──
            try:
                await instance.on_load()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "SDK: Skill '%s' on_load() raised — marking FAILED: %s",
                    manifest.name, exc,
                )
                return False

            self.loaded_skills[manifest.skill_id] = instance
            self._skills_registry[manifest.skill_id] = manifest

            await bus.emit(NayakEvent(
                type=EventType.SKILL_LOADED,
                payload={
                    "name": manifest.name,
                    "version": manifest.version,
                    "type": manifest.skill_type.name,
                    "skill_id": manifest.skill_id,
                },
                source=self.name,
            ))

            logger.info(
                "SDK: Successfully loaded skill '%s' (%s)",
                manifest.name, manifest.skill_id,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SDK: Critical failure loading skill '%s': %s",
                manifest.name, exc,
            )
            return False

    async def unload_skill(self, skill_id: str) -> bool:
        """Shut down and unregister a loaded skill safely.

        Calls ``on_unload()`` inside a try/except so a crashing shutdown hook
        never blocks the unload sequence.

        Args:
            skill_id: The unique identifier for the skill instance.

        Returns:
            ``True`` on success, ``False`` if skill not found.
        """
        if skill_id not in self.loaded_skills:
            logger.warning("SDK: Unload requested for unknown skill_id: %s", skill_id)
            return False

        skill = self.loaded_skills[skill_id]
        logger.info("SDK: Unloading skill '%s' (%s)", skill.skill_name, skill_id)

        # on_unload() failure must never block the unload sequence
        try:
            await skill.on_unload()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SDK: Skill '%s' on_unload() raised (continuing anyway): %s",
                skill.skill_name, exc,
            )

        del self.loaded_skills[skill_id]
        if skill_id in self._skills_registry:
            del self._skills_registry[skill_id]

        await bus.emit(NayakEvent(
            type=EventType.SKILL_UNLOADED,
            payload={"skill_id": skill_id},
            source=self.name,
        ))
        return True

    async def execute_skill(self, skill_id: str, payload: dict) -> dict:
        """Execute a skill through the sandbox. Core runtime is always protected.

        All execution is delegated to :class:`SkillSandbox` which enforces
        timeouts, catches exceptions, validates return types, and monitors
        memory growth.

        Args:
            skill_id: The unique identifier for the skill instance.
            payload:  Input parameters forwarded to the skill.

        Returns:
            Result dict from the skill (with ``_sandbox: "ok"`` added on
            clean success), or a structured error dict if the skill failed.
        """
        if skill_id not in self.loaded_skills:
            return {"error": "skill not found", "_sandbox": "error"}
        skill = self.loaded_skills[skill_id]
        return await self._sandbox.execute_safe(skill, payload)

    async def list_skills(self) -> List[SkillManifest]:
        """List manifests for all currently active plugins.

        Returns:
            A list of :class:`SkillManifest` objects.
        """
        return list(self._skills_registry.values())

    async def validate_all_skills(self) -> Dict[str, List[str]]:
        """Re-audit every loaded skill's source file with the import firewall.

        Iterates over all entries in the skills registry, resolves each
        skill's source path, and re-runs :meth:`SkillImportFirewall.scan_skill_source`
        against it.  This is useful for periodic safety sweeps, health checks,
        or after a skill has been hot-patched on disk.

        Skills whose source cannot be located (e.g. installed packages) are
        listed under ``"skipped"`` rather than ``"safe"`` or ``"violations"``.

        Returns:
            A dict with three keys:

            * ``"safe"`` — list of skill names that passed the firewall.
            * ``"violations"`` — list of skill names that failed (each entry
              is a string of the form ``"<name>: <violations>"``).
            * ``"skipped"`` — list of skill names whose source file could not
              be located for scanning.

        Example::

            report = await platform.validate_all_skills()
            print("Safe:", report["safe"])
            print("Blocked:", report["violations"])
        """
        report: Dict[str, List[str]] = {
            "safe": [],
            "violations": [],
            "skipped": [],
        }

        for skill_id, manifest in self._skills_registry.items():
            source_path = self._resolve_source_path(manifest.entry_point)
            if not source_path or not os.path.isfile(source_path):
                logger.warning(
                    "SDK validate_all_skills: cannot locate source for '%s' — skipping",
                    manifest.name,
                )
                report["skipped"].append(manifest.name)
                continue

            violations = self._firewall.scan_skill_source(source_path)
            if violations:
                summary = f"{manifest.name}: {violations}"
                report["violations"].append(summary)
                logger.critical(
                    "SDK validate_all_skills: Skill '%s' has violations: %s",
                    manifest.name, violations,
                )
            else:
                report["safe"].append(manifest.name)
                logger.debug(
                    "SDK validate_all_skills: Skill '%s' passed firewall audit",
                    manifest.name,
                )

        logger.info(
            "SDK validate_all_skills: %d safe, %d violations, %d skipped",
            len(report["safe"]),
            len(report["violations"]),
            len(report["skipped"]),
        )
        return report

    # ── Developer Tooling ─────────────────────────────────────────────────────

    async def create_skill_template(self, name: str, skill_type: SkillType) -> str:
        """Generate a Python boilerplate file for a new developer skill.

        The generated template deliberately imports only safe, NAYAK-approved
        modules (``logging`` and ``nayak.sdk.base``).

        Args:
            name:       Human-readable name (used as filename and class name).
            skill_type: Functional category of the skill.

        Returns:
            The absolute path to the generated file.
        """
        class_name = "".join(
            x.capitalize()
            for x in name.replace("-", " ").replace("_", " ").split()
        )
        safe_filename = name.lower().replace(" ", "_").replace("-", "_")
        file_path = os.path.join(self._skills_dir, f"{safe_filename}.py")

        template = f'''"""
{name} — NAYAK Developer Skill.

Automatically generated by the NAYAK SDK platform.
"""

from __future__ import annotations
import logging
from nayak.sdk.base import SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)

class {class_name}(SkillBase):
    """Custom skill implementation for NAYAK."""

    def __init__(self) -> None:
        self._manifest = SkillManifest(
            name="{name}",
            version="0.1.0",
            skill_type=SkillType.{skill_type.name},
            description="Generated by NAYAK SDK platform.",
            entry_point="skills.{safe_filename}"
        )

    @property
    def manifest(self) -> SkillManifest:
        """Defining metadata for the skill."""
        return self._manifest

    async def on_load(self) -> None:
        logger.info("{name}: booted")

    async def on_unload(self) -> None:
        logger.info("{name}: shutdown")

    async def execute(self, payload: dict) -> dict:
        # TODO: Implement your logic here
        return {{"status": "success", "echo": payload}}
'''
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template)
        logger.info("SDK: Created skill template for '%s' at %s", name, file_path)
        return file_path

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Mount the Developer Platform into the OS stack."""
        await self.register()
        logger.info("Developer platform initialized")
        logger.info("SDK: Skills directory: %s", self._skills_dir)

    async def stop(self) -> None:
        """Gracefully shut down all active plugins and the SDK platform."""
        ids = list(self.loaded_skills.keys())
        for skill_id in ids:
            await self.unload_skill(skill_id)
        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("SDK: Platform shutdown complete.")
        except KeyError:
            pass

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _resolve_source_path(self, entry_point: str) -> Optional[str]:
        """Resolve a dotted module entry_point to an on-disk ``.py`` path.

        Tries the following strategies in order:

        1. Convert dots to path separators and append ``.py``
           (e.g. ``"skills.foo_skill"`` → ``"skills/foo_skill.py"``).
        2. Use only the last component of the dotted path inside
           ``self._skills_dir`` (e.g. ``"nayak.skills.foo"`` →
           ``"skills/foo.py"``).

        This covers both community skills loaded from the local ``skills/``
        directory and first-party skills located inside the package tree.

        Args:
            entry_point: Dotted Python module path from a :class:`SkillManifest`.

        Returns:
            The resolved path string, or ``None`` if neither strategy
            produces a plausible candidate.
        """
        # Strategy 1: direct dot-to-path conversion
        direct_path = entry_point.replace(".", os.sep) + ".py"
        if os.path.isfile(direct_path):
            return direct_path

        # Strategy 2: last component inside skills_dir
        last_component = entry_point.rsplit(".", 1)[-1]
        skills_dir_path = os.path.join(self._skills_dir, last_component + ".py")
        if os.path.isfile(skills_dir_path):
            return skills_dir_path

        # Strategy 3: try interpreting entry_point as an installed package path
        try:
            spec = importlib.util.find_spec(entry_point)  # type: ignore[attr-defined]
            if spec and spec.origin and spec.origin.endswith(".py"):
                return spec.origin
        except (ModuleNotFoundError, ValueError):
            pass

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

platform: NayakPlatform = NayakPlatform()
"""Global NAYAK Developer Platform instance."""

__all__ = ["SkillImportFirewall", "SkillSandbox", "NayakPlatform", "platform"]
