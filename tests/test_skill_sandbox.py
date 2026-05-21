"""
tests/test_skill_sandbox.py — Unit tests for the three-layer NAYAK skill
security architecture.

Tests are fully self-contained and use local mocks — no external network
calls, no real psutil process memory sampling where inappropriate.

Layers under test
-----------------
Layer 1 — SkillImportFirewall:
    scan_skill_source(), is_safe()

Layer 2 — SkillSandbox:
    execute_safe() — timeout, memory overrun, exception isolation,
    return-type validation, SAFETY_VIOLATION bus event emission.

Layer 3 — NayakPlatform.load_skill / validate_all_skills:
    Firewall gate blocks dangerous skills before import.
    validate_all_skills() returns correct safe/violations/skipped buckets.
"""

from __future__ import annotations

import asyncio
import textwrap
import tempfile
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ── Subjects under test ───────────────────────────────────────────────────────
from nayak.sdk.platform import SkillImportFirewall, SkillSandbox, NayakPlatform
from nayak.sdk.base import SkillBase, SkillManifest, SkillType
from nayak.core.bus import EventType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — minimal in-memory SkillBase implementations
# ─────────────────────────────────────────────────────────────────────────────

def _make_manifest(name: str = "test-skill") -> SkillManifest:
    """Return a minimal valid SkillManifest for testing."""
    return SkillManifest(
        name=name,
        version="0.1.0",
        skill_type=SkillType.UTILITY,
        description="Test skill",
        entry_point=f"skills.{name.replace('-', '_')}",
    )


class GoodSkill(SkillBase):
    """A well-behaved skill that returns a dict immediately."""

    def __init__(self) -> None:
        self._manifest = _make_manifest("good-skill")

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    async def on_load(self) -> None:
        pass

    async def on_unload(self) -> None:
        pass

    async def execute(self, payload: dict) -> dict:
        return {"status": "ok", "echo": payload}


class TimeoutSkill(SkillBase):
    """A skill that sleeps forever (will trigger timeout)."""

    def __init__(self) -> None:
        self._manifest = _make_manifest("timeout-skill")

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    async def on_load(self) -> None:
        pass

    async def on_unload(self) -> None:
        pass

    async def execute(self, payload: dict) -> dict:
        await asyncio.sleep(9999)
        return {}


class CrashingSkill(SkillBase):
    """A skill that raises a RuntimeError on every execute()."""

    def __init__(self) -> None:
        self._manifest = _make_manifest("crashing-skill")

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    async def on_load(self) -> None:
        pass

    async def on_unload(self) -> None:
        pass

    async def execute(self, payload: dict) -> dict:  # type: ignore[override]
        raise RuntimeError("skill internal error")


class BadReturnSkill(SkillBase):
    """A skill that returns a non-dict value."""

    def __init__(self) -> None:
        self._manifest = _make_manifest("bad-return-skill")

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    async def on_load(self) -> None:
        pass

    async def on_unload(self) -> None:
        pass

    async def execute(self, payload: dict) -> dict:  # type: ignore[override]
        return "this is a string, not a dict"  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 tests — SkillImportFirewall
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillImportFirewall(unittest.TestCase):
    """Tests for :class:`SkillImportFirewall`."""

    def setUp(self) -> None:
        self.fw = SkillImportFirewall()

    # ── scan_skill_source ─────────────────────────────────────────────────────

    def _write_temp(self, source: str) -> str:
        """Write *source* to a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        tmp.write(textwrap.dedent(source))
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_clean_skill_passes(self) -> None:
        path = self._write_temp("""
            import logging
            from nayak.sdk.base import SkillBase
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertEqual(violations, [])
        finally:
            os.unlink(path)

    def test_direct_os_import_blocked(self) -> None:
        path = self._write_temp("""
            import os
            import logging
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertTrue(any("os" in v for v in violations))
        finally:
            os.unlink(path)

    def test_from_os_import_blocked(self) -> None:
        path = self._write_temp("""
            from os import path
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertTrue(any("os" in v for v in violations))
        finally:
            os.unlink(path)

    def test_subprocess_blocked(self) -> None:
        path = self._write_temp("""
            import subprocess
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertTrue(any("subprocess" in v for v in violations))
        finally:
            os.unlink(path)

    def test_sys_blocked(self) -> None:
        path = self._write_temp("""
            import sys
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertTrue(any("sys" in v for v in violations))
        finally:
            os.unlink(path)

    def test_ctypes_blocked(self) -> None:
        path = self._write_temp("""
            import ctypes
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertTrue(any("ctypes" in v for v in violations))
        finally:
            os.unlink(path)

    def test_multiple_violations_all_reported(self) -> None:
        path = self._write_temp("""
            import os
            import subprocess
            import sys
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertGreaterEqual(len(violations), 3)
        finally:
            os.unlink(path)

    def test_missing_file_returns_read_error(self) -> None:
        violations = self.fw.scan_skill_source("/nonexistent/path/evil_skill.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("READ ERROR", violations[0])

    def test_syntax_error_file_blocked_conservatively(self) -> None:
        path = self._write_temp("""
            def broken(:
                pass
        """)
        try:
            violations = self.fw.scan_skill_source(path)
            self.assertEqual(len(violations), 1)
            self.assertIn("SYNTAX ERROR", violations[0])
        finally:
            os.unlink(path)

    # ── is_safe ───────────────────────────────────────────────────────────────

    def test_is_safe_returns_true_for_clean_file(self) -> None:
        path = self._write_temp("""
            import logging
        """)
        try:
            self.assertTrue(self.fw.is_safe(path))
        finally:
            os.unlink(path)

    def test_is_safe_returns_false_for_blocked_import(self) -> None:
        path = self._write_temp("""
            import os
        """)
        try:
            self.assertFalse(self.fw.is_safe(path))
        finally:
            os.unlink(path)

    def test_is_safe_returns_false_for_missing_file(self) -> None:
        self.assertFalse(self.fw.is_safe("/does/not/exist.py"))

    def test_blocked_modules_constant_is_frozenset(self) -> None:
        self.assertIsInstance(SkillImportFirewall.BLOCKED_MODULES, frozenset)

    def test_all_required_modules_are_blocked(self) -> None:
        required = {
            "os", "subprocess", "shutil", "sys",
            "socket", "ctypes", "importlib",
            "multiprocessing", "threading", "__builtins__",
        }
        self.assertTrue(required.issubset(SkillImportFirewall.BLOCKED_MODULES))


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 tests — SkillSandbox
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillSandbox(unittest.IsolatedAsyncioTestCase):
    """Tests for :class:`SkillSandbox`."""

    def setUp(self) -> None:
        # Use a very short timeout so tests run fast
        self.sandbox = SkillSandbox(timeout_seconds=1)

    # ── Happy path ────────────────────────────────────────────────────────────

    async def test_good_skill_returns_ok_sandbox_marker(self) -> None:
        skill = GoodSkill()
        result = await self.sandbox.execute_safe(skill, {"x": 1})
        self.assertEqual(result["_sandbox"], "ok")
        self.assertEqual(result["status"], "ok")

    async def test_good_skill_echoes_payload(self) -> None:
        skill = GoodSkill()
        result = await self.sandbox.execute_safe(skill, {"hello": "world"})
        self.assertEqual(result["echo"], {"hello": "world"})

    # ── Timeout ───────────────────────────────────────────────────────────────

    async def test_timeout_returns_error_dict(self) -> None:
        skill = TimeoutSkill()
        result = await self.sandbox.execute_safe(skill, {})
        self.assertIn("timed out", result["error"])
        self.assertEqual(result["skill"], "timeout-skill")
        self.assertEqual(result["_sandbox"], "error")

    async def test_timeout_emits_safety_violation(self) -> None:
        emitted_events: list = []

        async def capture(event):
            emitted_events.append(event)

        from nayak.core.bus import bus, EventType
        bus.subscribe(EventType.SAFETY_VIOLATION, capture)
        try:
            skill = TimeoutSkill()
            await self.sandbox.execute_safe(skill, {})
            self.assertTrue(
                any(e.payload.get("reason") == "timeout" for e in emitted_events),
                "Expected a SAFETY_VIOLATION with reason='timeout' to be emitted",
            )
        finally:
            bus.unsubscribe(EventType.SAFETY_VIOLATION, capture)

    async def test_timeout_violation_payload_has_skill_name(self) -> None:
        emitted_events: list = []

        async def capture(event):
            emitted_events.append(event)

        from nayak.core.bus import bus, EventType
        bus.subscribe(EventType.SAFETY_VIOLATION, capture)
        try:
            skill = TimeoutSkill()
            await self.sandbox.execute_safe(skill, {})
            violation = next(
                (e for e in emitted_events if e.payload.get("reason") == "timeout"),
                None,
            )
            self.assertIsNotNone(violation)
            self.assertEqual(violation.payload["skill"], "timeout-skill")
            self.assertEqual(violation.payload["source"], "skill-sandbox")
        finally:
            bus.unsubscribe(EventType.SAFETY_VIOLATION, capture)

    # ── Exception isolation ───────────────────────────────────────────────────

    async def test_crashing_skill_returns_error_dict(self) -> None:
        skill = CrashingSkill()
        result = await self.sandbox.execute_safe(skill, {})
        self.assertIn("skill internal error", result["error"])
        self.assertEqual(result["skill"], "crashing-skill")
        self.assertEqual(result["_sandbox"], "error")

    async def test_crashing_skill_does_not_raise(self) -> None:
        """The sandbox must never let an exception propagate to the caller."""
        skill = CrashingSkill()
        try:
            await self.sandbox.execute_safe(skill, {})
        except Exception as exc:
            self.fail(f"execute_safe() propagated an exception: {exc}")

    # ── Return-type validation ────────────────────────────────────────────────

    async def test_bad_return_type_returns_error_dict(self) -> None:
        skill = BadReturnSkill()
        result = await self.sandbox.execute_safe(skill, {})
        self.assertEqual(result["error"], "invalid result type")
        self.assertEqual(result["skill"], "bad-return-skill")
        self.assertEqual(result["_sandbox"], "error")

    # ── Memory limit ──────────────────────────────────────────────────────────

    async def test_excessive_memory_emits_safety_violation(self) -> None:
        """Simulate a skill that causes 200 MB of memory growth."""
        emitted_events: list = []

        async def capture(event):
            emitted_events.append(event)

        from nayak.core.bus import bus, EventType

        # Patch _rss_bytes to simulate memory growth > 100 MB
        with patch.object(
            self.sandbox, "_rss_bytes",
            side_effect=[
                100 * 1024 * 1024,       # before: 100 MB
                300 * 1024 * 1024,       # after: 300 MB  (200 MB growth)
            ],
        ):
            bus.subscribe(EventType.SAFETY_VIOLATION, capture)
            try:
                skill = GoodSkill()
                result = await self.sandbox.execute_safe(skill, {})
                # Skill completed but sandbox flagged the memory abuse
                self.assertEqual(result["_sandbox"], "memory_warning")
                self.assertTrue(
                    any(
                        e.payload.get("reason") == "excessive_memory"
                        for e in emitted_events
                    ),
                    "Expected a SAFETY_VIOLATION with reason='excessive_memory'",
                )
            finally:
                bus.unsubscribe(EventType.SAFETY_VIOLATION, capture)

    async def test_normal_memory_growth_does_not_violate(self) -> None:
        """Small memory delta must not trigger any violation."""
        emitted_events: list = []

        async def capture(event):
            emitted_events.append(event)

        from nayak.core.bus import bus, EventType

        with patch.object(
            self.sandbox, "_rss_bytes",
            side_effect=[
                100 * 1024 * 1024,    # before
                101 * 1024 * 1024,    # after: only 1 MB growth
            ],
        ):
            bus.subscribe(EventType.SAFETY_VIOLATION, capture)
            try:
                skill = GoodSkill()
                result = await self.sandbox.execute_safe(skill, {})
                self.assertEqual(result["_sandbox"], "ok")
                self.assertEqual(emitted_events, [])
            finally:
                bus.unsubscribe(EventType.SAFETY_VIOLATION, capture)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 tests — NayakPlatform.load_skill + validate_all_skills
# ─────────────────────────────────────────────────────────────────────────────

class TestNayakPlatformFirewallGate(unittest.IsolatedAsyncioTestCase):
    """Tests for the Layer 3 firewall gate in :meth:`NayakPlatform.load_skill`
    and :meth:`NayakPlatform.validate_all_skills`."""

    def _make_platform(self) -> NayakPlatform:
        """Return a fresh NayakPlatform with registry/bus side-effects mocked."""
        p = NayakPlatform.__new__(NayakPlatform)
        # Initialise parent manually to avoid real registry calls
        p.loaded_skills = {}
        p._skills_registry = {}
        p._skills_dir = "skills"
        p._sandbox = SkillSandbox(timeout_seconds=10)
        p._firewall = SkillImportFirewall()
        return p

    def _write_safe_skill(self, tmpdir: str, name: str = "safe_skill") -> str:
        path = os.path.join(tmpdir, f"{name}.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent("""
                import logging
                from nayak.sdk.base import SkillBase, SkillManifest, SkillType
            """))
        return path

    def _write_evil_skill(self, tmpdir: str, name: str = "evil_skill") -> str:
        path = os.path.join(tmpdir, f"{name}.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent("""
                import os
                import subprocess
                import sys
            """))
        return path

    # ── load_skill firewall gate ──────────────────────────────────────────────

    async def test_load_skill_blocked_when_violations_found(self) -> None:
        platform = self._make_platform()
        with tempfile.TemporaryDirectory() as tmpdir:
            evil_path = self._write_evil_skill(tmpdir)
            manifest = _make_manifest("evil-skill")

            # Point _resolve_source_path at our temp evil file
            with patch.object(platform, "_resolve_source_path", return_value=evil_path):
                result = await platform.load_skill(manifest)

            self.assertFalse(result, "Skill with blocked imports must not be loaded")

    async def test_load_skill_blocked_emits_safety_violation(self) -> None:
        emitted: list = []

        async def capture(event):
            emitted.append(event)

        from nayak.core.bus import bus, EventType
        bus.subscribe(EventType.SAFETY_VIOLATION, capture)

        platform = self._make_platform()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                evil_path = self._write_evil_skill(tmpdir)
                manifest = _make_manifest("evil-skill-2")

                with patch.object(platform, "_resolve_source_path", return_value=evil_path):
                    await platform.load_skill(manifest)

            self.assertTrue(
                any(e.payload.get("reason") == "blocked_imports" for e in emitted),
                "SAFETY_VIOLATION with reason='blocked_imports' must be emitted",
            )
        finally:
            bus.unsubscribe(EventType.SAFETY_VIOLATION, capture)

    async def test_load_skill_proceeds_when_safe_source(self) -> None:
        """A safe skill source should still be loaded normally (mocking importlib)."""
        platform = self._make_platform()

        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = self._write_safe_skill(tmpdir)
            manifest = _make_manifest("safe-skill")

            # Firewall passes but importlib.import_module won't find the temp file —
            # we mock it to return a module with our GoodSkill class.
            mock_module = MagicMock()
            mock_module.__dict__["GoodSkill"] = GoodSkill

            with patch.object(platform, "_resolve_source_path", return_value=safe_path):
                with patch("importlib.import_module", return_value=mock_module):
                    with patch("inspect.getmembers", return_value=[("GoodSkill", GoodSkill)]):
                        # bus.emit also needs to work; patch it to avoid registry errors
                        with patch("nayak.sdk.platform.bus") as mock_bus:
                            mock_bus.emit = AsyncMock()
                            result = await platform.load_skill(manifest)

            self.assertTrue(result, "A clean skill should load successfully")

    # ── validate_all_skills ───────────────────────────────────────────────────

    async def test_validate_all_skills_safe_bucket(self) -> None:
        platform = self._make_platform()

        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = self._write_safe_skill(tmpdir, "safe_one")
            manifest = _make_manifest("safe-one")
            platform._skills_registry[manifest.skill_id] = manifest

            with patch.object(platform, "_resolve_source_path", return_value=safe_path):
                report = await platform.validate_all_skills()

        self.assertIn("safe-one", report["safe"])
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["skipped"], [])

    async def test_validate_all_skills_violations_bucket(self) -> None:
        platform = self._make_platform()

        with tempfile.TemporaryDirectory() as tmpdir:
            evil_path = self._write_evil_skill(tmpdir, "evil_two")
            manifest = _make_manifest("evil-two")
            platform._skills_registry[manifest.skill_id] = manifest

            with patch.object(platform, "_resolve_source_path", return_value=evil_path):
                report = await platform.validate_all_skills()

        self.assertTrue(
            any("evil-two" in v for v in report["violations"]),
            f"Expected 'evil-two' in violations, got: {report['violations']}",
        )
        self.assertEqual(report["safe"], [])

    async def test_validate_all_skills_skipped_when_no_source(self) -> None:
        platform = self._make_platform()
        manifest = _make_manifest("mystery-skill")
        platform._skills_registry[manifest.skill_id] = manifest

        with patch.object(platform, "_resolve_source_path", return_value=None):
            report = await platform.validate_all_skills()

        self.assertIn("mystery-skill", report["skipped"])
        self.assertEqual(report["safe"], [])
        self.assertEqual(report["violations"], [])

    async def test_validate_all_skills_mixed(self) -> None:
        platform = self._make_platform()

        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = self._write_safe_skill(tmpdir, "clean")
            evil_path = self._write_evil_skill(tmpdir, "dirty")

            m_safe = _make_manifest("clean-skill")
            m_evil = _make_manifest("dirty-skill")
            platform._skills_registry[m_safe.skill_id] = m_safe
            platform._skills_registry[m_evil.skill_id] = m_evil

            def resolve(ep):
                if "clean" in ep:
                    return safe_path
                return evil_path

            with patch.object(platform, "_resolve_source_path", side_effect=resolve):
                report = await platform.validate_all_skills()

        self.assertIn("clean-skill", report["safe"])
        self.assertTrue(any("dirty-skill" in v for v in report["violations"]))

    async def test_validate_all_skills_empty_registry(self) -> None:
        platform = self._make_platform()
        report = await platform.validate_all_skills()
        self.assertEqual(report, {"safe": [], "violations": [], "skipped": []})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
