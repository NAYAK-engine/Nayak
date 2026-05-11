"""
tests/test_registry.py — Unit tests for the NAYAK Module Registry.

Validates registration, unregistration, layer queries, status mutations,
listing, and the summary display on a fresh ModuleRegistry per test.
"""

import pytest

from nayak.core.registry import ModuleRegistry, ModuleStatus, NayakModule


class TestModuleRegistry:
    """Tests for the core module registry."""

    def setup_method(self) -> None:
        """Create a fresh ModuleRegistry for every test."""
        self.registry = ModuleRegistry()

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_register_module(self) -> None:
        """A registered module should be retrievable by name."""
        module = NayakModule(
            name="test-module",
            version="1.0.0",
            layer=3,
            description="Test module",
        )
        await self.registry.register(module)

        result = self.registry.get("test-module")
        assert result is not None
        assert result.name == "test-module"
        assert result.version == "1.0.0"
        assert result.layer == 3

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unregister_module(self) -> None:
        """After unregistering, get() should return None."""
        module = NayakModule(
            name="temp-module",
            version="0.1.0",
            layer=5,
        )
        await self.registry.register(module)
        await self.registry.unregister("temp-module")

        assert self.registry.get("temp-module") is None

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_by_layer(self) -> None:
        """get_by_layer() should return only modules on the requested layer."""
        # 3 modules on layer 3
        for i in range(3):
            await self.registry.register(NayakModule(
                name=f"layer3-mod-{i}",
                version="1.0.0",
                layer=3,
            ))

        # 2 modules on layer 1
        for i in range(2):
            await self.registry.register(NayakModule(
                name=f"layer1-mod-{i}",
                version="1.0.0",
                layer=1,
            ))

        assert len(self.registry.get_by_layer(3)) == 3
        assert len(self.registry.get_by_layer(1)) == 2
        assert len(self.registry.get_by_layer(7)) == 0

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_set_status(self) -> None:
        """set_status() should update the module's lifecycle status."""
        module = NayakModule(
            name="status-test",
            version="1.0.0",
            layer=2,
        )
        await self.registry.register(module)

        self.registry.set_status("status-test", ModuleStatus.READY)
        result = self.registry.get("status-test")
        assert result is not None
        assert result.status == ModuleStatus.READY

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        """list_all() should return every registered module."""
        for i in range(5):
            await self.registry.register(NayakModule(
                name=f"mod-{i}",
                version="1.0.0",
                layer=(i % 9) + 1,
            ))

        assert len(self.registry.list_all()) == 5

    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summary_returns_string(self) -> None:
        """summary() should return a non-empty formatted string."""
        await self.registry.register(NayakModule(
            name="summary-mod-a",
            version="1.0.0",
            layer=1,
            description="First module",
        ))
        await self.registry.register(NayakModule(
            name="summary-mod-b",
            version="2.0.0",
            layer=4,
            description="Second module",
        ))

        result = self.registry.summary()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "summary-mod-a" in result
        assert "summary-mod-b" in result
