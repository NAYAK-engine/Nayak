"""
tests/test_gpio_hal.py — Unit tests for nayak.hal.gpio.GpioHAL.

Tests the GPIO HAL backend in simulated mode (no real Pi hardware required).
Verifies:
  - Auto-detection defaults to simulated mode on non-Pi hosts.
  - Connect / disconnect lifecycle transitions.
  - Read returns shadow register values.
  - Write updates shadow register and device metadata.
  - Pin number extraction from device_id strings.
  - Cleanup resets all state.
  - hardware mode property reports correctly.
"""

from __future__ import annotations

import asyncio
import pytest

from nayak.hal.gpio import GpioHAL


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def gpio() -> GpioHAL:
    """Create a fresh GpioHAL instance for each test.

    Returns a new instance (not the module-level singleton) and clears the
    class-level devices dict to ensure full test isolation.
    """
    hal = GpioHAL()
    hal.devices.clear()
    hal._pin_values.clear()
    hal._pin_directions.clear()
    return hal


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoDetection:
    """Verify that hardware auto-detection works on non-Pi hosts."""

    def test_simulated_mode_on_dev_machine(self, gpio: GpioHAL) -> None:
        """On a dev machine, the GPIO backend must enter simulated mode."""
        assert gpio.on_pi is False
        assert gpio._gpio is None

    def test_name_property(self, gpio: GpioHAL) -> None:
        """The name property returns the expected registry identifier."""
        assert gpio.name == "gpio-hal"


# ─────────────────────────────────────────────────────────────────────────────
# Connect / Disconnect lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycle:
    """Test device connect / disconnect transitions."""

    @pytest.mark.asyncio
    async def test_connect_creates_device_entry(self, gpio: GpioHAL) -> None:
        """Connecting a device must create a DeviceInfo entry."""
        result = await gpio.connect("gpio_17")
        assert result is True
        assert "gpio_17" in gpio.devices
        device = gpio.devices["gpio_17"]
        assert device.metadata["simulated"] is True
        assert device.metadata["gpio_available"] is False

    @pytest.mark.asyncio
    async def test_disconnect_sets_status(self, gpio: GpioHAL) -> None:
        """Disconnecting must set the device status to DISCONNECTED."""
        await gpio.connect("gpio_17")
        result = await gpio.disconnect("gpio_17")
        assert result is True

        from nayak.hal.base import DeviceStatus
        assert gpio.devices["gpio_17"].status == DeviceStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_unknown_device(self, gpio: GpioHAL) -> None:
        """Disconnecting an unknown device must return True (safe no-op)."""
        result = await gpio.disconnect("nonexistent")
        assert result is True

    @pytest.mark.asyncio
    async def test_list_devices(self, gpio: GpioHAL) -> None:
        """list_devices returns all connected devices."""
        await gpio.connect("gpio_17")
        await gpio.connect("gpio_27")
        devices = await gpio.list_devices()
        assert len(devices) == 2
        ids = {d.device_id for d in devices}
        assert ids == {"gpio_17", "gpio_27"}


# ─────────────────────────────────────────────────────────────────────────────
# Read / Write in simulated mode
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatedIO:
    """Test GPIO read/write operations in simulated mode."""

    @pytest.mark.asyncio
    async def test_read_default_value(self, gpio: GpioHAL) -> None:
        """Reading a pin that was never written must return 0."""
        await gpio.connect("gpio_17")
        data = await gpio.read("gpio_17")
        assert data is not None
        assert data["value"] == 0
        assert data["simulated"] is True
        assert data["pin"] == 17

    @pytest.mark.asyncio
    async def test_write_then_read(self, gpio: GpioHAL) -> None:
        """Writing a value must be reflected in subsequent reads."""
        await gpio.connect("gpio_22")
        ok = await gpio.write("gpio_22", {"pin": 22, "value": 1})
        assert ok is True

        data = await gpio.read("gpio_22")
        assert data is not None
        assert data["value"] == 1
        assert data["pin"] == 22

    @pytest.mark.asyncio
    async def test_write_updates_shadow_register(self, gpio: GpioHAL) -> None:
        """Write must update the internal shadow register."""
        await gpio.connect("gpio_4")
        await gpio.write("gpio_4", {"pin": 4, "value": 1})
        assert gpio._pin_values[4] == 1

        await gpio.write("gpio_4", {"pin": 4, "value": 0})
        assert gpio._pin_values[4] == 0

    @pytest.mark.asyncio
    async def test_write_without_pin_uses_device_id(
        self, gpio: GpioHAL,
    ) -> None:
        """If 'pin' is missing from data, it must be extracted from device_id."""
        await gpio.connect("gpio_27")
        ok = await gpio.write("gpio_27", {"value": 1})
        assert ok is True
        assert gpio._pin_values[27] == 1

    @pytest.mark.asyncio
    async def test_write_invalid_data_type(self, gpio: GpioHAL) -> None:
        """Writing non-dict data must return False."""
        await gpio.connect("gpio_17")
        ok = await gpio.write("gpio_17", 42)
        assert ok is False

    @pytest.mark.asyncio
    async def test_read_not_connected(self, gpio: GpioHAL) -> None:
        """Reading a device that is not connected must return None."""
        data = await gpio.read("gpio_17")
        assert data is None

    @pytest.mark.asyncio
    async def test_write_not_connected(self, gpio: GpioHAL) -> None:
        """Writing to a device that is not connected must return False."""
        ok = await gpio.write("gpio_17", {"pin": 17, "value": 1})
        assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
# Pin extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestPinExtraction:
    """Test pin number extraction from device identifier strings."""

    def test_extract_from_gpio_prefix(self, gpio: GpioHAL) -> None:
        """'gpio_17' should extract pin 17."""
        assert gpio._extract_pin_number("gpio_17") == 17

    def test_extract_from_pin_prefix(self, gpio: GpioHAL) -> None:
        """'pin_4' should extract pin 4."""
        assert gpio._extract_pin_number("pin_4") == 4

    def test_extract_from_led_prefix(self, gpio: GpioHAL) -> None:
        """'led_13' should extract pin 13."""
        assert gpio._extract_pin_number("led_13") == 13

    def test_extract_fallback(self, gpio: GpioHAL) -> None:
        """A device_id with no digits should return 0."""
        assert gpio._extract_pin_number("unknown") == 0


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanup:
    """Test cleanup resets internal state."""

    @pytest.mark.asyncio
    async def test_cleanup_clears_state(self, gpio: GpioHAL) -> None:
        """cleanup() must clear pin values and directions."""
        await gpio.connect("gpio_17")
        await gpio.write("gpio_17", {"pin": 17, "value": 1})
        assert len(gpio._pin_values) > 0

        gpio.cleanup()
        assert len(gpio._pin_values) == 0
        assert len(gpio._pin_directions) == 0
