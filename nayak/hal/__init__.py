"""
nayak.hal — NAYAK Layer 1: Hardware Abstraction Layer.

The foundation of NAYAK's physical world interface.
Every robot. Every sensor. Every motor. Every camera.
All hardware talks to NAYAK through this layer.
Write once. Run on any robot.
"""

from nayak.hal.base import HardwareBase, DeviceType, DeviceStatus, DeviceInfo
from nayak.hal.raspberry_pi import RaspberryPiHAL, raspberry_pi
from nayak.hal.camera import CameraHAL, camera

__all__ = [
    "HardwareBase", "DeviceType", "DeviceStatus", "DeviceInfo",
    "RaspberryPiHAL", "raspberry_pi",
    "CameraHAL", "camera",
]
