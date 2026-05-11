"""
nayak.core — The NAYAK runtime core.

Contains the fundamental building blocks that all layers depend on:
- EventBus / bus:       async pub/sub backbone for control signals
- SharedMemoryBus / shm_bus: zero-copy channels for heavy payloads (frames, tensors)
- ModuleRegistry:       tracks all loaded NAYAK layers and modules
- NayakRuntime:         boots and manages the full NAYAK OS lifecycle
"""

from nayak.core.bus import (
    EventBus, EventType, NayakEvent, bus,
    PayloadType, SharedMemoryChannel, SharedMemoryBus, shm_bus,
)
from nayak.core.registry import ModuleRegistry, ModuleStatus, NayakModule, registry
from nayak.core.runtime import NayakRuntime, RuntimeConfig, runtime

__all__ = [
    # EventBus
    "EventBus", "EventType", "NayakEvent", "bus",
    # SharedMemoryBus
    "PayloadType", "SharedMemoryChannel", "SharedMemoryBus", "shm_bus",
    # Registry
    "ModuleRegistry", "ModuleStatus", "NayakModule", "registry",
    # Runtime
    "NayakRuntime", "RuntimeConfig", "runtime",
]
