"""
nayak.core — The NAYAK runtime core.

Contains the fundamental building blocks that all layers depend on:
- EventBus: async pub/sub backbone for inter-layer communication
- ModuleRegistry: tracks all loaded NAYAK layers and modules
- NayakRuntime: boots and manages the full NAYAK OS lifecycle
"""

from nayak.core.bus import EventBus, EventType, NayakEvent, bus
from nayak.core.registry import ModuleRegistry, ModuleStatus, NayakModule, registry
from nayak.core.runtime import NayakRuntime, RuntimeConfig, runtime

__all__ = [
    "EventBus", "EventType", "NayakEvent", "bus",
    "ModuleRegistry", "ModuleStatus", "NayakModule", "registry",
    "NayakRuntime", "RuntimeConfig", "runtime",
]
