"""
nayak.core — The NAYAK runtime core.

Contains the fundamental building blocks that all layers depend on:
- EventBus: async pub/sub backbone for inter-layer communication
- ModuleRegistry: tracks all loaded NAYAK layers and modules
"""

from nayak.core.bus import EventBus, EventType, NayakEvent, bus
from nayak.core.registry import ModuleRegistry, ModuleStatus, NayakModule, registry

__all__ = [
    "EventBus", "EventType", "NayakEvent", "bus",
    "ModuleRegistry", "ModuleStatus", "NayakModule", "registry",
]
