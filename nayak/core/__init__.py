"""
nayak.core — The NAYAK runtime core.

Contains the fundamental building blocks that all layers depend on:
- EventBus: async pub/sub backbone for inter-layer communication
"""

from nayak.core.bus import EventBus, EventType, NayakEvent, bus

__all__ = ["EventBus", "EventType", "NayakEvent", "bus"]
