"""
Agent shared state module for NAYAK multi-agent coordination.
Provides a distributed key-value store synchronized across the network.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .discovery import AgentInfo
from .messenger import AgentMessage, AgentMessenger, MessageType


@dataclass
class StateEntry:
    """Dataclass representing an entry in the shared state."""
    key: str
    value: Any
    updated_by: str
    updated_at: float = field(default_factory=time.time)
    version: int = 1


class SharedState:
    """
    Manages a synchronized key-value store across multiple agents.
    Uses AgentMessenger to broadcast state updates.
    """

    def __init__(self, agent_id: str, messenger: AgentMessenger):
        """
        Initializes the SharedState store.
        
        Args:
            agent_id: The unique identifier of this agent.
            messenger: An initialized AgentMessenger instance.
        """
        self.agent_id = agent_id
        self.messenger = messenger
        self._store: Dict[str, StateEntry] = {}
        self._known_agents: List[AgentInfo] = []

    async def set(self, key: str, value: Any) -> None:
        """
        Stores key/value locally, then broadcasts the update to all known agents.
        
        Args:
            key: The state key to update.
            value: The new value.
        """
        current_entry = self._store.get(key)
        new_version = (current_entry.version + 1) if current_entry else 1
        
        entry = StateEntry(
            key=key,
            value=value,
            updated_by=self.agent_id,
            version=new_version
        )
        self._store[key] = entry
        
        tasks = []
        for agent in self._known_agents:
            if agent.agent_id == self.agent_id:
                continue
                
            message = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=agent.agent_id,
                message_type=MessageType.BROADCAST,
                payload={
                    "action": "state_update",
                    "key": key,
                    "value": value,
                    "version": new_version,
                    "updated_by": self.agent_id
                }
            )
            task = asyncio.create_task(
                self._safe_send(agent.host, agent.port, message)
            )
            tasks.append(task)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_send(self, host: str, port: int, message: AgentMessage) -> None:
        """Safely sends a message, ignoring connection errors during broadcast."""
        try:
            await self.messenger.send_message(host, port, message)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """
        Returns value for key from local store, or default if not found.
        
        Args:
            key: The state key to retrieve.
            default: The default value to return if key does not exist.
            
        Returns:
            The stored value or the default.
        """
        entry = self._store.get(key)
        if entry:
            return entry.value
        return default

    def get_all(self) -> Dict[str, Any]:
        """
        Returns full dict of all current state entries (just key-value pairs).
        
        Returns:
            A dictionary mapping keys to their current values.
        """
        return {k: v.value for k, v in self._store.items()}

    async def handle_state_message(self, message: AgentMessage) -> None:
        """
        Receives BROADCAST messages, applies state_update or state_sync 
        if incoming version is higher than local version.
        
        Args:
            message: The received AgentMessage object.
        """
        if message.message_type != MessageType.BROADCAST:
            return
            
        action = message.payload.get("action")
        
        if action == "state_update":
            self._apply_update(message.payload)
        elif action == "state_sync":
            entries = message.payload.get("entries", [])
            for entry_data in entries:
                self._apply_update(entry_data)

    def _apply_update(self, payload: dict) -> None:
        """Internal helper to apply a state update based on versioning."""
        key = payload.get("key")
        if not key:
            return
            
        incoming_version = payload.get("version", 1)
        current_entry = self._store.get(key)
        
        # Last-write-wins by version
        if not current_entry or incoming_version > current_entry.version:
            self._store[key] = StateEntry(
                key=key,
                value=payload.get("value"),
                updated_by=payload.get("updated_by", "unknown"),
                version=incoming_version
            )

    async def sync_with_agents(self, agents: List[AgentInfo]) -> None:
        """
        Takes list of AgentInfo, updates internal known agents, and sends 
        current full state to each as a BROADCAST with action state_sync.
        
        Args:
            agents: List of discovered AgentInfo objects.
        """
        self._known_agents = agents
        
        if not self._store:
            return
            
        entries = [
            {
                "key": entry.key,
                "value": entry.value,
                "version": entry.version,
                "updated_by": entry.updated_by
            }
            for entry in self._store.values()
        ]
        
        tasks = []
        for agent in agents:
            if agent.agent_id == self.agent_id:
                continue
                
            message = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=agent.agent_id,
                message_type=MessageType.BROADCAST,
                payload={
                    "action": "state_sync",
                    "entries": entries
                }
            )
            task = asyncio.create_task(
                self._safe_send(agent.host, agent.port, message)
            )
            tasks.append(task)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
