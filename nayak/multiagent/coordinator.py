"""
Agent coordinator module for NAYAK multi-agent coordination.
Provides a top-level orchestrator combining discovery, messaging,
delegation, and shared state into a unified interface.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional

from .discovery import AgentDiscovery, AgentInfo
from .messenger import AgentMessage, AgentMessenger, MessageType
from .delegation import Task, TaskDelegator
from .shared_state import SharedState


@dataclass
class CoordinatorConfig:
    """Configuration for the NayakCoordinator."""
    agent_id: str
    host: str = "0.0.0.0"
    port: int = 8100
    capabilities: List[str] = field(default_factory=list)
    broadcast_interval: float = 5.0


class NayakCoordinator:
    """
    Top-level orchestrator that ties together all multi-agent subsystems.
    Manages discovery, point-to-point messaging, task delegation, and shared state.
    """

    def __init__(self, config: CoordinatorConfig):
        """
        Initializes the NayakCoordinator and its internal subsystems.
        
        Args:
            config: The CoordinatorConfig instance.
        """
        self.config = config
        self.discovery = AgentDiscovery()
        self.messenger = AgentMessenger()
        self.delegator = TaskDelegator(agent_id=config.agent_id, messenger=self.messenger)
        self.shared_state = SharedState(agent_id=config.agent_id, messenger=self.messenger)
        
        self.discovery.BROADCAST_INTERVAL = config.broadcast_interval
        
        self._external_message_handler: Optional[Callable[[AgentMessage], Coroutine[Any, Any, None]]] = None

    async def start(self) -> None:
        """
        Starts discovery, messenger server, and registers this agent.
        Sets initial shared state key.
        """
        self.messenger.on_message(self._route_message)
        
        await self.messenger.start_server(
            host=self.config.host,
            port=self.config.port,
            agent_id=self.config.agent_id
        )
        
        self.discovery.register_self(
            agent_id=self.config.agent_id,
            port=self.config.port,
            capabilities=self.config.capabilities
        )
        await self.discovery.start()
        
        await self.shared_state.set("coordinator_id", self.config.agent_id)

    async def stop(self) -> None:
        """Clean shutdown of all subsystems in reverse order."""
        await self.discovery.stop()
        await self.messenger.stop()

    async def delegate(self, goal: str, target_agent_id: str) -> Optional[Task]:
        """
        Looks up target agent from discovery, delegates a task, and returns the result.
        
        Args:
            goal: The task goal to execute.
            target_agent_id: The ID of the agent to delegate to.
            
        Returns:
            The completed Task object or None if target agent is not found.
        """
        target_agent = None
        for agent in self.discovery.get_agents():
            if agent.agent_id == target_agent_id:
                target_agent = agent
                break
                
        if not target_agent:
            return None
            
        return await self.delegator.delegate_task(
            target_agent_id=target_agent_id,
            target_host=target_agent.host,
            target_port=target_agent.port,
            goal=goal
        )

    async def broadcast_state(self, key: str, value: Any) -> None:
        """
        Sets a value in SharedState and broadcasts it to known peers.
        
        Args:
            key: The state key to update.
            value: The new value.
        """
        self.shared_state._known_agents = self.discovery.get_agents()
        await self.shared_state.set(key, value)

    def get_peers(self) -> List[AgentInfo]:
        """
        Returns a list of currently known agents.
        
        Returns:
            A list of AgentInfo objects.
        """
        return self.discovery.get_agents()

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from SharedState.
        
        Args:
            key: The state key to retrieve.
            default: The default value if not found.
            
        Returns:
            The state value or default.
        """
        return self.shared_state.get(key, default)

    def on_message(self, handler: Callable[[AgentMessage], Coroutine[Any, Any, None]]) -> None:
        """
        Registers an external message handler for custom message types.
        
        Args:
            handler: An async function accepting an AgentMessage.
        """
        self._external_message_handler = handler

    async def _route_message(self, message: AgentMessage) -> None:
        """
        Internal router mapping incoming messages to their respective subsystems.
        
        Args:
            message: The incoming AgentMessage.
        """
        if message.message_type in (MessageType.TASK_REQUEST, MessageType.TASK_RESULT):
            await self.delegator.handle_incoming_task(message)
        elif message.message_type == MessageType.BROADCAST:
            await self.shared_state.handle_state_message(message)
        else:
            if self._external_message_handler:
                await self._external_message_handler(message)
