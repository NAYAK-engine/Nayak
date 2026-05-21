"""
Agent discovery module for NAYAK multi-agent coordination.
Handles UDP broadcasting and listening for agent presence on the network.
"""
import asyncio
import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class AgentInfo:
    """Dataclass representing information about a discovered agent."""
    agent_id: str
    host: str
    port: int
    capabilities: List[str]
    status: str
    last_seen: float


class DiscoveryProtocol(asyncio.DatagramProtocol):
    """UDP protocol for handling incoming agent announcements."""
    
    def __init__(self, discovery_manager: 'AgentDiscovery'):
        """Initializes the discovery protocol with a reference to the manager."""
        self.discovery_manager = discovery_manager
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """Called when the UDP socket is created."""
        self.transport = transport # type: ignore

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Called when a UDP datagram is received."""
        try:
            payload = json.loads(data.decode('utf-8'))
            self.discovery_manager._handle_announcement(payload, addr[0])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass


class AgentDiscovery:
    """
    Manages discovery of other agents on the network using UDP broadcast.
    Broadcasts this agent's presence and listens for others.
    """
    BROADCAST_PORT = 47700
    BROADCAST_INTERVAL = 5.0
    AGENT_TIMEOUT = 30.0

    def __init__(self):
        """Initializes the AgentDiscovery service."""
        self._self_info: Optional[AgentInfo] = None
        self._agents: Dict[str, AgentInfo] = {}
        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[asyncio.DatagramProtocol] = None

    def register_self(self, agent_id: str, port: int, capabilities: List[str]) -> None:
        """
        Registers this agent's own information before broadcasting.
        
        Args:
            agent_id: Unique identifier for this agent.
            port: The TCP port this agent is listening on for direct communication.
            capabilities: A list of capabilities this agent supports.
        """
        self._self_info = AgentInfo(
            agent_id=agent_id,
            host='0.0.0.0',
            port=port,
            capabilities=capabilities,
            status='active',
            last_seen=time.time()
        )

    def _handle_announcement(self, payload: dict, host: str) -> None:
        """
        Processes an incoming announcement payload.
        
        Args:
            payload: The decoded JSON payload from the broadcast.
            host: The IP address of the sender.
        """
        agent_id = payload.get('agent_id')
        if not agent_id or (self._self_info and agent_id == self._self_info.agent_id):
            return

        port = payload.get('port', 0)
        capabilities = payload.get('capabilities', [])
        status = payload.get('status', 'unknown')

        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            host=host,
            port=port,
            capabilities=capabilities,
            status=status,
            last_seen=time.time()
        )

    async def _broadcast_loop(self) -> None:
        """Continuously broadcasts this agent's presence to the network."""
        loop = asyncio.get_running_loop()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        try:
            while self._running:
                if self._self_info:
                    self._self_info.last_seen = time.time()
                    payload = json.dumps(asdict(self._self_info)).encode('utf-8')
                    try:
                        await loop.sock_sendto(sock, payload, ('255.255.255.255', self.BROADCAST_PORT))
                    except Exception:
                        pass
                
                await asyncio.sleep(self.BROADCAST_INTERVAL)
        finally:
            sock.close()

    async def start(self) -> None:
        """Starts the broadcaster and listener."""
        if self._running:
            return
        
        self._running = True
        loop = asyncio.get_running_loop()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except AttributeError:
                pass
                
        sock.bind(('0.0.0.0', self.BROADCAST_PORT))
        
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: DiscoveryProtocol(self),
            sock=sock
        )

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Clean shutdown of the discovery service."""
        self._running = False
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
            
        if self._transport:
            self._transport.close()
            self._transport = None
            self._protocol = None

    def get_agents(self) -> List[AgentInfo]:
        """
        Returns a list of all currently known AgentInfo objects.
        Excludes this agent and any agents not seen in the last 30 seconds.
        """
        current_time = time.time()
        active_agents = []
        expired = []
        
        for agent_id, agent_info in self._agents.items():
            if current_time - agent_info.last_seen <= self.AGENT_TIMEOUT:
                active_agents.append(agent_info)
            else:
                expired.append(agent_id)
                
        for agent_id in expired:
            del self._agents[agent_id]
            
        return active_agents
