"""
Agent messenger module for NAYAK multi-agent coordination.
Handles agent-to-agent messaging over TCP using asyncio streams.
"""
import asyncio
import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any, Optional


class MessageType:
    """Constants for defining the type of an AgentMessage."""
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESULT = "TASK_RESULT"
    PING = "PING"
    PONG = "PONG"
    BROADCAST = "BROADCAST"


@dataclass
class AgentMessage:
    """Dataclass representing a structured message exchanged between agents."""
    sender_id: str
    receiver_id: str
    message_type: str
    payload: dict
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


class AgentMessenger:
    """
    Manages direct TCP communication between agents.
    Provides methods to send and receive messages with 4-byte length framing.
    """

    def __init__(self):
        """Initializes the AgentMessenger."""
        self._server: Optional[asyncio.Server] = None
        self._agent_id: Optional[str] = None
        self._handler: Optional[Callable[[AgentMessage], Coroutine[Any, Any, None]]] = None

    def on_message(self, handler: Callable[[AgentMessage], Coroutine[Any, Any, None]]) -> None:
        """
        Registers an async callback function to handle incoming messages.
        
        Args:
            handler: An async function that takes an AgentMessage as an argument.
        """
        self._handler = handler

    async def start_server(self, host: str, port: int, agent_id: str) -> None:
        """
        Starts the TCP server to listen for incoming messages.
        
        Args:
            host: The interface to bind to.
            port: The TCP port to listen on.
            agent_id: The unique identifier of this agent.
        """
        self._agent_id = agent_id
        self._server = await asyncio.start_server(
            self._handle_connection, host, port
        )

    async def stop(self) -> None:
        """Gracefully shuts down the TCP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def send_message(self, host: str, port: int, message: AgentMessage) -> None:
        """
        Sends an AgentMessage to a remote agent over TCP.
        
        Args:
            host: The remote host IP address.
            port: The remote port.
            message: The AgentMessage instance to send.
        """
        reader, writer = await asyncio.open_connection(host, port)
        try:
            # Serialize message to JSON bytes
            message_dict = {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "message_type": message.message_type,
                "payload": message.payload,
                "timestamp": message.timestamp
            }
            data = json.dumps(message_dict).encode('utf-8')
            
            # Frame with 4-byte length prefix (big-endian unsigned int)
            header = struct.pack('>I', len(data))
            
            writer.write(header + data)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Internal connection handler for reading incoming TCP streams.
        
        Args:
            reader: The asyncio StreamReader for reading data.
            writer: The asyncio StreamWriter for responding/closing.
        """
        try:
            # Read the 4-byte length prefix
            header = await reader.readexactly(4)
            message_length = struct.unpack('>I', header)[0]
            
            # Read the full JSON payload
            data = await reader.readexactly(message_length)
            
            # Decode and deserialize
            message_dict = json.loads(data.decode('utf-8'))
            message = AgentMessage(
                message_id=message_dict['message_id'],
                sender_id=message_dict['sender_id'],
                receiver_id=message_dict['receiver_id'],
                message_type=message_dict['message_type'],
                payload=message_dict['payload'],
                timestamp=message_dict['timestamp']
            )
            
            # Dispatch to registered handler
            if self._handler:
                await self._handler(message)
                
        except (asyncio.IncompleteReadError, struct.error, json.JSONDecodeError, KeyError):
            # Ignore malformed packets and disconnects
            pass
        except Exception:
            # Broad catch to ensure server doesn't crash on handler errors
            pass
        finally:
            writer.close()
            await writer.wait_closed()
