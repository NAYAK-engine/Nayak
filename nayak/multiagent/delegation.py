"""
Agent task delegation module for NAYAK multi-agent coordination.
Handles assigning tasks to other agents and awaiting results.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .messenger import AgentMessage, AgentMessenger, MessageType


class TaskStatus:
    """Constants for defining the status of a Task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class Task:
    """Dataclass representing a task assigned to an agent."""
    goal: str
    assigned_to: str
    assigned_by: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = TaskStatus.PENDING
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    timeout: float = 60.0


class TaskDelegator:
    """
    Manages task delegation between agents.
    Sends tasks via AgentMessenger and tracks their completion.
    """

    def __init__(self, agent_id: str, messenger: AgentMessenger):
        """
        Initializes the TaskDelegator.
        
        Args:
            agent_id: The unique identifier of this agent.
            messenger: An initialized AgentMessenger instance.
        """
        self.agent_id = agent_id
        self.messenger = messenger
        self._pending_tasks: Dict[str, Task] = {}
        self._task_futures: Dict[str, asyncio.Future] = {}

    async def delegate_task(self, target_agent_id: str, target_host: str, target_port: int, goal: str, timeout: float = 60.0) -> Task:
        """
        Creates a task and assigns it to a target agent, waiting for a result.
        
        Args:
            target_agent_id: The ID of the agent to execute the task.
            target_host: The IP address of the target agent.
            target_port: The TCP port of the target agent's messenger.
            goal: The task description or objective.
            timeout: Maximum time in seconds to wait for a result.
            
        Returns:
            The completed Task object.
            
        Raises:
            TimeoutError: If the target agent does not respond within the timeout.
        """
        task = Task(
            goal=goal,
            assigned_to=target_agent_id,
            assigned_by=self.agent_id,
            timeout=timeout
        )
        self._pending_tasks[task.task_id] = task
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._task_futures[task.task_id] = future
        
        # Ideally, we would include our own listener host/port for the reply,
        # but we assume the receiver will have it in the payload or via discovery.
        # Here we embed a dummy reply port just for the echo implementation, 
        # normally this comes from a configuration registry.
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=MessageType.TASK_REQUEST,
            payload={
                "task_id": task.task_id,
                "goal": task.goal,
                "timeout": task.timeout,
                "reply_host": "127.0.0.1",  # Placeholder for return address
                "reply_port": 8000         # Placeholder for return address
            }
        )
        
        task.status = TaskStatus.RUNNING
        await self.messenger.send_message(target_host, target_port, message)
        
        try:
            await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            raise TimeoutError(f"Task {task.task_id} timed out after {timeout} seconds.")
        finally:
            self._task_futures.pop(task.task_id, None)
            
        return task

    async def handle_incoming_task(self, message: AgentMessage) -> None:
        """
        Handles incoming TASK_REQUEST and TASK_RESULT messages.
        
        Args:
            message: The received AgentMessage object.
        """
        if message.message_type == MessageType.TASK_REQUEST:
            task_id = message.payload.get("task_id")
            goal = message.payload.get("goal")
            
            if not task_id or not goal:
                return
                
            # Execute goal as a simple string echo
            await asyncio.sleep(0.1)
            result_str = f"ECHO: {goal}"
            
            # Send result back
            reply_host = message.payload.get("reply_host", "127.0.0.1")
            reply_port = message.payload.get("reply_port", 0)
            
            if reply_port:
                response = AgentMessage(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.TASK_RESULT,
                    payload={
                        "task_id": task_id,
                        "status": TaskStatus.COMPLETED,
                        "result": result_str
                    }
                )
                try:
                    await self.messenger.send_message(reply_host, reply_port, response)
                except Exception:
                    pass

        elif message.message_type == MessageType.TASK_RESULT:
            task_id = message.payload.get("task_id")
            if not task_id or task_id not in self._pending_tasks:
                return
                
            task = self._pending_tasks[task_id]
            task.status = message.payload.get("status", TaskStatus.FAILED)
            task.result = message.payload.get("result")
            
            future = self._task_futures.get(task_id)
            if future and not future.done():
                future.set_result(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieves a task by its ID.
        
        Args:
            task_id: The unique identifier of the task.
            
        Returns:
            The Task object if found, otherwise None.
        """
        return self._pending_tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        """
        Returns a list of all tracked tasks.
        
        Returns:
            A list containing all Task objects.
        """
        return list(self._pending_tasks.values())
