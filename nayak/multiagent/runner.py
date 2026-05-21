"""
Runner module for NAYAK multi-agent coordination.
Provides helper functions to spin up a Coordinator robustly.
"""
import asyncio

from .coordinator import CoordinatorConfig, NayakCoordinator


async def run_coordinator(config: CoordinatorConfig) -> None:
    """
    Starts the coordinator and runs forever until manually interrupted.
    
    Args:
        config: The CoordinatorConfig instance.
    """
    coordinator = NayakCoordinator(config)
    await coordinator.start()
    
    print(f"NAYAK Coordinator {config.agent_id} running on {config.host}:{config.port}")
    
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\\nShutting down NAYAK Coordinator {config.agent_id}...")
        await coordinator.stop()
