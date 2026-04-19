"""
nayak/core/loader.py — NAYAK Default Module Auto-Loader.

Initializes the base layer of the NAYAK OS by instantiating the default
backend for each OS layer. Some modules (like cognition) are booted
immediately at the OS level, while others (like memory, perception, and action)
are instantiated dynamically per-agent session.
"""

import logging
import os

from nayak.core.registry import registry

logger = logging.getLogger(__name__)


async def load_default_modules() -> None:
    """Load and initialize default NAYAK modules by OS layer order.

    - Layer 1 (HAL):       Loaded immediately — Raspberry Pi and Camera backends.
    - Layer 2 (Perception): Deferred. Handled per-agent session.
    - Layer 3 (Cognition): Loaded immediately based on NAYAK_PROVIDER.
    - Layer 4 (Action):    Deferred. Handled per-agent session.
    - Layer 5 (Memory):    Deferred. Handled per-agent session.

    Logs the final registry summary upon completion.
    """
    logger.info("Initializing default NAYAK modules...")

    # Layer 1 — HAL
    from nayak.hal.raspberry_pi import raspberry_pi
    from nayak.hal.camera import camera
    logger.info("Layer 1: Booting HAL modules...")
    await raspberry_pi.init()
    await camera.init()
    logger.info("Layer 1: HAL initialized")

    # Layer 2 — Perception
    logger.info("Layer 2: Perception engine ready for browser attachment")

    # Layer 3 — Cognition
    from nayak.cognition.gemini import gemini_cognition
    from nayak.cognition.ollama import ollama_cognition

    provider = os.environ.get("NAYAK_PROVIDER", "ollama").lower()
    if provider == "gemini":
        logger.info("Layer 3: Booting Gemini cognition backend...")
        await gemini_cognition.init()
    else:
        logger.info("Layer 3: Booting Ollama cognition backend...")
        await ollama_cognition.init()

    # Layer 4 — Action
    logger.info("Layer 4: Action engine ready for browser attachment")

    # Layer 5 — Memory
    logger.info("Layer 5: Memory engine initializes per agent session")

    # Layer 6 — Communication
    from nayak.communication.text import text_comm
    logger.info("Layer 6: Booting text communication...")
    await text_comm.init()
    logger.info("Layer 6: Communication initialized")

    # Layer 7 — Safety
    from nayak.safety.engine import safety
    logger.info("Layer 7: Booting safety engine...")
    await safety.init()
    logger.info("Layer 7: Safety engine online — all actions monitored")

    # Layer 8 — Update Engine
    from nayak.update.engine import updater
    logger.info("Layer 8: Booting update engine...")
    await updater.init()
    logger.info("Layer 8: Update engine online")

    logger.info("Default modules loaded.\n%s", registry.summary())
