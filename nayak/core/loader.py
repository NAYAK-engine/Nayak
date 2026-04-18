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

    - Layer 2 (Perception): Deferred. Handled per-agent session.
    - Layer 3 (Cognition): Loaded immediately based on NAYAK_PROVIDER.
    - Layer 4 (Action): Deferred. Handled per-agent session.
    - Layer 5 (Memory): Deferred. Handled per-agent session.

    Logs the final registry summary upon completion.
    """
    logger.info("Initializing default NAYAK modules...")

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

    logger.info("Default modules loaded.\n%s", registry.summary())
