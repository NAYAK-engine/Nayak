"""
examples/web_researcher.py — Fully autonomous web research agent.

Goal: Search Google for top 3 AI robotics companies. Visit each company website.
      Read their about page. Save a detailed report to report.md.

Setup (NVIDIA NIM — recommended):
    1. Get a free key at https://build.nvidia.com
    2. cp .env.example .env
    3. Set NAYAK_PROVIDER=nvidia and NVIDIA_API_KEY=nvapi-... in .env
    4. pip install -e .
    5. playwright install chromium
    6. python examples/web_researcher.py

Setup (Ollama — local, no key needed):
    1. Download Ollama from https://ollama.com
    2. ollama pull llama3.2
    3. ollama serve
    4. Set NAYAK_PROVIDER=ollama in .env
    5. python examples/web_researcher.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path when running as a script
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    from nayak.agent import Agent, AgentConfig

    provider = os.environ.get("NAYAK_PROVIDER", "nvidia").lower()

    # Quick key check for NVIDIA mode
    if provider == "nvidia" and not os.environ.get("NVIDIA_API_KEY"):
        print(
            "\nERROR: NVIDIA_API_KEY is not set.\n"
            "Get your free key at: https://build.nvidia.com\n"
            "Or switch to Ollama: set NAYAK_PROVIDER=ollama in .env\n"
        )
        sys.exit(1)

    config = AgentConfig(
        goal=(
            "Search Google for top 3 AI robotics companies. "
            "Visit each company website. Read their about page. "
            "Save a detailed report to report.md with company name, "
            "description, and main products."
        ),
        max_steps=100,
        agent_id="web-researcher",
        headless=False,
    )

    asyncio.run(Agent(config).run())
