"""
examples/web_researcher.py — Fully autonomous web research agent.

Goal: Search Google for top 3 AI robotics companies. Visit each company website.
      Read their about page. Save a detailed report to report.md with company name, 
      description, and main products.

Setup:
    1. Go to https://console.groq.com
    2. Sign up free
    3. Click API Keys -> Create API Key
    4. Copy .env.example to .env and paste your key starting with gsk_
    5. pip install -e .
    6. playwright install chromium
    7. python examples/web_researcher.py
"""

from __future__ import annotations

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
    import asyncio
    from nayak.brain.groq import Brain
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nERROR: GROQ_API_KEY is not set.\n")
        sys.exit(1)

    config = AgentConfig(
        goal="Search Google for top 3 AI robotics companies. "
             "Visit each company website. Read their about page. "
             "Save a detailed report to report.md with company name, "
             "description, and main products.",
        max_steps=100,
        agent_id="web-researcher",
        groq_api_key=api_key
    )
    
    agent = Agent(config)
    asyncio.run(agent.run())
