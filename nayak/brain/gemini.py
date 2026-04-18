"""
brain/gemini.py — Google Gemini 2.0 implementation using functional exports.
"""

import os
import json
import logging
from google import genai

from nayak.brain import Action, ActionType  # noqa: F401

logger = logging.getLogger(__name__)

# Initialize the client at module level
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

MODEL = "gemini-2.0-flash"

def _parse_json_safely(text: str) -> dict | list:
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                text = "\n".join(lines[1:-1])
            else:
                text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        logger.warning(f"Gemini JSON parse error: {e}. Raw: {text}")
        return {}

async def decide(
    goal: str,
    step: int,
    url: str,
    page_title: str,
    page_text: str,
    screenshot_b64: str | None,
    memory_context: str,
) -> Action:
    """Ask Gemini for the next action."""
    prompt = f"""
You are NAYAK autonomous agent.
Goal: {goal}
Step: {step}
Current URL: {url}
Page Title: {page_title}

Memory:
{memory_context}

Page Content:
{page_text[:5000]}

Return ONLY a JSON action object.
Example: {{"type": "navigate", "params": {{"url": "https://example.com"}}}}
Available actions: navigate, click, type_text, press_key, scroll, extract, save_file, search, finish
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    data = _parse_json_safely(response.text)
    return Action.from_dict(data)

async def generate(prompt: str) -> str:
    """Generate a markdown report or long-form text."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text

async def plan(goal: str, context: str = "") -> list:
    """Generate a multi-step execution plan."""
    prompt = f"Goal: {goal}\nContext: {context}\nReturn a JSON array of initial actions."
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return _parse_json_safely(response.text)
