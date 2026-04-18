"""
brain/ollama.py — Ollama local (and cloud) AI implementation.

Supports two modes via OLLAMA_MODE env var:
  local  (default) — connects to http://localhost:11434, no key needed
  cloud             — connects to https://api.ollama.com/v1 using OLLAMA_API_KEY

Download Ollama: https://ollama.com
Start local server: ollama serve
Pull a model:       ollama pull llama3.2
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from nayak.brain import Action, ActionType  # noqa: F401 (ActionType re-exported)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
OLLAMA_MODE = os.environ.get("OLLAMA_MODE", "local").lower()   # "local" | "cloud"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

if OLLAMA_MODE == "cloud":
    _BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://api.ollama.com/v1")
    _API_KEY = os.environ.get("OLLAMA_API_KEY", "")
    _ENDPOINT = f"{_BASE_URL.rstrip('/')}/chat/completions"
    _USE_OPENAI_COMPAT = True          # cloud uses /chat/completions
else:
    _BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    _API_KEY = ""                      # local needs no key
    _ENDPOINT = f"{_BASE_URL.rstrip('/')}/api/generate"
    _USE_OPENAI_COMPAT = False         # local uses /api/generate


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _parse_json_safely(text: str) -> dict | list:
    """Parse JSON, stripping markdown fences if present."""
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        logger.warning("Ollama JSON parse error: %s — raw: %.200s", exc, text)
        return {}


async def _call(prompt: str) -> str:
    """
    Send a prompt to Ollama and return the response text.

    Routes to the correct endpoint depending on OLLAMA_MODE:
    - local → POST /api/generate  (Ollama native REST)
    - cloud → POST /v1/chat/completions  (OpenAI-compatible)
    """
    headers = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"

    async with httpx.AsyncClient(timeout=120) as http:
        try:
            if _USE_OPENAI_COMPAT:
                # OpenAI-compatible endpoint (Ollama cloud)
                payload = {
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = await http.post(_ENDPOINT, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                # Ollama native endpoint (local)
                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096
                    }
                }
                response = await http.post(_ENDPOINT, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()

        except httpx.ConnectError:
            if OLLAMA_MODE == "local":
                raise ConnectionError(
                    "Cannot connect to local Ollama.\n"
                    "Make sure Ollama is running:  ollama serve\n"
                    "Download free at:             https://ollama.com"
                )
            else:
                raise ConnectionError(
                    "Cannot connect to Ollama cloud API.\n"
                    "Check OLLAMA_BASE_URL and OLLAMA_API_KEY in your .env file."
                )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama API error {exc.response.status_code}: {exc.response.text}"
            ) from exc


# ──────────────────────────────────────────────────────────────────
# Public functional API (matches nvidia.py / gemini.py signatures)
# ──────────────────────────────────────────────────────────────────

async def decide(
    goal: str,
    step: int,
    url: str,
    page_title: str,
    page_text: str,
    screenshot_b64: str | None,
    memory_context: str,
) -> Action:
    """Ask Ollama for the next browser action."""
    prompt = f"""You are NAYAK, an autonomous web agent.
Goal: {goal}
Step: {step}
Current URL: {url}
Page Title: {page_title}

Memory (recent steps):
{memory_context}

Page Content (first 5000 chars):
{page_text[:5000]}

Return ONLY a JSON action object — no explanation, no markdown fences.
Example: {{"type": "navigate", "params": {{"url": "https://example.com"}}}}
Available actions: navigate, click, type_text, press_key, scroll, extract, save_file, search, finish
"""
    raw = await _call(prompt)
    data = _parse_json_safely(raw)
    return Action.from_dict(data)


async def generate(prompt: str) -> str:
    """Generate a long-form markdown report or free-form text."""
    return await _call(prompt)


async def plan(goal: str, context: str = "") -> list:
    """Generate a multi-step execution plan as a JSON array."""
    prompt = (
        f"Goal: {goal}\nContext: {context}\n"
        "Return ONLY a valid JSON array of action objects. No explanation."
    )
    raw = await _call(prompt)
    result = _parse_json_safely(raw)
    return result if isinstance(result, list) else []
