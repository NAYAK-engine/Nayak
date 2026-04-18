"""
nayak/cognition/ollama.py — Ollama cognition backend for NAYAK.

Wraps the Ollama local and cloud API as a proper :class:`CognitionBase`
subclass, so it can be registered with the module registry, lifecycle-managed
by the runtime, and swapped with any other cognition backend transparently.

Supports two modes via the ``OLLAMA_MODE`` environment variable:

    local  (default) — connects to http://localhost:11434, no key needed
    cloud             — connects to https://api.ollama.com/v1 using OLLAMA_API_KEY

Environment variables
---------------------
OLLAMA_MODE      local | cloud  (default: local)
OLLAMA_MODEL     model name     (default: llama3.2)
OLLAMA_BASE_URL  base URL override
OLLAMA_API_KEY   API key for cloud mode

Download Ollama : https://ollama.com
Start local     : ollama serve
Pull a model    : ollama pull llama3.2
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from nayak.brain import Action, ActionType  # noqa: F401 (ActionType re-exported)
from nayak.cognition.base import CognitionBase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level configuration  (resolved once at import time)
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODE: str = os.environ.get("OLLAMA_MODE", "local").lower()   # "local" | "cloud"
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2")

if OLLAMA_MODE == "cloud":
    _BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://api.ollama.com/v1")
    _API_KEY = os.environ.get("OLLAMA_API_KEY", "")
    _ENDPOINT = f"{_BASE_URL.rstrip('/')}/chat/completions"
    _USE_OPENAI_COMPAT = True           # cloud uses /chat/completions
else:
    _BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    _API_KEY = ""                       # local needs no key
    _ENDPOINT = f"{_BASE_URL.rstrip('/')}/api/generate"
    _USE_OPENAI_COMPAT = False          # local uses /api/generate


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_safely(text: str) -> dict | list:
    """Parse JSON from *text*, stripping markdown fences if present.

    Args:
        text: Raw string returned by the Ollama API.

    Returns:
        A parsed ``dict`` or ``list``.  Returns an empty ``dict`` if parsing
        fails, after logging a warning.
    """
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = (
                "\n".join(lines[1:-1])
                if lines[-1].strip() == "```"
                else "\n".join(lines[1:])
            )
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        logger.warning("Ollama JSON parse error: %s — raw: %.200s", exc, text)
        return {}


async def _call(prompt: str) -> str:
    """Send *prompt* to Ollama and return the response text.

    Routes to the correct endpoint depending on ``OLLAMA_MODE``:

    - ``local`` → ``POST /api/generate``  (Ollama native REST)
    - ``cloud`` → ``POST /v1/chat/completions``  (OpenAI-compatible)

    Args:
        prompt: The full prompt string to send.

    Returns:
        The model's response as a plain ``str``.

    Raises:
        ConnectionError: If Ollama cannot be reached.
        RuntimeError:    On non-2xx HTTP responses.
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
                    },
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


# ─────────────────────────────────────────────────────────────────────────────
# OllamaCognition
# ─────────────────────────────────────────────────────────────────────────────

class OllamaCognition(CognitionBase):
    """Ollama-backed NAYAK cognition module.

    Implements :class:`~nayak.cognition.base.CognitionBase` using the Ollama
    local or cloud REST API.  The class delegates all HTTP work to the
    module-level :func:`_call` helper so that the connection and retry
    behaviour is identical to the original ``brain/ollama.py`` implementation.

    Lifecycle integration (via :class:`~nayak.cognition.base.CognitionBase`):
        * :meth:`init` — registers the module with the runtime registry and
          emits ``COGNITION_READY`` on the bus.
        * :meth:`stop` — sets the registry status to ``STOPPED``.
    """

    @property
    def name(self) -> str:
        """Unique module identifier used in the registry and bus events."""
        return "cognition.ollama"

    # ------------------------------------------------------------------
    # CognitionBase interface
    # ------------------------------------------------------------------

    async def plan(self, goal: str, context: str = "") -> list[str]:
        """Decompose *goal* into a multi-step execution plan.

        Sends a structured prompt to Ollama requesting a JSON array of action
        objects.  If the response cannot be parsed as a list, an empty list is
        returned rather than raising.

        Args:
            goal:    The high-level objective.
            context: Optional prior context to inform planning.

        Returns:
            An ordered ``list[str]`` of planned steps (may be empty on parse
            failure).
        """
        prompt = (
            f"Goal: {goal}\nContext: {context}\n"
            "Return ONLY a valid JSON array of action objects. No explanation."
        )
        raw = await _call(prompt)
        result = _parse_json_safely(raw)
        return result if isinstance(result, list) else []

    async def decide(
        self,
        goal: str,
        step: int,
        url: str,
        page_title: str,
        page_text: str,
        screenshot_b64: str | None,
        memory_context: str,
    ) -> Action:
        """Ask Ollama for the next browser action to execute.

        Constructs a structured prompt from the full browser snapshot and
        current memory, then parses the model's JSON response into an
        :class:`~nayak.brain.Action`.

        Args:
            goal:           The top-level goal string.
            step:           Current step index (1-based).
            url:            URL of the currently visible page.
            page_title:     ``<title>`` of the current page.
            page_text:      Visible text content of the current page.
            screenshot_b64: Base-64-encoded PNG screenshot (may be ``None``).
            memory_context: Concatenated string of recent memory entries.

        Returns:
            A single :class:`~nayak.brain.Action` instance.
        """
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

    async def generate(self, prompt: str) -> str:
        """Generate free-form text (reports, summaries, etc.) from *prompt*.

        Args:
            prompt: The full prompt to send to Ollama.

        Returns:
            The model's plain-text response.
        """
        return await _call(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

ollama_cognition: OllamaCognition = OllamaCognition()
"""The process-wide Ollama cognition instance.

Import and use directly::

    from nayak.cognition.ollama import ollama_cognition
"""

__all__ = ["OllamaCognition", "ollama_cognition"]
