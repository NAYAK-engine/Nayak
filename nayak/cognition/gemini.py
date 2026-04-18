"""
nayak/cognition/gemini.py — Google Gemini cognition backend for NAYAK.

Wraps the Google Gemini 2.0 API as a proper :class:`CognitionBase` subclass
so it can be registered with the module registry, lifecycle-managed by the
runtime, and swapped with any other cognition backend transparently.

Environment variables
---------------------
GEMINI_API_KEY   Your Google Gemini API key (required for cloud usage)

The Gemini client and model name are resolved once at import time.
"""

from __future__ import annotations

import json
import logging
import os

from google import genai

from nayak.brain import Action, ActionType  # noqa: F401 (ActionType re-exported)
from nayak.cognition.base import CognitionBase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level configuration  (resolved once at import time)
# ─────────────────────────────────────────────────────────────────────────────

_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

_MODEL = "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_safely(text: str) -> dict | list:
    """Parse JSON from *text*, stripping markdown fences if present.

    Args:
        text: Raw string returned by the Gemini API.

    Returns:
        A parsed ``dict`` or ``list``.  Returns an empty ``dict`` if parsing
        fails, after logging a warning with the raw response.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# GeminiCognition
# ─────────────────────────────────────────────────────────────────────────────

class GeminiCognition(CognitionBase):
    """Google Gemini-backed NAYAK cognition module.

    Implements :class:`~nayak.cognition.base.CognitionBase` using the
    ``google-genai`` SDK.  All API calls go through the module-level
    ``_client`` singleton so the connection configuration (API key, model
    name) is identical to the original ``brain/gemini.py`` implementation.

    Lifecycle integration (via :class:`~nayak.cognition.base.CognitionBase`):
        * :meth:`init` — registers the module with the runtime registry and
          emits ``COGNITION_READY`` on the bus.
        * :meth:`stop` — sets the registry status to ``STOPPED``.
    """

    @property
    def name(self) -> str:
        """Unique module identifier used in the registry and bus events."""
        return "cognition.gemini"

    # ------------------------------------------------------------------
    # CognitionBase interface
    # ------------------------------------------------------------------

    async def plan(self, goal: str, context: str = "") -> list[str]:
        """Decompose *goal* into a multi-step execution plan.

        Sends a structured prompt to Gemini requesting a JSON array of action
        objects.  If the response cannot be parsed as a list, whatever
        ``_parse_json_safely`` returns is handed back directly (may be an
        empty ``dict`` or partial result).

        Args:
            goal:    The high-level objective.
            context: Optional prior context to inform planning.

        Returns:
            A parsed ``list`` of planned steps (may be empty on parse failure).
        """
        prompt = (
            f"Goal: {goal}\nContext: {context}\n"
            "Return a JSON array of initial actions."
        )
        response = _client.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )
        return _parse_json_safely(response.text)

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
        """Ask Gemini for the next browser action to execute.

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
        response = _client.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )
        data = _parse_json_safely(response.text)
        return Action.from_dict(data)

    async def generate(self, prompt: str) -> str:
        """Generate free-form text (reports, summaries, etc.) from *prompt*.

        Args:
            prompt: The full prompt to send to Gemini.

        Returns:
            The model's plain-text response string.
        """
        response = _client.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )
        return response.text


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

gemini_cognition: GeminiCognition = GeminiCognition()
"""The process-wide Gemini cognition instance.

Import and use directly::

    from nayak.cognition.gemini import gemini_cognition
"""

__all__ = ["GeminiCognition", "gemini_cognition"]
