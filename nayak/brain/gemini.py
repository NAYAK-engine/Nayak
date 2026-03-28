"""
brain/gemini.py — Reasoning engine powered by Google Gemini (free tier).

Uses gemini-1.5-flash via the google-generativeai SDK.
Receives a full world-state observation and returns a single typed Action
the agent should execute next. Handles retries and structured output parsing.

Get your free API key at: https://aistudio.google.com
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    EXTRACT = "extract"
    SAVE_FILE = "save_file"
    FINISH = "finish"


@dataclass
class Action:
    """Structured action returned by the brain after reasoning."""

    type: ActionType
    # Flexible parameter bag — each action type uses a subset of these.
    selector: str | None = None          # CSS selector for click/type
    coordinates: tuple[int, int] | None = None  # (x, y) fallback for click
    text: str | None = None              # text to type or content to save
    url: str | None = None               # URL for navigate
    direction: str | None = None         # "up" or "down" for scroll
    amount: int | None = None            # pixels for scroll
    filename: str | None = None          # destination path for save_file
    reason: str = ""                     # Brain's justification for the action
    raw: dict[str, Any] = field(default_factory=dict)  # Full parsed JSON blob

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        action_type = ActionType(data["type"])
        coords = data.get("coordinates")
        return cls(
            type=action_type,
            selector=data.get("selector"),
            coordinates=tuple(coords) if coords else None,  # type: ignore[arg-type]
            text=data.get("text"),
            url=data.get("url"),
            direction=data.get("direction"),
            amount=data.get("amount"),
            filename=data.get("filename"),
            reason=data.get("reason", ""),
            raw=data,
        )

    def __str__(self) -> str:
        parts = [f"[{self.type.value.upper()}]"]
        if self.selector:
            parts.append(f"selector={self.selector!r}")
        if self.coordinates:
            parts.append(f"coords={self.coordinates}")
        if self.text:
            preview = self.text[:60] + "…" if len(self.text) > 60 else self.text
            parts.append(f"text={preview!r}")
        if self.url:
            parts.append(f"url={self.url!r}")
        if self.direction:
            parts.append(f"direction={self.direction} amount={self.amount}")
        if self.filename:
            parts.append(f"file={self.filename!r}")
        if self.reason:
            parts.append(f"| {self.reason}")
        return " ".join(parts)


_SYSTEM_PROMPT = """\
You are NAYAK — an autonomous web-browsing agent that executes multi-step tasks in a real Chromium browser.

Your role is to reason about the current state of the world (provided as a screenshot + page content + memory), \
then decide ONE action to take next from the allowed action set.

## Allowed actions (respond with EXACTLY one):

{"type": "click", "selector": "CSS selector or aria-label", "coordinates": null, "reason": "why this click advances the goal"}

{"type": "type", "selector": "input CSS selector", "text": "text to type", "reason": "what this text input accomplishes"}

{"type": "navigate", "url": "https://example.com", "reason": "why navigating here"}

{"type": "scroll", "direction": "down", "amount": 500, "reason": "why scrolling"}

{"type": "extract", "reason": "what information you are extracting and why"}

{"type": "save_file", "filename": "report.md", "text": "Full markdown content to write", "reason": "saving final output"}

{"type": "finish", "reason": "Goal has been fully completed because..."}

## Rules
- Respond ONLY with a single valid JSON object. No prose, no markdown fences, no backticks.
- Prefer CSS selectors over coordinates when possible.
- If the page has a CAPTCHA or is blocked, use navigate to a different source.
- Think step by step inside the reason field before committing to the action.
- Use finish ONLY when the user's goal is completely satisfied.
"""


class Brain:
    """
    Wrapper around the Google Gemini API (gemini-1.5-flash — free tier).

    Converts a rich observation into a single typed Action.
    Retries up to `max_retries` times on transient API failures.
    """

    MODEL_NAME = "gemini-1.5-flash"

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key= to Brain()."
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            system_instruction=_SYSTEM_PROMPT,
        )
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _build_prompt(
        self,
        goal: str,
        step: int,
        url: str,
        page_title: str,
        page_text: str,
        screenshot_b64: str | None,
        memory_context: str,
    ) -> list[Any]:
        """
        Build a multimodal content list for the Gemini API.

        Gemini accepts a flat list of text strings and inline image blobs.
        """
        import base64

        parts: list[Any] = []

        # Attach screenshot if available (Gemini multimodal)
        if screenshot_b64:
            try:
                img_bytes = base64.b64decode(screenshot_b64)
                parts.append({"mime_type": "image/png", "data": img_bytes})
            except Exception as exc:
                logger.warning("Could not attach screenshot to Gemini prompt: %s", exc)

        truncated_text = page_text[:4000] if page_text else "(no page text)"
        memory_block = memory_context if memory_context else "(no prior memory)"

        observation = (
            f"STEP {step}\n"
            f"GOAL: {goal}\n\n"
            f"CURRENT URL: {url}\n"
            f"PAGE TITLE: {page_title}\n\n"
            f"RECENT MEMORY:\n{memory_block}\n\n"
            f"PAGE CONTENT (truncated to 4 000 chars):\n{truncated_text}\n\n"
            f"Decide your next action. Respond with ONLY a single JSON object — no prose, no fences."
        )
        parts.append(observation)
        return parts

    def decide(
        self,
        goal: str,
        step: int,
        url: str,
        page_title: str,
        page_text: str,
        screenshot_b64: str | None,
        memory_context: str,
    ) -> Action:
        """
        Call Gemini and return the next Action.
        Raises RuntimeError after exhausting retries.
        """
        prompt_parts = self._build_prompt(
            goal, step, url, page_title, page_text, screenshot_b64, memory_context
        )
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug("Brain.decide() — attempt %d/%d", attempt, self._max_retries)
                response = self._model.generate_content(prompt_parts)
                raw_text = response.text.strip()

                # Strip optional markdown code fences the model sometimes adds
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    # Drop first line (```json or ```) and last line (```)
                    inner = "\n".join(lines[1:])
                    if inner.endswith("```"):
                        inner = inner[: inner.rfind("```")]
                    raw_text = inner.strip()

                data: dict[str, Any] = json.loads(raw_text)
                action = Action.from_dict(data)
                logger.debug("Brain decided: %s", action)
                return action

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Failed to parse action JSON on attempt %d: %s\nRaw: %s",
                    attempt,
                    exc,
                    getattr(response, "text", "N/A")[:200] if "response" in dir() else "N/A",
                )
                time.sleep(self._retry_delay)

            except Exception as exc:
                # Covers google.api_core quota / rate limit errors
                err_str = str(exc).lower()
                if any(k in err_str for k in ("quota", "rate", "429", "503", "unavailable")):
                    last_error = exc
                    wait = self._retry_delay * attempt
                    logger.warning(
                        "Transient API error on attempt %d: %s — retrying in %.1fs",
                        attempt,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Non-retriable Gemini API error: {exc}") from exc

        raise RuntimeError(
            f"Brain.decide() failed after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )
