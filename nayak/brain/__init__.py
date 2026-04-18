"""
brain/__init__.py — Shared types used by all NAYAK brain providers.

Providers:
  ollama  — Ollama local or cloud (100 % free, no key for local)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    NAVIGATE   = "navigate"
    CLICK      = "click"
    TYPE_TEXT  = "type_text"
    PRESS_KEY  = "press_key"
    SCROLL     = "scroll"
    EXTRACT    = "extract"
    SAVE_FILE  = "save_file"
    SEARCH     = "search"
    FINISH     = "finish"


@dataclass
class Action:
    type: ActionType
    # Optional fields — populated depending on action type
    url: str | None = None
    selector: str | None = None
    text: str | None = None
    key: str | None = None
    direction: str = "down"
    amount: int = 500
    filename: str | None = None
    coordinates: tuple[int, int] | None = None
    reason: str = ""

    # ----------------------------------------------------------------
    # Factory
    # ----------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        """
        Build an Action from a JSON dict returned by a brain provider.

        Expected shape:
            {"type": "navigate", "params": {"url": "https://example.com"}}
        """
        if not isinstance(data, dict):
            logger.warning("Action.from_dict received non-dict: %r — using finish", data)
            return cls(type=ActionType.FINISH, reason="bad action data from brain")

        raw_type = data.get("type", "finish")
        try:
            action_type = ActionType(raw_type)
        except ValueError:
            logger.warning("Unknown action type '%s' — defaulting to finish", raw_type)
            action_type = ActionType.FINISH

        params: dict[str, Any] = data.get("params", {})

        coords = None
        if "x" in params and "y" in params:
            coords = (int(params["x"]), int(params["y"]))

        return cls(
            type=action_type,
            url=params.get("url"),
            selector=params.get("selector"),
            text=params.get("text"),
            key=params.get("key"),
            direction=params.get("direction", "down"),
            amount=int(params.get("amount", 500)),
            filename=params.get("filename"),
            coordinates=coords,
            reason=params.get("reason", ""),
        )

    def __str__(self) -> str:
        parts = [f"type={self.type.value}"]
        if self.url:        parts.append(f"url={self.url!r}")
        if self.selector:   parts.append(f"selector={self.selector!r}")
        if self.text:       parts.append(f"text={self.text[:60]!r}")
        if self.key:        parts.append(f"key={self.key!r}")
        if self.filename:   parts.append(f"filename={self.filename!r}")
        if self.reason:     parts.append(f"reason={self.reason!r}")
        return f"Action({', '.join(parts)})"


__all__ = ["Action", "ActionType"]
