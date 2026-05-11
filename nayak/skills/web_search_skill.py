"""
nayak/skills/web_search_skill.py — NAYAK Web Search Skill.

First-party skill that searches the web via DuckDuckGo's Instant Answer API
and returns structured results.  Built on the NAYAK SDK (Layer 9).

Usage::

    from nayak.skills.web_search_skill import WebSearchSkill

    skill = WebSearchSkill()
    await skill.on_load()
    result = await skill.execute({"query": "Python programming language"})
    print(result)
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import httpx

from nayak.sdk.base import SkillBase, SkillManifest, SkillType

logger = logging.getLogger(__name__)


class WebSearchSkill(SkillBase):
    """Search the web via DuckDuckGo and return structured results.

    This skill queries the DuckDuckGo Instant Answer API and extracts:
    - Direct answers
    - Abstract summaries
    - Related topic titles (up to 5)

    Requires ``NETWORK_ACCESS`` permission.
    """

    def __init__(self) -> None:
        self._manifest = SkillManifest(
            name="web-search",
            version="1.0.0",
            skill_type=SkillType.ACTION,
            description="Search the web and return results",
            author="nayak-core",
            entry_point="nayak.skills.web_search_skill",
            permissions=["NETWORK_ACCESS"],
        )

    @property
    def manifest(self) -> SkillManifest:
        """The defining metadata for this skill."""
        return self._manifest

    async def on_load(self) -> None:
        """Called when the SDK boots this skill."""
        logger.info("Web Search Skill loaded")

    async def on_unload(self) -> None:
        """Called when the SDK shuts down this skill."""
        logger.info("Web Search Skill unloaded")

    async def execute(self, payload: dict) -> dict:
        """Search DuckDuckGo for the query in *payload*.

        Args:
            payload: Must contain a ``"query"`` key with the search string.

        Returns:
            A dict with ``query``, ``answer``, ``abstract``, ``topics``,
            and ``source`` keys on success, or ``{"error": ...}`` on failure.
        """
        query = payload.get("query")
        if not query:
            return {"error": "no query provided"}

        try:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://api.duckduckgo.com/"
                f"?q={encoded}&format=json&no_html=1"
            )

            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            answer = data.get("Answer", "") or ""
            abstract = data.get("AbstractText", "") or ""

            # Extract up to 5 related topic titles
            topics: list[str] = []
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and "Text" in topic:
                    topics.append(topic["Text"])
                    if len(topics) >= 5:
                        break

            logger.info(
                "WebSearchSkill: query=%r → answer=%d chars, abstract=%d chars, topics=%d",
                query, len(answer), len(abstract), len(topics),
            )

            return {
                "query": query,
                "answer": answer,
                "abstract": abstract,
                "topics": topics,
                "source": "duckduckgo",
            }

        except Exception as exc:
            logger.error("WebSearchSkill: execute() failed: %s", exc)
            return {"error": str(exc)}


__all__ = ["WebSearchSkill"]
