"""
nayak/cognition/fastpath.py — NAYAK Cognition: Fast Path Engine.

Provides an optimized fast path execution engine that attempts to bypass the full
agent loop for simple or instantaneous queries. Uses keyword heuristics to categorize
incoming goals and directly consult the cognition backend without browser orchestration.
"""

from __future__ import annotations

import logging
import urllib.parse
from enum import Enum, auto
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QueryComplexity
# ─────────────────────────────────────────────────────────────────────────────

class QueryComplexity(Enum):
    """Categorization of query complexity for fast path routing.

    Members:
        INSTANT: A query that can be answered immediately by relying on LLM
                 knowledge (e.g., factual lookups, single values).
        SIMPLE:  A query that may require minimal thought but still bypasses
                 the complex multi-step autonomous loops.
        COMPLEX: A query requiring full agentic planning, multi-step browser
                 execution, and deep research.
    """
    INSTANT = auto()
    SIMPLE = auto()
    COMPLEX = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Classification Engine
# ─────────────────────────────────────────────────────────────────────────────

async def classify_query(goal: str) -> QueryComplexity:
    """Classify an incoming user goal to determine its complexity track.

    Uses zero-dependency keyword pattern matching to route queries.

    Args:
        goal: The user-provided goal string.

    Returns:
        The evaluated QueryComplexity track.
    """
    goal_lower = goal.lower()

    instant_patterns = [
        "price of", "current price", "what is the price",
        "weather in", "what is today", "who is", "what is",
        "define ", "meaning of", "capital of", "population of"
    ]

    complex_patterns = [
        "research", "find all", "compare", "write a report",
        "analyze", "list top", "summarize", "investigate"
    ]

    if any(p in goal_lower for p in instant_patterns):
        return QueryComplexity.INSTANT

    if any(p in goal_lower for p in complex_patterns):
        return QueryComplexity.COMPLEX

    return QueryComplexity.SIMPLE


# ─────────────────────────────────────────────────────────────────────────────
# Fast Execution
# ─────────────────────────────────────────────────────────────────────────────

async def web_search(query: str) -> str | None:
    """Search the web for instant answers using multiple sources."""
    import httpx
    import urllib.parse

    encoded = urllib.parse.quote(query)

    # Source 1 — DuckDuckGo instant answer
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
            )
            data = r.json()
            for field in ("Answer", "AbstractText", "Definition"):
                if data.get(field):
                    return data[field]
    except Exception:
        pass

    # Source 2 — wttr.in for weather queries
    weather_keywords = ("weather", "temperature", "forecast")
    if any(k in query.lower() for k in weather_keywords):
        try:
            city = query.lower()
            for w in weather_keywords:
                city = city.replace(w, "").strip()
            city = city.replace("in", "").replace("at", "").strip()
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"https://wttr.in/{city}?format=3")
                if r.status_code == 200:
                    return r.text.strip()
        except Exception:
            pass

    # Source 3 — CoinGecko for crypto prices
    crypto_map = {
        "bitcoin": "bitcoin", "btc": "bitcoin",
        "ethereum": "ethereum", "eth": "ethereum",
        "solana": "solana", "sol": "solana",
        "dogecoin": "dogecoin", "doge": "dogecoin",
    }
    query_lower = query.lower()
    for keyword, coin_id in crypto_map.items():
        if keyword in query_lower:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    r = await client.get(
                        f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                    )
                    data = r.json()
                    price = data.get(coin_id, {}).get("usd")
                    if price:
                        return f"{coin_id.capitalize()} current price: ${price:,.2f} USD"
            except Exception:
                pass

    return None


async def fast_answer(goal: str, cognition: Any) -> str | None:
    """Attempt to answer the user query directly via the fast path engines.

    If the query is INSTANT, checking the web via DuckDuckGo is attempted first.
    If no result is found (or if SIMPLE), the query is forwarded to the cognition backend.
    If the query is COMPLEX, it returns None to trigger the full agent loop.

    Args:
        goal: The user-provided goal string.
        cognition: A cognition backend instance with a ``generate(prompt)`` awaitable.

    Returns:
        The direct string answer from the model or web if successfully evaluated,
        or None if the query must go to the full agent loop.
    """
    complexity = await classify_query(goal)
    
    if complexity == QueryComplexity.COMPLEX:
        logger.info("FastPath: Goal classified as COMPLEX. Bypassing fast path.")
        return None

    logger.info("FastPath: Goal classified as %s. Taking fast path.", complexity.name)

    if complexity == QueryComplexity.INSTANT:
        logger.debug("FastPath: Attempting DuckDuckGo instant answer.")
        web_ans = await web_search(goal)
        if web_ans:
            return f"{web_ans}"

    prompt = f"""Answer this question directly and concisely: {goal}
Use your knowledge. Be brief. Give the answer only."""

    try:
        answer = await cognition.generate(prompt)
        return answer
    except Exception as exc:
        logger.error("FastPath: Execution failed, reverting to main loop: %s", exc)
        return None
