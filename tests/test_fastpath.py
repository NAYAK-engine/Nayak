"""
tests/test_fastpath.py — Unit tests for the NAYAK Fast Path engine.

Validates the keyword-based query classification heuristics that route
user goals into INSTANT, SIMPLE, or COMPLEX execution tracks.
"""

import pytest

from nayak.cognition.fastpath import classify_query, QueryComplexity


class TestFastPath:
    """Tests for the cognition fast-path classifier."""

    @pytest.mark.asyncio
    async def test_bitcoin_is_instant(self) -> None:
        """A price query should be classified as INSTANT."""
        result = await classify_query("What is the price of Bitcoin?")
        assert result == QueryComplexity.INSTANT

    @pytest.mark.asyncio
    async def test_research_is_complex(self) -> None:
        """A research query should be classified as COMPLEX."""
        result = await classify_query(
            "Research the top 10 AI companies and write a report"
        )
        assert result == QueryComplexity.COMPLEX

    @pytest.mark.asyncio
    async def test_weather_is_instant(self) -> None:
        """A weather query should be classified as INSTANT."""
        result = await classify_query("What is the weather in Tokyo?")
        assert result == QueryComplexity.INSTANT

    @pytest.mark.asyncio
    async def test_unknown_is_simple(self) -> None:
        """A generic query should default to SIMPLE."""
        result = await classify_query("Tell me something interesting")
        assert result == QueryComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_define_is_instant(self) -> None:
        """A definition query should be classified as INSTANT."""
        result = await classify_query("Define machine learning")
        assert result == QueryComplexity.INSTANT

    @pytest.mark.asyncio
    async def test_analyze_is_complex(self) -> None:
        """An analysis query should be classified as COMPLEX."""
        result = await classify_query("Analyze the stock market trends for 2025")
        assert result == QueryComplexity.COMPLEX
