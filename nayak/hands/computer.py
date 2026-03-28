"""
hands/computer.py — Physical execution layer powered by Playwright.

Translates Action objects into real browser interactions.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 8_000   # ms to wait for elements


class Computer:
    """
    Execution engine wrapping a Playwright Page.

    Every public method returns a short English description of the result,
    which the agent stores in memory.
    """

    def __init__(self, page: Page, timeout_ms: int = _DEFAULT_TIMEOUT) -> None:
        self._page = page
        self._timeout = timeout_ms

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> str:
        """Go to *url*. Returns result description."""
        try:
            response = await self._page.goto(
                url, wait_until="domcontentloaded", timeout=30_000
            )
            status = response.status if response else "?"
            final_url = self._page.url
            logger.info("Navigated to %s (HTTP %s)", final_url, status)
            return f"Navigated to {final_url} — HTTP {status}"
        except PlaywrightTimeoutError:
            return f"Navigation to {url} timed out; page may be partially loaded"
        except Exception as exc:
            logger.error("navigate(%s) error: %s", url, exc)
            return f"Navigation error: {exc}"

    # ------------------------------------------------------------------
    # Clicking
    # ------------------------------------------------------------------

    async def click(self, selector: str) -> str:
        """Click an element by CSS selector with aria-label / text fallbacks."""
        try:
            await self._page.wait_for_selector(selector, timeout=self._timeout)
            await self._page.click(selector, timeout=self._timeout)
            logger.info("Clicked selector: %s", selector)
            return f"Clicked '{selector}'"
        except PlaywrightTimeoutError:
            # Fallback 1: aria-label
            try:
                await self._page.get_by_label(selector).click(timeout=self._timeout)
                return f"Clicked aria-label '{selector}'"
            except Exception:
                pass
            # Fallback 2: visible text
            try:
                await self._page.get_by_text(selector, exact=False).first.click(
                    timeout=self._timeout
                )
                return f"Clicked visible text '{selector}'"
            except Exception:
                pass
            return f"Element '{selector}' not found for click"
        except Exception as exc:
            logger.error("click(%s) error: %s", selector, exc)
            return f"Click error: {exc}"

    async def click_coordinates(self, x: int, y: int) -> str:
        """Click at pixel coordinates (x, y)."""
        try:
            await self._page.mouse.click(x, y)
            return f"Clicked at ({x}, {y})"
        except Exception as exc:
            logger.error("click_coordinates(%d,%d) error: %s", x, y, exc)
            return f"Click-at-coordinates error: {exc}"

    # ------------------------------------------------------------------
    # Typing
    # ------------------------------------------------------------------

    async def type_text(self, selector: str, text: str) -> str:
        """Click *selector*, clear it, and type *text*."""
        try:
            await self._page.wait_for_selector(selector, timeout=self._timeout)
            await self._page.click(selector)
            await self._page.fill(selector, text)
            logger.info("Typed %d chars into '%s'", len(text), selector)
            return f"Typed {len(text)} char(s) into '{selector}'"
        except PlaywrightTimeoutError:
            return f"type_text: '{selector}' not found within timeout"
        except Exception as exc:
            logger.error("type_text(%s) error: %s", selector, exc)
            return f"type_text error: {exc}"

    async def press_key(self, key: str) -> str:
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        try:
            await self._page.keyboard.press(key)
            return f"Pressed '{key}'"
        except Exception as exc:
            return f"press_key error: {exc}"

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        """Scroll the page by *amount* pixels in *direction*."""
        delta = amount if direction == "down" else -amount
        try:
            await self._page.evaluate(f"window.scrollBy(0, {delta})")
            return f"Scrolled {direction} {amount}px"
        except Exception as exc:
            logger.error("scroll() error: %s", exc)
            return f"Scroll error: {exc}"

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    async def extract_text(self) -> str:
        """Return full visible text of the current page."""
        try:
            text: str = await self._page.evaluate(
                "() => document.body.innerText"
            )
            lines = text.splitlines()
            cleaned: list[str] = []
            prev_blank = False
            for line in lines:
                stripped = line.strip()
                is_blank = not stripped
                if is_blank and prev_blank:
                    continue
                cleaned.append(stripped)
                prev_blank = is_blank
            return "\n".join(cleaned)
        except Exception as exc:
            logger.error("extract_text() error: %s", exc)
            return f"(extraction error: {exc})"

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    async def save_file(self, filename: str, content: str) -> str:
        """Write *content* to *filename*. Creates parent dirs as needed."""
        try:
            path = Path(filename).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            size_kb = path.stat().st_size / 1024
            logger.info("Saved '%s' (%.1f KB)", path, size_kb)
            return f"Saved {size_kb:.1f} KB → '{path}'"
        except Exception as exc:
            logger.error("save_file() error: %s", exc)
            return f"save_file error: {exc}"

    # ------------------------------------------------------------------
    # Specialized Flows
    # ------------------------------------------------------------------

    async def google_search(self, query: str) -> str:
        """Perform a rigid Google search flow."""
        try:
            await self.navigate("https://www.google.com")
            await asyncio.sleep(2)
            await self.type_text("textarea[name='q']", query)
            await asyncio.sleep(1)
            await self.press_key("Enter")
            await asyncio.sleep(3)
            return f"Searched Google for: {query}"
        except Exception as exc:
            logger.error("google_search() error: %s", exc)
            return f"Google search error: {exc}"

        except Exception as exc:
            logger.error("save_file(%s) error: %s", filename, exc)
            return f"save_file error: {exc}"
