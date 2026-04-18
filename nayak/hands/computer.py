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

    @property
    def page(self) -> Page:
        """Expose the underlying Playwright Page (used by agent for direct key presses)."""
        return self._page

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
    # Clicking — FIX 4: Four-strategy reliable click
    # ------------------------------------------------------------------

    async def click(self, selector: str) -> str:
        """
        Click an element using four progressively softer strategies:
        1. Direct CSS selector click (strict, fast)
        2. wait_for_selector then element.click()
        3. get_by_text match (useful when selector looks like label text)
        4. JavaScript evaluate click (last resort, no-op safe)
        """
        # Strategy 1: direct CSS selector click
        try:
            await self._page.click(selector, timeout=3_000)
            await asyncio.sleep(1)
            logger.info("Clicked (strategy 1 — direct): %s", selector)
            return f"Clicked '{selector}'"
        except Exception:
            pass

        # Strategy 2: wait_for_selector then locator.click()
        try:
            el = await self._page.wait_for_selector(selector, timeout=3_000)
            if el:
                await el.click()
                await asyncio.sleep(1)
                logger.info("Clicked (strategy 2 — wait+click): %s", selector)
                return f"Clicked '{selector}' (waited for element)"
        except Exception:
            pass

        # Strategy 3: get_by_text — handy when the selector is human-readable text
        try:
            await self._page.get_by_text(selector).first.click()
            await asyncio.sleep(1)
            logger.info("Clicked (strategy 3 — by text): %s", selector)
            return f"Clicked visible text '{selector}'"
        except Exception:
            pass

        # Strategy 4: JavaScript evaluate click
        try:
            await self._page.evaluate(
                f"document.querySelector('{selector}')?.click()"
            )
            await asyncio.sleep(1)
            logger.info("Clicked (strategy 4 — JS eval): %s", selector)
            return f"Clicked '{selector}' via JavaScript"
        except Exception as exc:
            logger.error("click(%s) all strategies exhausted: %s", selector, exc)
            return f"Could not click: {selector}"

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
    # Content extraction — FIX 5: clean extract() method
    # ------------------------------------------------------------------

    async def extract(self) -> str:
        """
        Extract clean, useful text from the current page.

        Removes noise elements (scripts, styles, nav, footer, cookie banners,
        ads) before capturing inner text, then filters short/blank lines and
        caps output at 6 000 characters.
        """
        try:
            # Strip noisy elements from the live DOM
            await self._page.evaluate(
                """
                document.querySelectorAll(
                    'script, style, nav, footer, ' +
                    '.cookie-banner, .advertisement, ' +
                    '[class*="cookie"], [class*="banner"], ' +
                    '[id*="cookie"], [id*="ad"]'
                ).forEach(el => el.remove())
                """
            )

            # Grab body text after cleaning
            content = await self._page.inner_text("body")

            # Filter empty / very short lines
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and len(line.strip()) > 20
            ]
            clean = "\n".join(lines)

            logger.info("extract() → %d chars (cleaned)", len(clean))
            return clean[:6_000]

        except Exception as exc:
            logger.error("extract() error: %s", exc)
            return f"Extract failed: {exc}"

    async def extract_text(self) -> str:
        """
        Legacy alias — returns full visible text without stripping noise.

        Prefer extract() for new code.
        """
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
            logger.error("save_file(%s) error: %s", filename, exc)
            return f"save_file error: {exc}"

    # ------------------------------------------------------------------
    # Specialized Flows — FIX 3: Reliable Google search
    # ------------------------------------------------------------------

    async def google_search(self, query: str) -> str:
        """
        Navigate to Google and perform a search, returning up to 5 000 chars
        of results page body text.

        Tries three input selectors in priority order so it works even when
        Google serves the textarea vs input variant.
        """
        try:
            await self.navigate("https://www.google.com")
            await asyncio.sleep(2)

            # Try search-box selectors in order of preference
            selectors = [
                "textarea[name='q']",
                "input[name='q']",
                "input[type='search']",
            ]

            typed = False
            for sel in selectors:
                try:
                    await self._page.wait_for_selector(sel, timeout=3_000)
                    await self._page.fill(sel, query)
                    typed = True
                    logger.info("Filled search box (%s) with: %s", sel, query)
                    break
                except Exception:
                    continue

            if not typed:
                logger.error("google_search: no search box found for query: %s", query)
                return "Search box not found — could not perform Google search"

            await self._page.keyboard.press("Enter")
            await asyncio.sleep(3)

            # Return extracted results page text (up to 5 000 chars)
            content = await self._page.inner_text("body")
            logger.info("google_search('%s') → %d chars returned", query, len(content))
            return content[:5_000]

        except Exception as exc:
            logger.error("google_search() error: %s", exc)
            return f"Search failed: {exc}"
