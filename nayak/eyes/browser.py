"""
eyes/browser.py — Browser perception layer powered by Playwright.

Launches a real Chromium instance, captures screenshots (base64 PNG),
and extracts structured page content for the agent's perceive step.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

from playwright.async_api import (
    Browser as PlaywrightBrowser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from nayak.perception.base import PerceptionBase

logger = logging.getLogger(__name__)


@dataclass
class PageState:
    """A snapshot of the world at a single point in time."""

    url: str
    title: str
    text: str                          # Visible page text (no HTML)
    screenshot_b64: str | None = None  # Base-64 PNG, may be None on error
    links: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None

    def __str__(self) -> str:
        snippet = self.text[:120].replace("\n", " ")
        ok = "OK" if not self.error else f"ERROR:{self.error}"
        return f"PageState({ok} url={self.url!r} title={self.title!r} text={snippet!r}…)"


class Browser(PerceptionBase):
    """
    Async Playwright wrapper used by the agent as its 'eyes'.

    Usage::

        browser = Browser(headless=True)
        await browser.start()
        state = await browser.see()
        await browser.stop()
    """

    _VIEWPORT = {"width": 1280, "height": 900}
    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    @property
    def name(self) -> str:
        return "browser-perception"

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Playwright | None = None
        self._browser: PlaywrightBrowser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Initialize perception — calls start()."""
        await self.start()

    async def start(self) -> None:
        """Launch Chromium and open a blank page."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._context = await self._browser.new_context(
            viewport=self._VIEWPORT,
            user_agent=self._USER_AGENT,
        )
        self._page = await self._context.new_page()
        logger.info("Browser started (headless=%s)", self._headless)
        await self.register()

    async def stop(self) -> None:
        """Gracefully shut down browser and Playwright."""
        from nayak.core import registry, ModuleStatus
        registry.set_status(self.name, ModuleStatus.STOPPED)
        
        if self._page and not self._page.is_closed():
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser stopped")

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser.start() has not been called yet.")
        return self._page

    async def see(self) -> PageState:
        """
        Capture the full world-state of the current page.
        Returns URL, title, visible text, base64 PNG screenshot, and links.
        """
        try:
            url = self._page.url if self._page else "about:blank"
            title = await self.page.title()
            text = await self._extract_text()
            screenshot_b64 = await self._take_screenshot()
            links = await self._extract_links()
            return PageState(
                url=url,
                title=title,
                text=text,
                screenshot_b64=screenshot_b64,
                links=links,
            )
        except Exception as exc:
            logger.error("see() failed: %s", exc)
            return PageState(
                url=self._page.url if self._page else "unknown",
                title="error",
                text="",
                screenshot_b64=None,
                error=str(exc),
            )

    async def _take_screenshot(self) -> str | None:
        """Return a PNG screenshot of the viewport as a Base64 string."""
        try:
            png_bytes: bytes = await self.page.screenshot(type="png", full_page=False)
            return base64.b64encode(png_bytes).decode("ascii")
        except Exception as exc:
            logger.warning("Screenshot failed: %s", exc)
            return None

    async def _extract_text(self) -> str:
        """Extract all readable text from the page, deduplicated."""
        try:
            text: str = await self.page.evaluate(
                """() => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode(node) {
                                const p = node.parentElement;
                                if (!p) return NodeFilter.FILTER_REJECT;
                                const tag = p.tagName.toLowerCase();
                                if (['script','style','noscript','svg'].includes(tag))
                                    return NodeFilter.FILTER_REJECT;
                                return node.textContent.trim().length > 1
                                    ? NodeFilter.FILTER_ACCEPT
                                    : NodeFilter.FILTER_SKIP;
                            }
                        }
                    );
                    const seen = new Set();
                    const chunks = [];
                    let node;
                    while ((node = walker.nextNode())) {
                        const t = node.textContent.trim();
                        if (!seen.has(t)) { seen.add(t); chunks.push(t); }
                    }
                    return chunks.join('\\n');
                }"""
            )
            return text
        except Exception as exc:
            logger.warning("Text extraction failed: %s", exc)
            return ""

    async def _extract_links(self) -> list[dict[str, str]]:
        """Return [{text, href}] for up to 60 anchor elements."""
        try:
            links: list[dict[str, str]] = await self.page.evaluate(
                """() => Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({ text: a.innerText.trim(), href: a.href }))
                    .filter(l => l.href.startsWith('http') && l.text.length > 0)
                    .slice(0, 60)"""
            )
            return links
        except Exception as exc:
            logger.warning("Link extraction failed: %s", exc)
            return []
