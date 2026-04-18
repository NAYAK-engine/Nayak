"""
agent.py — The NAYAK Agent: the core autonomous runtime loop.

Orchestrates the perceive -> think -> act -> remember cycle.

Provider selection via NAYAK_PROVIDER env var:
  nvidia  (default) — NVIDIA NIM cloud API
  ollama            — Ollama local or cloud (no key needed locally)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from nayak.brain import Action, ActionType
from nayak.eyes.browser import Browser, PageState
from nayak.hands.computer import Computer
from nayak.memory.store import MemoryStore

load_dotenv()  # Load .env file if present

logger = logging.getLogger(__name__)
console = Console()


def _setup_file_logging() -> None:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "nayak.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)


_setup_file_logging()

# ──────────────────────────────────────────────────────────────────
# PROVIDER
# ──────────────────────────────────────────────────────────────────
PROVIDER = os.environ.get("NAYAK_PROVIDER", "ollama").lower()

if PROVIDER == "gemini":
    from nayak.brain.gemini import decide, generate, plan
else:
    from nayak.brain.ollama import decide, generate, plan


@dataclass
class AgentConfig:
    goal: str
    agent_id: str = ""
    session_id: str = ""
    max_steps: int = 30
    headless: bool = True
    db_path: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            self.agent_id = "nayak-agent"
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


class Agent:
    """
    NAYAK autonomous agent.

    Perceive -> Think -> Act -> Remember until goal is complete or max_steps hit.

    Example::

        config = AgentConfig(goal="Find the top 3 AI robotics companies")
        agent = Agent(config)
        await agent.run()
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._browser = Browser(headless=config.headless)
        self._memory = MemoryStore(
            agent_id=config.agent_id,
            session_id=config.session_id,
            db_path=config.db_path,
        )

        # Internal state
        self._step = 0
        self._done = False
        self._computer: Computer | None = None

        # FIX 1 — Guaranteed report saving
        self.extracted_content: list[str] = []

        # FIX 2 — Failure / stuck-action tracking
        self._last_action_type: ActionType | None = None
        self._failure_count = 0

        # FIX 2 — URL visit counter (in-memory for O(1) look-up)
        self._url_visit_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> str:
        """
        Execute the perceive -> think -> act -> remember loop.

        FIX 6 — Always calls _force_save_report() before exiting and
        always cleans up the browser / DB connection via finally.
        """
        self._print_banner()
        await self._memory.init()
        await self._browser.start()
        self._computer = Computer(self._browser.page)

        try:
            return await self._run_loop()
        except KeyboardInterrupt:
            logger.warning("Agent stopped by user (KeyboardInterrupt)")
            console.print("\n[NAYAK] Stopped by user.")
            await self._force_save_report()
            return "Stopped by user"
        except Exception as e:
            logger.error("Unhandled exception in agent run loop: %s", e)
            console.print(f"\n[NAYAK] Unexpected error: {e}")
            logger.exception("Unhandled exception in agent run loop")
            await self._force_save_report()
            return f"Error: {e}"
        finally:
            await self._browser.stop()
            await self._memory.close()
            logger.info("Agent shutdown complete")
            console.print("[NAYAK] Agent shutdown complete.")

    async def _run_loop(self) -> str:
        """The main perceive → think → act → remember loop."""
        await self._computer.navigate("https://www.google.com")

        while self._step < self.config.max_steps and not self._done:
            self._step += 1
            steps_remaining = self.config.max_steps - self._step

            # FIX 2 — Hard stop at < 10 steps: save immediately and exit
            if steps_remaining < 10:
                logger.warning("Less than 10 steps remaining — saving and exiting (step %d)", self._step)
                console.print(
                    "[bold red][NAYAK] Less than 10 steps remaining. "
                    "Saving report and exiting.[/bold red]"
                )
                await self._force_save_report()
                return "Saved report — ran out of steps."

            # FIX 2 — Soft warning at < 20 steps
            if steps_remaining < 20:
                logger.warning("Less than 20 steps remaining — prioritising extraction (step %d)", self._step)
                console.print(
                    "[yellow][NAYAK] Less than 20 steps remaining. "
                    "Prioritising extraction.[/yellow]"
                )

            logger.info("--- Step %d / %d ---", self._step, self.config.max_steps)
            console.rule(
                f"[bold cyan]Step {self._step} / {self.config.max_steps}[/bold cyan]"
            )

            # PERCEIVE → THINK → ACT → REMEMBER (120s per-step timeout)
            try:
                action = await asyncio.wait_for(self._run_step(), timeout=120)
            except asyncio.TimeoutError:
                logger.warning("Step %d timed out after 120s", self._step)
                console.print("[bold red][NAYAK] Step timed out. Skipping.[/bold red]")
                self._failure_count += 1
                continue

            if action.type == ActionType.FINISH:
                self._done = True
                await self._force_save_report()
                self._print_finish(action.reason)
                return action.reason

        if not self._done:
            msg = (
                f"Reached max steps ({self.config.max_steps}) "
                "without completing the goal."
            )
            logger.warning(msg)
            console.print(f"\n[yellow][WARNING] {msg}[/yellow]")
            await self._force_save_report()
            return msg

        return "Agent finished."

    # ------------------------------------------------------------------
    # Step runner (used by _run_loop with timeout)
    # ------------------------------------------------------------------

    async def _run_step(self) -> Action:
        """Run one perceive → think → act → remember cycle. Returns the action."""
        # 1. PERCEIVE
        state = await self._perceive()

        # 2. THINK
        action = await self._think(state)

        # 3. ACT
        result = await self._act_with_retry(action)

        # 4. REMEMBER
        await self._remember(action, result)

        return action

    # ------------------------------------------------------------------
    # Loop phases
    # ------------------------------------------------------------------

    async def _perceive(self) -> PageState:
        """Capture current browser state."""
        state = await self._browser.see()
        logger.info("[SEE] url=%s title=%r", state.url[:80], state.title[:50])
        console.print(
            f"[bold][SEE][/bold] [green]OK[/green] | "
            f"[link={state.url}]{state.url[:80]}[/link] | "
            f"'{state.title[:50]}'"
        )
        return state

    async def _think(self, state: PageState) -> Action:
        """Ask the brain for the next action."""
        memory_context = await self._memory.get_recent(n=15)
        context_str = "\n".join(memory_context) if memory_context else "(no prior memory)"
        logger.info("[THINK] step=%d memory_entries=%d", self._step, len(memory_context))
        console.print(
            f"[dim][THINK] Thinking... ({len(memory_context)} memory entries)[/dim]"
        )

        steps_remaining = self.config.max_steps - self._step
        system_notes: list[str] = []

        # FIX 2 — Urgent warning injected into prompt when steps are low
        if steps_remaining < 20:
            system_notes.append(
                "URGENT: Less than 20 steps remaining. "
                "Extract key information now and save report immediately."
            )

        # FIX 2 — Detect URL repeated visits
        current_url = state.url
        visit_count = self._url_visit_counts[current_url]
        if visit_count >= 3:
            logger.warning("URL visited %d times already: %s", visit_count, current_url)
            console.print(
                f"[bold red][NAYAK] Already visited {current_url} "
                f"{visit_count} times. Nudging agent to skip.[/bold red]"
            )
            system_notes.append(
                f"You already visited {current_url} {visit_count} times. "
                "Do not go there again. Move to the next planned step."
            )

        # FIX 2 — Stuck-action detection
        if self._failure_count >= 3:
            logger.warning("Stuck action '%s' — skipping after 3 failures", self._last_action_type)
            console.print(
                f"[NAYAK] Skipping stuck action '{self._last_action_type}' "
                "after 3 failures"
            )
            system_notes.append(
                f"Your last action '{self._last_action_type}' failed "
                "3 times in a row. Skip it and move to the next logical step."
            )
            self._failure_count = 0

        if system_notes:
            note_str = "\n".join([f"[SYSTEM]: {note}" for note in system_notes])
            context_str += f"\n\n{note_str}"

        action = await decide(
            goal=self.config.goal,
            step=self._step,
            url=state.url,
            page_title=state.title,
            page_text=state.text,
            screenshot_b64=state.screenshot_b64,
            memory_context=context_str,
        )
        logger.info("[ACTION] %s", action)
        console.print(
            Panel(
                str(action),
                title="[bold green]Action[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        return action

    async def _act(self, action: Action) -> str:
        """Execute the chosen action and return a result string."""
        computer = self._computer
        assert computer is not None, "Computer not initialised — call run() first"

        match action.type:
            case ActionType.NAVIGATE:
                url = action.url or "https://www.google.com"
                result = await computer.navigate(url)
                # FIX 2 — Track visited URLs
                self._url_visit_counts[self._browser.page.url] += 1

            case ActionType.CLICK:
                if action.selector:
                    result = await computer.click(action.selector)
                elif action.coordinates:
                    x, y = action.coordinates
                    result = await computer.click_coordinates(x, y)
                else:
                    result = "click action missing selector or coordinates — skipped"

            case ActionType.TYPE_TEXT:
                if not action.selector:
                    result = "type_text action missing selector — skipped"
                else:
                    result = await computer.type_text(
                        action.selector, action.text or ""
                    )

            case ActionType.SCROLL:
                result = await computer.scroll(
                    direction=action.direction,
                    amount=action.amount,
                )

            case ActionType.EXTRACT:
                # FIX 1 — Append every extraction to self.extracted_content
                text = await computer.extract()
                self.extracted_content.append(
                    f"[Source: {self._browser.page.url}]\n{text}"
                )
                result = f"Extracted {len(text)} chars from {self._browser.page.url}"

            case ActionType.SAVE_FILE:
                result = await computer.save_file(
                    filename=action.filename or "output.txt",
                    content=action.text or "",
                )
                if not ("failed" in result.lower() or "error" in result.lower()):
                    logger.info("File saved: %s — task complete", action.filename or "output.txt")
                    console.print("[NAYAK] File saved. Task complete.")
                    self._done = True

            case ActionType.SEARCH:
                search_result = await computer.google_search(
                    query=action.text or ""
                )
                # FIX 1 — Search results count as extracted content
                if search_result and "failed" not in search_result.lower():
                    self.extracted_content.append(
                        f"[Google Search: {action.text}]\n{search_result}"
                    )
                result = search_result

            case ActionType.PRESS_KEY:
                if action.key:
                    await computer.page.keyboard.press(action.key)
                    result = f"Pressed key: {action.key}"
                else:
                    result = "press_key action missing key — skipped"

            case ActionType.FINISH:
                result = f"Goal completed: {action.reason}"

            case _:
                result = f"Unknown action type '{action.type}' — skipped"

        # FIX 2 — Consecutive failure tracking
        curr_res = result.lower()
        failure_keywords = ("failed", "error", "not found", "timeout", "could not")
        is_failure = any(k in curr_res for k in failure_keywords)

        if action.type == self._last_action_type and is_failure:
            self._failure_count += 1
        elif not is_failure:
            self._failure_count = 0

        self._last_action_type = action.type

        if is_failure:
            logger.warning("[RESULT] action=%s result=%s", action.type.value, result)
        else:
            logger.info("[RESULT] action=%s result=%s", action.type.value, result)
        console.print(f"[bold blue]Result:[/bold blue] {result}")
        await asyncio.sleep(0.8)
        return result

    async def _act_with_retry(self, action: Action) -> str:
        """Execute an action with up to 3 retries on failure."""
        _no_retry_types = {ActionType.FINISH, ActionType.SAVE_FILE}
        _failure_keywords = ("failed", "error", "timeout", "not found", "could not")

        if action.type in _no_retry_types:
            return await self._act(action)

        result = ""
        for attempt in range(1, 4):
            result = await self._act(action)
            if not any(k in result.lower() for k in _failure_keywords):
                return result
            if attempt < 3:
                logger.warning(
                    "[NAYAK] Action '%s' failed (attempt %d/3): %s — retrying in 1s",
                    action.type.value, attempt, result,
                )
                await asyncio.sleep(1)
        logger.warning(
            "[NAYAK] Action '%s' failed after 3 attempts. Last result: %s",
            action.type.value, result,
        )
        return result

    # ------------------------------------------------------------------
    # FIX 1 — Guaranteed report saving
    # ------------------------------------------------------------------

    async def _force_save_report(self) -> bool:
        """
        Generate and save the final report from all extracted content.

        Called on every exit path — FINISH, steps < 10, KeyboardInterrupt,
        unhandled exception, and max_steps reached.
        """
        if not self.extracted_content:
            logger.warning("No content extracted — skipping report save")
            console.print("[NAYAK] No content extracted yet — skipping report save.")
            return False

        combined = "\n\n".join(self.extracted_content)
        if len(combined) < 100:
            logger.warning("Extracted content too short (%d chars) — skipping report", len(combined))
            console.print("[NAYAK] Extracted content too short — skipping report.")
            return False

        prompt = f"""
Goal was: {self.config.goal}

Content extracted from websites:
{combined}

Write a complete markdown report answering the goal.
Include all relevant information found.
Be detailed and well structured.
Use headers, bullet points, and sections.
"""
        try:
            report = await generate(prompt)
            filename = "report.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Report saved to %s", filename)
            console.print(f"[bold green][NAYAK] Report saved to {filename}[/bold green]")
            return True
        except Exception as e:
            logger.error("_force_save_report() failed: %s", e)
            console.print(f"[NAYAK] Failed to generate/save report: {e}")
            return False

    async def _remember(self, action: Action, result: str) -> None:
        """Persist step data to SQLite memory."""
        await self._memory.save(
            step=self._step,
            action=str(action),
            result=result,
            goal=self.config.goal,
        )

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        console.print(
            Panel(
                Text(self.config.goal, style="bold yellow"),
                title=Text("NAYAK", style="bold white on blue"),
                subtitle=Text(
                    f"Session: {self.config.session_id[:8]}…  "
                    f"Agent: {self.config.agent_id}  "
                    f"Max steps: {self.config.max_steps}  "
                    f"Provider: {PROVIDER.upper()}",
                    style="dim",
                ),
                border_style="blue",
                padding=(1, 4),
            )
        )

    def _print_finish(self, reason: str) -> None:
        console.print(
            Panel(
                Text(reason, style="bold green"),
                title="[DONE] GOAL COMPLETED",
                border_style="green",
                padding=(1, 4),
            )
        )
