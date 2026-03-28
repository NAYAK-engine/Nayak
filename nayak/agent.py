"""
agent.py — The NAYAK Agent: the core autonomous runtime loop.

Orchestrates the perceive → think → act → remember cycle.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from nayak.brain.groq import Action, ActionType, Brain
from nayak.eyes.browser import Browser, PageState
from nayak.hands.computer import Computer
from nayak.memory.store import MemoryStore

load_dotenv()  # Load .env file if present

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class AgentConfig:
    goal: str
    agent_id: str = ""
    session_id: str = ""
    max_steps: int = 30
    headless: bool = True
    groq_api_key: str | None = None
    db_path: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            self.agent_id = "nayak-agent"
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


class Agent:
    """
    NAYAK autonomous agent.

    Perceive → Think → Act → Remember until goal is complete or max_steps hit.

    Example::

        config = AgentConfig(goal="Find the top 3 AI robotics companies")
        agent = Agent(config)
        await agent.run()
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._brain = Brain(api_key=config.groq_api_key)
        self._browser = Browser(headless=config.headless)
        self._memory = MemoryStore(
            agent_id=config.agent_id,
            session_id=config.session_id,
            db_path=config.db_path,
        )
        self._computer: Computer | None = None
        self._step = 0
        self._done = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> str:
        """
        Execute the perceive→think→act→remember loop.
        Returns a short description of the final outcome.
        """
        self._print_banner()
        await self._memory.init()
        await self._browser.start()
        self._computer = Computer(self._browser.page)

        try:
            await self._computer.navigate("https://www.google.com")

            while self._step < self.config.max_steps and not self._done:
                self._step += 1
                console.rule(
                    f"[bold cyan]Step {self._step} / {self.config.max_steps}[/bold cyan]"
                )

                # 1. PERCEIVE
                state = await self._perceive()

                # 2. THINK
                action = await self._think(state)

                # 3. ACT
                result = await self._act(action)

                # 4. REMEMBER
                await self._remember(action, result)

                if action.type == ActionType.FINISH:
                    self._done = True
                    self._print_finish(action.reason)
                    return action.reason

            if not self._done:
                msg = (
                    f"Reached max steps ({self.config.max_steps}) "
                    "without completing the goal."
                )
                console.print(f"\n[yellow][WARNING] {msg}[/yellow]")
                return msg

        finally:
            await self._browser.stop()
            await self._memory.close()

        return "Agent finished."

    # ------------------------------------------------------------------
    # Loop phases
    # ------------------------------------------------------------------

    async def _perceive(self) -> PageState:
        """Capture current browser state."""
        state = await self._browser.see()
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
        console.print(
            f"[dim][THINK] Thinking... ({len(memory_context)} memory entries)[/dim]"
        )
        # Inject auto-save warning if running out of steps
        if self._step >= self.config.max_steps - 10:
            urgent_msg = (
                "URGENT: You have less than 10 steps remaining. "
                "Save your report NOW using save_file action then finish."
            )
            context_str += f"\n\n[SYSTEM]: {urgent_msg}"
            console.print(f"[bold red]{urgent_msg}[/bold red]")

        action = self._brain.decide(
            goal=self.config.goal,
            step=self._step,
            url=state.url,
            page_title=state.title,
            page_text=state.text,
            memory_context=context_str,
        )
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
                result = await computer.navigate(action.url or "https://www.google.com")

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
                text = await computer.extract_text()
                result = f"Extracted {len(text)} chars: {text[:200]}…"

            case ActionType.SAVE_FILE:
                result = await computer.save_file(
                    filename=action.filename or "output.txt",
                    content=action.content or action.text or "",
                )

            case ActionType.FINISH:
                result = f"Goal completed: {action.reason}"

            case _:
                result = f"Unknown action type '{action.type}' — skipped"

        console.print(f"[bold blue]Result:[/bold blue] {result}")
        await asyncio.sleep(0.8)
        return result

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
                    f"Max steps: {self.config.max_steps}",
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
