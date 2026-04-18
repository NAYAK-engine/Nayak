"""
cli.py — NAYAK command-line interface.

Entry point: nayak  (registered in pyproject.toml as nayak = "nayak.cli:app")

Commands:
  nayak run     --goal "..." [--max-steps N] [--agent-id ID] [--no-headless]
  nayak history --agent-id ID [--limit N]
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from nayak.agent import Agent, AgentConfig
from nayak.memory.store import MemoryStore

load_dotenv()  # Load .env automatically

app = typer.Typer(
    name="nayak",
    help="NAYAK — Open source runtime engine for autonomous robots and AI agents.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


def _check_provider() -> bool:
    """
    Validate that the selected provider has what it needs.
    Since we are only using Ollama, we only check for cloud API key if in cloud mode.
    """
    mode = os.environ.get("OLLAMA_MODE", "local").lower()
    if mode == "cloud":
        key = os.environ.get("OLLAMA_API_KEY", "")
        if not key:
            console.print(
                "[bold red]Error:[/bold red] OLLAMA_MODE=cloud requires OLLAMA_API_KEY.\n"
                "Either set OLLAMA_API_KEY or switch to OLLAMA_MODE=local."
            )
            return False
    return True


def calculate_steps(goal: str) -> int:
    goal_lower = goal.lower()
    
    # Simple tasks — fast
    simple_keywords = [
        "what", "who", "when", "where",
        "find", "search", "look up", "check"
    ]
    
    # Medium tasks
    medium_keywords = [
        "visit", "read", "compare", 
        "list", "summarize", "explain"
    ]
    
    # Complex tasks — need more steps
    complex_keywords = [
        "research", "analyze", "report",
        "save", "write", "collect", 
        "multiple", "all", "every"
    ]
    
    word_count = len(goal.split())
    
    if any(k in goal_lower for k in complex_keywords):
        base = 80
    elif any(k in goal_lower for k in medium_keywords):
        base = 40
    elif any(k in goal_lower for k in simple_keywords):
        base = 20
    else:
        base = 40
    
    # More words = more complex
    if word_count > 20:
        base += 20
    elif word_count > 10:
        base += 10
    
    return min(base, 100)


@app.command()
def run(goal: str):
    """Run NAYAK with your goal."""
    steps = calculate_steps(goal)
    print(f"[NAYAK] Goal received.")
    print(f"[NAYAK] Complexity: {steps} steps allocated.")
    print(f"[NAYAK] Starting now...")
    
    if not _check_provider():
        raise typer.Exit(code=1)

    config = AgentConfig(
        goal=goal,
        agent_id="nayak-agent",
        max_steps=steps,
        headless=True,
        db_path=None,
    )

    async def _run() -> None:
        agent = Agent(config)
        outcome = await agent.run()
        console.print(f"\n[bold]Final outcome:[/bold] {outcome}")

    asyncio.run(_run())


@app.command()
def ask(question: str):
    """Quick single question answers."""
    steps = 20
    print(f"[NAYAK] Question received.")
    print(f"[NAYAK] Complexity: {steps} steps allocated (Fast).")
    print(f"[NAYAK] Processing...")

    if not _check_provider():
        raise typer.Exit(code=1)

    config = AgentConfig(
        goal=question,
        agent_id="nayak-ask-agent",
        max_steps=steps,
        headless=True,
        db_path=None,
    )

    async def _ask() -> None:
        agent = Agent(config)
        outcome = await agent.run()
        console.print(f"\n[bold]Answer:[/bold] {outcome}")

    asyncio.run(_ask())


@app.command()
def history(
    agent_id: Annotated[
        str,
        typer.Option("--agent-id", help="Agent ID to retrieve history for."),
    ] = "nayak-agent",
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max recent entries to display."),
    ] = 30,
    db_path: Annotated[
        Optional[str],
        typer.Option("--db", help="Override the default SQLite database path."),
    ] = None,
) -> None:
    """
    Show past session history for an agent.

    Example:
        nayak history --agent-id my-robot --limit 50
    """

    async def _history() -> None:
        store = MemoryStore(
            agent_id=agent_id,
            session_id="__cli_query__",
            db_path=db_path,
        )
        await store.init()

        sessions = await store.list_sessions()
        if not sessions:
            console.print(f"[yellow]No history found for agent '{agent_id}'.[/yellow]")
            await store.close()
            return

        console.print(f"\n[bold cyan]Sessions — agent:[/bold cyan] [bold]{agent_id}[/bold]\n")
        t = Table(show_header=True, header_style="bold magenta")
        t.add_column("Session ID", style="dim", width=38)
        t.add_column("Started")
        t.add_column("Last Active")
        t.add_column("Steps", justify="right")
        t.add_column("Goal", max_width=60)
        for s in sessions:
            t.add_row(
                s["session_id"],
                str(s["started_at"])[:19],
                str(s["last_ts"])[:19],
                str(s["steps"]),
                s["goal"],
            )
        console.print(t)

        recent_lines = await store.get_recent(n=limit)
        console.print(f"\n[bold cyan]Last {limit} entries:[/bold cyan]\n")
        for line in recent_lines:
            console.print(f"  {line}")

        await store.close()

    asyncio.run(_history())


if __name__ == "__main__":
    app()
