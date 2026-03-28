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


@app.command()
def run(
    goal: Annotated[
        str,
        typer.Option("--goal", "-g", help="Natural language goal for the agent."),
    ],
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", "-n", help="Maximum perceive-think-act steps."),
    ] = 100,
    agent_id: Annotated[
        Optional[str],
        typer.Option("--agent-id", help="Persistent agent identifier."),
    ] = None,
    no_headless: Annotated[
        bool,
        typer.Option("--no-headless", help="Show the browser window."),
    ] = False,
    db_path: Annotated[
        Optional[str],
        typer.Option("--db", help="Override the default SQLite database path."),
    ] = None,
) -> None:
    """
    Run NAYAK autonomously toward a [bold yellow]goal[/bold yellow].

    Example:
        nayak run --goal "Find the top 3 AI robotics companies and save report.md"
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY is not set.\n"
            "Get your free key at [link=https://console.groq.com/keys]console.groq.com/keys[/link]\n"
            "Then run: [italic]set GROQ_API_KEY=gsk-...[/italic]  (Windows)\n"
            "       or: [italic]export GROQ_API_KEY=gsk-...[/italic]  (macOS/Linux)"
        )
        raise typer.Exit(code=1)

    config = AgentConfig(
        goal=goal,
        agent_id=agent_id or "nayak-agent",
        max_steps=max_steps,
        headless=not no_headless,
        groq_api_key=api_key,
        db_path=db_path,
    )

    async def _run() -> None:
        agent = Agent(config)
        outcome = await agent.run()
        console.print(f"\n[bold]Final outcome:[/bold] {outcome}")

    asyncio.run(_run())


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
