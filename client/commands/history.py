"""fmind history – view past chat sessions."""

from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from client.api import APIError, get_client

console = Console()
app = typer.Typer(help="Browse chat history.")


@app.command()
def history(
    chat_id: Optional[str] = typer.Argument(
        None, help="Chat session ID to display in full. Omit to list recent sessions."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to list."),
):
    """List recent chat sessions, or view a specific session's messages."""
    try:
        api = get_client()
    except APIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if chat_id:
        _show_chat(api, chat_id)
    else:
        _list_sessions(api, limit)


def _list_sessions(api, limit: int) -> None:
    try:
        sessions = api.list_history(limit=limit)
    except APIError as exc:
        console.print(f"[red]Could not fetch history:[/red] {exc}")
        raise typer.Exit(1)

    if not sessions:
        console.print("[dim]No chat sessions found.[/dim]")
        return

    table = Table(title="Recent Chats", show_lines=False)
    table.add_column("ID", style="dim", no_wrap=True, max_width=12)
    table.add_column("Title", style="bold")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Updated", no_wrap=True)

    for s in sessions:
        short_id = s["id"][:8] + "…"
        updated = s["updated_at"][:16].replace("T", " ")
        table.add_row(short_id, s["title"], s["model_used"], updated)

    console.print(table)
    console.print(
        "[dim]Run [bold]fmind history <id>[/bold] to view a session's messages.[/dim]"
    )


def _show_chat(api, chat_id: str) -> None:
    try:
        chat = api.get_chat(chat_id)
    except APIError as exc:
        console.print(f"[red]Could not fetch chat:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]{chat['title']}[/bold]\n"
            f"Model: [cyan]{chat['model_used']}[/cyan]  "
            f"Started: [dim]{chat['created_at'][:16].replace('T', ' ')}[/dim]",
            expand=False,
        )
    )

    for msg in chat.get("messages", []):
        if msg["role"] == "user":
            console.print(f"\n[bold blue]You:[/bold blue]")
            console.print(msg["content"])
        else:
            console.print(f"\n[bold green]{chat['model_used']}:[/bold green]")
            console.print(Markdown(msg["content"]))
