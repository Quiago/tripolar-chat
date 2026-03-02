"""fmind models – list and interactively select models."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from client import config as cfg
from client.api import APIError, get_client

console = Console()
app = typer.Typer(help="Manage available LLM models.")


def _fetch_and_display(interactive: bool) -> None:
    try:
        api = get_client()
    except APIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    try:
        models = api.list_models()
    except APIError as exc:
        console.print(f"[red]Could not fetch models:[/red] {exc}")
        raise typer.Exit(1)

    if not models:
        console.print("[yellow]No models found in the server catalog.[/yellow]")
        return

    if not interactive:
        table = Table(title="Available Models", show_lines=False)
        table.add_column("#", style="dim", justify="right")
        table.add_column("Model ID", style="bold cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Cached", justify="center")
        table.add_column("Loaded", justify="center")

        for i, m in enumerate(models, 1):
            table.add_row(
                str(i),
                m["id"],
                m.get("description", ""),
                "✓" if m.get("cached") else "–",
                "[green]●[/green]" if m.get("loaded") else "○",
            )
        console.print(table)
        return

    # ── interactive picker ────────────────────────────────────────────────────
    console.print("\n[bold]Select a model[/bold] (sets default for future chats)\n")
    for i, m in enumerate(models, 1):
        loaded = "[green]● loaded[/green]" if m.get("loaded") else ""
        cached = "[dim]cached[/dim]" if m.get("cached") else ""
        tags = " ".join(filter(None, [loaded, cached]))
        console.print(
            f"  [bold cyan][{i}][/bold cyan] {m['id']:<30} {m.get('description', '')} {tags}"
        )

    console.print()
    choice = typer.prompt(f"Select model [1-{len(models)}]", default="1")
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(models)):
            raise ValueError
    except ValueError:
        console.print("[red]Invalid selection.[/red]")
        raise typer.Exit(1)

    selected = models[idx]["id"]
    cfg.save_default_model(selected)
    console.print(f"\n[green]Default model set to[/green] [cyan]{selected}[/cyan]")
    console.print(f"[dim]Run [bold]fmind chat[/bold] to start chatting.[/dim]")


@app.command()
def models(
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactively select and save a default model.",
    ),
):
    """List available models. Use --interactive to pick a default."""
    _fetch_and_display(interactive)
