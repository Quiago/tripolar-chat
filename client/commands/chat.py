"""fmind chat – interactive and one-shot chat with streaming."""

import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from client import config as cfg
from client.api import APIClient, APIError, get_client

console = Console()
app = typer.Typer(help="Chat with an LLM.")

_SLASH_COMMANDS = """[dim]Commands: /models · /switch <model> · /clear · /history · /exit[/dim]"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_model(api: APIClient, requested: Optional[str]) -> str:
    """Return the model to use: explicit arg > config default > first in catalog."""
    if requested:
        return requested
    default = cfg.get_default_model()
    if default:
        return default
    try:
        models = api.list_models()
        if models:
            return models[0]["id"]
    except APIError:
        pass
    return "llama-3.1-8b-awq"


def _print_models_table(api: APIClient) -> None:
    from rich.table import Table

    try:
        models = api.list_models()
    except APIError as exc:
        console.print(f"[red]Could not fetch models:[/red] {exc}")
        return

    table = Table(title="Available Models", show_lines=False)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Cached", justify="center")
    table.add_column("Loaded", justify="center")

    for m in models:
        table.add_row(
            m["id"],
            m.get("description", ""),
            "✓" if m.get("cached") else "–",
            "[green]●[/green]" if m.get("loaded") else "○",
        )
    console.print(table)


def _stream_response(api: APIClient, model: str, messages: list) -> str:
    """Send messages, stream the response to terminal. Returns full text."""
    console.print()
    label = Text(f"[{model}] ", style="bold green")
    console.print(label, end="")

    full = ""
    first = True

    with console.status(
        f"[dim]Loading {model}… (first load may take a few minutes)[/dim]",
        spinner="dots",
    ) as status:
        try:
            for chunk in api.chat_stream(model=model, messages=messages):
                if first:
                    status.stop()
                    first = False
                console.print(chunk, end="", highlight=False, markup=False)
                full += chunk
        except APIError as exc:
            if first:
                status.stop()
            console.print(f"\n[red]Error:[/red] {exc}")
            return ""
        except KeyboardInterrupt:
            if first:
                status.stop()
            console.print("\n[yellow]Interrupted.[/yellow]")
            return full

    console.print()  # newline after streamed content
    return full


# ── commands ──────────────────────────────────────────────────────────────────

@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None, "--model", "-M", help="Model to use (e.g. llama-3.1-8b-awq)."
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="One-shot prompt; exits after responding."
    ),
):
    """Chat interactively with an LLM, or ask a single question with -m."""
    try:
        api = get_client()
    except APIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    current_model = _pick_model(api, model)

    # ── one-shot mode ─────────────────────────────────────────────────────────
    if message:
        messages = [{"role": "user", "content": message}]
        response = _stream_response(api, current_model, messages)
        if response:
            # Render final markdown after streaming
            console.print(Rule(style="dim"))
            console.print(Markdown(response))
        return

    # ── interactive mode ──────────────────────────────────────────────────────
    server_url = cfg.get_server_url()
    console.print(
        Panel(
            f"[bold]FactoryMind AI[/bold]\n"
            f"Server: [dim]{server_url}[/dim]\n"
            f"Model:  [cyan]{current_model}[/cyan]\n\n"
            + _SLASH_COMMANDS,
            expand=False,
        )
    )

    history: list[dict] = []
    chat_id: Optional[str] = None

    while True:
        try:
            user_input = console.input(
                f"\n[bold blue]\\[{current_model}] You:[/bold blue] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd == "/clear":
                os.system("clear" if sys.platform != "win32" else "cls")
                history.clear()
                chat_id = None
                console.print("[dim]Screen cleared. History reset.[/dim]")

            elif cmd == "/models":
                _print_models_table(api)

            elif cmd == "/switch":
                if len(parts) < 2:
                    console.print("[red]Usage:[/red] /switch <model-name>")
                else:
                    current_model = parts[1].strip()
                    history.clear()
                    chat_id = None
                    cfg.save_default_model(current_model)
                    console.print(
                        f"[green]Switched to[/green] [cyan]{current_model}[/cyan]. "
                        "History cleared."
                    )

            elif cmd == "/history":
                if not history:
                    console.print("[dim]No messages yet.[/dim]")
                else:
                    for msg in history:
                        role = "You" if msg["role"] == "user" else current_model
                        style = "blue" if msg["role"] == "user" else "green"
                        console.print(f"[{style}]{role}:[/{style}] {msg['content']}")

            else:
                console.print(
                    f"[yellow]Unknown command:[/yellow] {cmd}\n" + _SLASH_COMMANDS
                )
            continue

        # ── normal message ────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        response = _stream_response(api, current_model, history)
        if response:
            history.append({"role": "assistant", "content": response})
