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

import client.api as fmapi
from client import config as cfg
from client.api import APIClient, APIError, get_client

console = Console()
app = typer.Typer(help="Chat with an LLM.")

_SLASH_COMMANDS = "[dim]Commands: /models · /switch <model> · /clear · /history · /exit[/dim]"


# ── Internal signal ────────────────────────────────────────────────────────────

class _SessionExpired(Exception):
    """Raised by _stream_response when the server returns 403."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _relogin(server_url: str) -> Optional[APIClient]:
    """Interactive re-login prompt. Returns a new APIClient or None if user declines."""
    console.print(
        "\n[bold yellow]⚠  Session expired[/bold yellow] – "
        "the server rejected your API key (403 Forbidden).\n"
        "You need to log in again to continue.\n"
    )
    if not typer.confirm("Log in now?", default=True):
        console.print("[dim]Run [bold]fmind login[/bold] when ready.[/dim]")
        return None

    username = typer.prompt("Username")
    password = typer.prompt("Password", hide_input=True)
    try:
        user = fmapi.login(server_url, username, password)
        cfg.save_credentials(server_url, user["api_key"])
        console.print(f"[green]✓ Logged in as[/green] [bold]{user['username']}[/bold]\n")
        return APIClient(server_url, user["api_key"])
    except APIError as exc:
        console.print(f"[red]Login failed:[/red] {exc}")
        return None


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
        if exc.status_code == 403:
            console.print("[yellow]⚠ Session expired.[/yellow] Run [bold]fmind login[/bold].")
        else:
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
    """Send messages, stream the response to terminal. Returns full text.

    Raises _SessionExpired if the server returns 403.
    """
    import time as _time

    console.print()

    full = ""
    first = True
    start = _time.monotonic()

    with console.status(
        f"[dim]⏳ {model} – thinking…[/dim]",
        spinner="dots",
    ) as status:
        try:
            for chunk in api.chat_stream(model=model, messages=messages):
                if first:
                    status.stop()
                    label = Text(f"[{model}] ", style="bold green")
                    console.print(label, end="")
                    first = False
                console.print(chunk, end="", highlight=False, markup=False)
                full += chunk
        except APIError as exc:
            if first:
                status.stop()
            if exc.status_code == 403:
                raise _SessionExpired()
            console.print(f"\n[red]Error {exc.status_code}:[/red] {exc.detail}")
            return ""
        except KeyboardInterrupt:
            if first:
                status.stop()
            console.print("\n[yellow]Interrupted.[/yellow]")
            return full

    elapsed = _time.monotonic() - start
    console.print(f"\n[dim]({elapsed:.1f}s)[/dim]")
    return full


# ── Commands ──────────────────────────────────────────────────────────────────

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

    # ── Validate session before doing anything ────────────────────────────────
    try:
        api.whoami()
    except APIError as exc:
        if exc.status_code == 403:
            new_api = _relogin(api.server_url)
            if not new_api:
                raise typer.Exit(1)
            api = new_api
        else:
            console.print(f"[red]Cannot reach server:[/red] {exc}")
            raise typer.Exit(1)

    current_model = _pick_model(api, model)

    # ── One-shot mode ─────────────────────────────────────────────────────────
    if message:
        messages = [{"role": "user", "content": message}]
        try:
            response = _stream_response(api, current_model, messages)
        except _SessionExpired:
            console.print(
                "\n[bold yellow]⚠ Session expired[/bold yellow] – "
                "run [bold]fmind login[/bold] and try again."
            )
            raise typer.Exit(1)
        if response:
            console.print(Rule(style="dim"))
            console.print(Markdown(response))
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
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

        # ── Slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd == "/clear":
                os.system("clear" if sys.platform != "win32" else "cls")
                history.clear()
                console.print("[dim]Screen cleared. History reset.[/dim]")

            elif cmd == "/models":
                _print_models_table(api)

            elif cmd == "/switch":
                if len(parts) < 2:
                    console.print("[red]Usage:[/red] /switch <model-name>")
                else:
                    current_model = parts[1].strip()
                    history.clear()
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

        # ── Normal message ────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        try:
            response = _stream_response(api, current_model, history)
        except _SessionExpired:
            history.pop()  # remove the unsent user message
            new_api = _relogin(api.server_url)
            if new_api:
                api = new_api
                console.print(
                    "[dim]Session renewed – please re-send your last message.[/dim]"
                )
            else:
                break
            continue

        if response:
            history.append({"role": "assistant", "content": response})
