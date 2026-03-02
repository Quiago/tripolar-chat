"""fmind – FactoryMind AI CLI entrypoint."""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from client import config as cfg
from client.api import APIError
from client.commands import auth, chat, history, models

console = Console()

app = typer.Typer(
    name="fmind",
    help="FactoryMind AI CLI – talk to your on-premise LLMs.",
    no_args_is_help=False,
    invoke_without_command=True,
)

# ── Mount auth commands at top level (fmind login / logout / register …) ──────
for _cmd in auth.app.registered_commands:
    app.registered_commands.append(_cmd)

# ── Mount remaining command groups ─────────────────────────────────────────────
app.add_typer(chat.app, name="chat", invoke_without_command=True)
app.add_typer(models.app, name="models", invoke_without_command=True)
app.add_typer(history.app, name="history", invoke_without_command=True)

# Hidden alias groups for explicit namespace access
app.add_typer(auth.app, name="auth", hidden=True)


# ── First-run setup wizard ─────────────────────────────────────────────────────

def _run_setup_wizard() -> None:
    console.print(
        Panel(
            "[bold]Welcome to FactoryMind AI! 🏭[/bold]\n\n"
            "Let's get you set up…",
            expand=False,
        )
    )

    server_url = typer.prompt("Server URL").strip().rstrip("/")

    # Verify the server is reachable
    import httpx

    try:
        with httpx.Client(base_url=server_url, timeout=10.0) as c:
            r = c.get("/health")
        if r.is_error:
            console.print(
                f"[yellow]Warning:[/yellow] Server responded with {r.status_code}. "
                "Proceeding anyway."
            )
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Could not reach {server_url}: {exc}\n"
            "Proceeding anyway – you can change the URL later."
        )

    has_account = typer.confirm("Do you already have an account?", default=False)

    if has_account:
        from client.api import login as _login

        username = typer.prompt("Username")
        password = typer.prompt("Password", hide_input=True)
        try:
            user = _login(server_url, username, password)
        except APIError as exc:
            console.print(f"[red]Login failed:[/red] {exc}")
            raise typer.Exit(1)

        cfg.save_credentials(server_url, user["api_key"])
        console.print(f"\n[green]✓ Logged in as[/green] [bold]{user['username']}[/bold]")
    else:
        from client.api import register as _register

        username = typer.prompt("Username")
        email = typer.prompt("Email")
        password = typer.prompt("Password", hide_input=True)
        typer.prompt("Confirm password", hide_input=True, confirmation_prompt=True)
        try:
            user = _register(server_url, username, email, password)
        except APIError as exc:
            console.print(f"[red]Registration failed:[/red] {exc}")
            raise typer.Exit(1)

        cfg.save_credentials(server_url, user["api_key"])
        console.print(f"\n[green]✓ Account created![/green] Logged in as [bold]{user['username']}[/bold]")

    console.print(f"[green]✓ Config saved to[/green] ~/.fmind/config.yaml")
    console.print()
    console.print(Rule(style="dim"))
    console.print("\nTry [bold]fmind chat[/bold] to start chatting!\n")


# ── No-args callback: wizard → TUI ────────────────────────────────────────────

@app.callback()
def _callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if not cfg.get_server_url() or not cfg.get_api_key():
        _run_setup_wizard()
        return

    # Launch TUI if Textual is available
    try:
        from client.tui import launch_tui
        launch_tui()
    except ImportError:
        console.print(
            "[yellow]Tip:[/yellow] Install [bold]textual[/bold] for the interactive TUI:\n"
            "  pip install 'textual>=0.50'\n\n"
            "Available commands:"
        )
        console.print("  [cyan]fmind chat[/cyan]       – interactive chat")
        console.print("  [cyan]fmind models[/cyan]     – list models")
        console.print("  [cyan]fmind history[/cyan]    – view chat history")
        console.print("  [cyan]fmind --help[/cyan]     – full help")


# ── tui command ────────────────────────────────────────────────────────────────

@app.command()
def tui() -> None:
    """Launch the interactive Textual TUI."""
    try:
        from client.tui import launch_tui
        launch_tui()
    except ImportError:
        console.print(
            "[red]Textual is not installed.[/red] Install it with:\n"
            "  pip install 'textual>=0.50'"
        )
        raise typer.Exit(1)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
