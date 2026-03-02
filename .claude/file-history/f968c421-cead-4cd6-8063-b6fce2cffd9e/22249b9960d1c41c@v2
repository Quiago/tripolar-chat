"""fmind auth commands: login, logout, register, whoami, rotate-key."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from client import api as fmapi
from client import config as cfg
from client.api import APIError

console = Console()
app = typer.Typer(help="Authentication commands.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_server(server: Optional[str]) -> str:
    url = server or cfg.get_server_url()
    if not url:
        console.print(
            "[red]No server URL configured.[/red] "
            "Pass [bold]--server[/bold] or run [bold]fmind login[/bold] first."
        )
        raise typer.Exit(1)
    return url


def _require_credentials() -> tuple[str, str]:
    url = cfg.get_server_url()
    key = cfg.get_api_key()
    if not url or not key:
        console.print(
            "[red]Not logged in.[/red] Run [bold]fmind login[/bold] first."
        )
        raise typer.Exit(1)
    return url, key


def _print_user(user: dict) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold]Username[/bold]", user["username"])
    table.add_row("[bold]Email[/bold]", user["email"])
    table.add_row("[bold]API Key[/bold]", user["api_key"])
    table.add_row("[bold]Created[/bold]", user["created_at"])
    table.add_row("[bold]Active[/bold]", str(user["is_active"]))
    console.print(table)


# ── commands ──────────────────────────────────────────────────────────────────

@app.command()
def login(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Server URL, e.g. https://factorymind.example.com"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Use an existing API key directly."),
):
    """Log in to a FactoryMind server.

    If --api-key is provided the key is stored without contacting the server.
    Otherwise you will be prompted for your username and password.
    """
    url = _require_server(server)

    if api_key:
        # Verify the key is valid before saving
        try:
            user = fmapi.whoami(url, api_key)
        except APIError as exc:
            console.print(f"[red]Login failed:[/red] {exc}")
            raise typer.Exit(1)

        cfg.save_credentials(url, api_key)
        console.print(f"[green]Logged in as[/green] [bold]{user['username']}[/bold] ({url})")
        return

    # Interactive username / password flow
    username = typer.prompt("Username")
    password = typer.prompt("Password", hide_input=True)

    try:
        user = fmapi.login(url, username, password)
    except APIError as exc:
        console.print(f"[red]Login failed:[/red] {exc}")
        raise typer.Exit(1)

    cfg.save_credentials(url, user["api_key"])
    console.print(f"[green]Logged in as[/green] [bold]{user['username']}[/bold] ({url})")
    console.print(f"[dim]API key saved to ~/.fmind/config.yaml[/dim]")


@app.command()
def logout():
    """Clear stored credentials."""
    cfg.clear_credentials()
    console.print("[yellow]Logged out.[/yellow] Credentials removed from ~/.fmind/config.yaml.")


@app.command()
def register(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Server URL."),
    username: Optional[str] = typer.Option(None, "--username", "-u"),
    email: Optional[str] = typer.Option(None, "--email", "-e"),
):
    """Register a new account on a FactoryMind server."""
    url = _require_server(server)

    if not username:
        username = typer.prompt("Username")
    if not email:
        email = typer.prompt("Email")
    password = typer.prompt("Password", hide_input=True)
    typer.prompt("Confirm password", hide_input=True, confirmation_prompt=True)

    try:
        user = fmapi.register(url, username, email, password)
    except APIError as exc:
        console.print(f"[red]Registration failed:[/red] {exc}")
        raise typer.Exit(1)

    cfg.save_credentials(url, user["api_key"])
    console.print(f"[green]Account created![/green] Logged in as [bold]{user['username']}[/bold].")
    console.print(f"[dim]API key saved to ~/.fmind/config.yaml[/dim]")


@app.command()
def whoami():
    """Show the currently authenticated user."""
    url, key = _require_credentials()
    try:
        user = fmapi.whoami(url, key)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    _print_user(user)


@app.command(name="rotate-key")
def rotate_key():
    """Generate a new API key and update local config."""
    url, key = _require_credentials()
    try:
        user = fmapi.rotate_key(url, key)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    cfg.save_credentials(url, user["api_key"])
    console.print("[green]New API key generated and saved.[/green]")
    console.print(f"[dim]{user['api_key']}[/dim]")
