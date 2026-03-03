"""fmind connectors — live data gateway commands.

fmind connectors query --type opcua --url opc.tcp://localhost:62541
fmind connectors query --type simulator --config '{"assets": [...]}'
fmind connectors query --type opcua --url opc.tcp://... --env-id <id> --store
"""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from client.api import APIError, get_client

console = Console()
app = typer.Typer(help="Query live data from industrial connectors.")

# Severity → Rich colour
_SEVERITY_STYLE = {
    "ok":       "green",
    "info":     "cyan",
    "warning":  "yellow",
    "critical": "bold red",
}


def _severity_text(severity: str) -> Text:
    style = _SEVERITY_STYLE.get(severity.lower(), "white")
    return Text(severity.upper(), style=style)


def _health_style(score: float) -> str:
    if score >= 80:
        return "green"
    if score >= 50:
        return "yellow"
    return "bold red"


# ── query ─────────────────────────────────────────────────────────────────────


@app.command()
def query(
    connector_type: str = typer.Option(..., "--type", "-t", help="Connector type: opcua, simulator, http_push"),
    config_json: Optional[str] = typer.Option(None, "--config", "-c", help="JSON config dict for the connector"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="OPC-UA endpoint URL shorthand (opcua only)"),
    environment_id: Optional[str] = typer.Option(None, "--env-id", "-e", help="Environment ID for caching / storage"),
    store: bool = typer.Option(False, "--store", "-s", help="Persist readings to the asset registry (requires --env-id)"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of a table"),
) -> None:
    """Fetch live readings from a connector (config passed inline).

    Examples:\n
      fmind connectors query --type opcua --url opc.tcp://localhost:62541\n
      fmind connectors query --type simulator --config '{"assets": [...]}'\n
      fmind connectors query --type opcua --url opc.tcp://... --env-id <id> --store
    """
    # ── Validate --url shorthand ──────────────────────────────────────────────
    if url is not None and connector_type != "opcua":
        console.print(
            "[red]Error:[/red] --url is only valid with --type opcua. "
            "Use --config for other connector types."
        )
        raise typer.Exit(1)

    # ── Build config dict ─────────────────────────────────────────────────────
    if url is not None:
        config: dict = {"endpoint_url": url}
    elif config_json is not None:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Invalid JSON config:[/red] {exc}")
            raise typer.Exit(1)
    else:
        console.print(
            "[red]Error:[/red] Provide either --config <json> or --url <opc.tcp://...>"
        )
        raise typer.Exit(1)

    # ── Validate store requirements ───────────────────────────────────────────
    if store and not environment_id:
        console.print("[red]Error:[/red] --store requires --env-id.")
        raise typer.Exit(1)

    # ── Call the API ──────────────────────────────────────────────────────────
    try:
        api = get_client()
    except APIError as exc:
        console.print(f"[red]Not logged in:[/red] {exc.detail}")
        raise typer.Exit(1)

    with console.status(f"[dim]Querying {connector_type} connector…[/dim]"):
        try:
            result = api.query_connector(
                connector_type=connector_type,
                config=config,
                environment_id=environment_id,
                store_readings=store,
            )
        except APIError as exc:
            console.print(f"[red]Error [{exc.status_code}]:[/red] {exc.detail}")
            raise typer.Exit(1)

    # ── JSON output ───────────────────────────────────────────────────────────
    if as_json:
        console.print_json(json.dumps(result))
        return

    # ── Table output ──────────────────────────────────────────────────────────
    readings = result.get("readings", [])
    asset_count = result.get("asset_count", len(readings))
    stored = result.get("stored", 0)
    queried_at = result.get("queried_at", "")

    title = (
        f"[bold]{connector_type.upper()}[/bold] — "
        f"{asset_count} asset{'s' if asset_count != 1 else ''}"
        f"  [dim]{queried_at}[/dim]"
    )
    if stored:
        title += f"  [green]({stored} stored)[/green]"

    table = Table(title=title, show_lines=False, expand=False)
    table.add_column("Asset", style="bold cyan", no_wrap=True)
    table.add_column("Type", style="dim")
    table.add_column("External ID", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Severity", justify="center")
    table.add_column("Message")

    if not readings:
        console.print(f"[yellow]No assets discovered from {connector_type}.[/yellow]")
        return

    for r in readings:
        score = r.get("health_score", 0.0)
        severity = r.get("severity", "")
        message = r.get("message") or r.get("failure_mode") or ""
        table.add_row(
            r.get("name", ""),
            r.get("asset_type", ""),
            r.get("external_id", ""),
            Text(f"{score:.1f}", style=_health_style(score)),
            _severity_text(severity),
            message,
        )

    console.print()
    console.print(table)
    console.print()
