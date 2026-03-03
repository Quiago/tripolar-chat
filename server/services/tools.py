"""Tool definitions and executor for the FactoryMind AI chat agent.

Tools follow the OpenAI function-calling schema.  When the LLM returns a
tool_call, ``execute_tool`` runs the corresponding service function and
returns the result as a plain string (JSON where possible, human-readable
error on failure).

``build_connector_context`` produces a system-prompt snippet describing the
user's registered environments and connectors so the model knows what it can
query.

``run_tool_loop`` drives the multi-turn tool-execution cycle:
    LLM call → tool_call? → execute → inject result → LLM call → … → stop
"""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Optional

from sqlmodel import Session

from server.services import asset_service
from server.services.connector_manager import _CONNECTOR_FACTORIES

log = logging.getLogger(__name__)


# ── Tool definitions (OpenAI function-calling schema) ─────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_live_data",
            "description": (
                "Query live sensor readings from an industrial connector "
                "(OPC-UA, simulator, http_push). Use this when the user asks "
                "about current equipment status, temperatures, vibration, health "
                "scores, or any real-time data. Returns a list of asset readings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "connector_type": {
                        "type": "string",
                        "enum": list(_CONNECTOR_FACTORIES.keys()),
                        "description": "The connector protocol to use.",
                    },
                    "config": {
                        "type": "object",
                        "description": (
                            "Connector-specific config dict. "
                            "For opcua: {\"endpoint_url\": \"opc.tcp://host:port\"}. "
                            "For simulator: {\"assets\": [...]}."
                        ),
                    },
                    "environment_id": {
                        "type": "string",
                        "description": (
                            "Optional environment ID to scope the query "
                            "and reuse cached connections."
                        ),
                    },
                },
                "required": ["connector_type", "config"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_summary",
            "description": (
                "Get an aggregated health snapshot of ALL assets in an environment. "
                "Returns total asset count, severity breakdown, and lists of "
                "critical / degrading assets. Use this for overview questions like "
                "'how is the plant doing?' or 'any critical alarms?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "environment_id": {
                        "type": "string",
                        "description": "The environment ID to summarise.",
                    }
                },
                "required": ["environment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_asset_history",
            "description": (
                "Retrieve the recent health-event history for a specific asset. "
                "Use this when the user asks about trends, past alarms, or "
                "how an asset's health has changed over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Internal asset UUID.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max events to return (default 20).",
                        "default": 20,
                    },
                },
                "required": ["asset_id"],
            },
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────


def execute_tool(
    tool_name: str,
    args: dict,
    user_id: str,
    db: Session,
) -> str:
    """Execute a named tool and return its result as a JSON string.

    Unknown tools or execution errors return a plain-text error message so
    the model can report them gracefully rather than crashing.

    Args:
        tool_name: The function name from the tool_call.
        args: Parsed arguments dict from the tool_call.
        user_id: Authenticated user ID (for ownership checks).
        db: Open database session.

    Returns:
        JSON string of the result, or a human-readable error string.
    """
    try:
        if tool_name == "query_live_data":
            return _run_sync(_exec_query_live_data(args, user_id, db))
        elif tool_name == "get_fleet_summary":
            return _exec_get_fleet_summary(args, user_id, db)
        elif tool_name == "get_asset_history":
            return _exec_get_asset_history(args, user_id, db)
        else:
            return f"Error: unknown tool '{tool_name}'."
    except Exception as exc:
        log.error("Tool '%s' raised: %s", tool_name, exc)
        return f"Error executing tool '{tool_name}': {exc}"


# ── Individual tool implementations ──────────────────────────────────────────


async def _exec_query_live_data(args: dict, user_id: str, db: Session) -> str:
    connector_type = args.get("connector_type", "")
    config = args.get("config", {})
    environment_id = args.get("environment_id")

    if connector_type not in _CONNECTOR_FACTORIES:
        return f"Error: unknown connector_type '{connector_type}'."

    factory = _CONNECTOR_FACTORIES[connector_type]
    connector = factory(config)

    try:
        if not connector.is_connected():
            await connector.connect()
        readings = await connector.discover_assets()
    except Exception as exc:
        return f"Error: could not connect to {connector_type} data source: {exc}"
    finally:
        try:
            await connector.disconnect()
        except Exception:
            pass

    if not readings:
        return f"No assets discovered from {connector_type} connector."

    rows = []
    for r in readings:
        row: dict[str, Any] = {
            "name": r.name,
            "external_id": r.external_id,
            "asset_type": r.asset_type,
            "health_score": r.health_score,
            "severity": r.severity,
        }
        if r.failure_mode:
            row["failure_mode"] = r.failure_mode
        if r.raw_value is not None:
            row["value"] = r.raw_value
            if r.raw_unit:
                row["unit"] = r.raw_unit
        if r.message:
            row["message"] = r.message
        rows.append(row)

    return json.dumps({"connector_type": connector_type, "readings": rows}, indent=2)


def _exec_get_fleet_summary(args: dict, user_id: str, db: Session) -> str:
    env_id = args.get("environment_id", "")
    summary = asset_service.get_fleet_summary(db, env_id, user_id)
    # Convert SQLModel / dataclass to plain dict
    if hasattr(summary, "model_dump"):
        data = summary.model_dump()
    else:
        data = dict(summary)
    # Serialize datetime fields
    return json.dumps(data, default=str, indent=2)


def _exec_get_asset_history(args: dict, user_id: str, db: Session) -> str:
    asset_id = args.get("asset_id", "")
    limit = int(args.get("limit", 20))
    events = asset_service.get_asset_history(db, asset_id, user_id, limit=limit)
    rows = []
    for e in events:
        rows.append({
            "timestamp": str(e.timestamp) if hasattr(e, "timestamp") else str(e.get("timestamp")),
            "health_score": e.health_score if hasattr(e, "health_score") else e.get("health_score"),
            "severity": e.severity if hasattr(e, "severity") else e.get("severity"),
            "failure_mode": e.failure_mode if hasattr(e, "failure_mode") else e.get("failure_mode"),
            "message": e.message if hasattr(e, "message") else e.get("message"),
        })
    return json.dumps(rows, default=str, indent=2)


# ── Sync helper ───────────────────────────────────────────────────────────────


def _run_sync(coro) -> str:
    """Run a coroutine synchronously (used when execute_tool is called from sync code)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an async context (e.g. FastAPI) — schedule and wait
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(coro)
    except Exception as exc:
        return f"Error: {exc}"


# ── Context builder ────────────────────────────────────────────────────────────


def build_connector_context(db: Session, user_id: str) -> str:
    """Build a system-prompt snippet describing the user's connected infrastructure.

    Injected before every chat request so the model knows what tools it can
    call and which environments/connectors exist.

    Args:
        db: Open database session.
        user_id: Authenticated user ID.

    Returns:
        Multi-line string to append to the system prompt.
    """
    try:
        environments = asset_service.get_environments(db, user_id=user_id)
    except Exception:
        environments = []

    if not environments:
        return (
            "No industrial environments are registered yet. "
            "The user can add one via the asset registry API."
        )

    lines = ["You have access to the following industrial environments and connectors:"]
    for env in environments:
        lines.append(f'\n• Environment: "{env.name}" (id: {env.id}, type: {env.env_type})')
        try:
            connectors = asset_service.list_connectors(db, env.id, user_id)
        except Exception:
            connectors = []

        if not connectors:
            lines.append("  No connectors registered.")
        else:
            for conn in connectors:
                cfg = {}
                try:
                    cfg = json.loads(conn.config_json)
                except Exception:
                    pass
                endpoint = cfg.get("endpoint_url") or cfg.get("host") or ""
                lines.append(
                    f"  - {conn.connector_type.upper()} connector: \"{conn.name}\""
                    + (f" at {endpoint}" if endpoint else "")
                    + f" (env_id={env.id})"
                )

    lines.append(
        "\nWhen the user asks about real-time data, equipment health, temperatures, "
        "alarms, or any live readings — use the query_live_data tool with the "
        "appropriate connector_type and config. "
        "If the connector is unreachable, report it clearly. "
        "Use get_fleet_summary for overview questions about an entire environment."
    )
    return "\n".join(lines)


# ── Tool-call loop ────────────────────────────────────────────────────────────

_VLLMCall = Callable[[list[dict]], Coroutine[Any, Any, dict]]


async def run_tool_loop(
    response: dict,
    messages: list[dict],
    user_id: str,
    db: Session,
    vllm_call: Optional[_VLLMCall] = None,
    max_iterations: int = 5,
) -> dict:
    """Drive the tool-execution loop until the model stops calling tools.

    Args:
        response: The initial vLLM response (may or may not contain tool_calls).
        messages: The conversation so far (mutated in-place with tool results).
        user_id: Authenticated user ID for tool execution.
        db: Open database session.
        vllm_call: Async callable that sends messages to vLLM and returns a
            response dict.  Required when tool calls are present.
        max_iterations: Safety cap to prevent infinite loops.

    Returns:
        The final vLLM response dict with finish_reason != "tool_calls".
    """
    for _ in range(max_iterations):
        choice = response["choices"][0]
        if choice.get("finish_reason") != "tool_calls":
            return response

        tool_calls = choice["message"].get("tool_calls", [])
        if not tool_calls:
            return response

        # Add the assistant tool-call message to conversation history
        messages.append(choice["message"])

        # Execute each tool call and append the results
        for tc in tool_calls:
            fn = tc["function"]
            tool_name = fn["name"]
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            log.info("Executing tool '%s' with args: %s", tool_name, args)
            result_str = execute_tool(tool_name, args, user_id=user_id, db=db)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_str,
            })

        # Call vLLM again with the updated messages
        if vllm_call is None:
            break
        response = await vllm_call(messages)

    return response
