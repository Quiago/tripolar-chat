"""TDD tests for server/services/tools.py — tool definitions and executor.

Coverage:
  Schema        — TOOLS list is valid OpenAI function-calling format
  Executor      — each tool calls the right service; errors are reported gracefully
  Context       — build_connector_context returns a useful system prompt snippet
  Chat loop     — tool_call response is detected, executed, and looped correctly
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_db(environments=None, connectors=None):
    """Minimal mock DB session for context-building tests."""
    db = MagicMock()
    return db


def _tool_call_response(tool_name: str, args: dict, call_id: str = "call_001"):
    """Fake a vLLM non-streaming response with a single tool_call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }]
    }


def _final_response(content: str = "Pump A is healthy."):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }]
    }


# ── Schema validation ──────────────────────────────────────────────────────────


class TestToolSchemas:
    def test_tools_list_is_not_empty(self):
        from server.services.tools import TOOLS
        assert len(TOOLS) > 0

    def test_each_tool_has_type_function(self):
        from server.services.tools import TOOLS
        for tool in TOOLS:
            assert tool["type"] == "function"

    def test_each_tool_has_name_and_description(self):
        from server.services.tools import TOOLS
        for tool in TOOLS:
            fn = tool["function"]
            assert "name" in fn and fn["name"]
            assert "description" in fn and fn["description"]

    def test_each_tool_parameters_has_required_type_object(self):
        from server.services.tools import TOOLS
        for tool in TOOLS:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_query_live_data_tool_exists(self):
        from server.services.tools import TOOLS
        names = [t["function"]["name"] for t in TOOLS]
        assert "query_live_data" in names

    def test_get_fleet_summary_tool_exists(self):
        from server.services.tools import TOOLS
        names = [t["function"]["name"] for t in TOOLS]
        assert "get_fleet_summary" in names


# ── Tool executor ──────────────────────────────────────────────────────────────


class TestToolExecutor:
    def test_unknown_tool_returns_error_string(self):
        from server.services.tools import execute_tool
        db = MagicMock()
        result = execute_tool("nonexistent_tool", {}, user_id="u1", db=db)
        assert "unknown" in result.lower() or "error" in result.lower()

    def test_query_live_data_calls_connector_factory(self):
        """query_live_data must build a connector and call discover_assets."""
        from server.services.tools import execute_tool

        mock_connector = MagicMock()
        mock_connector.is_connected.return_value = False
        mock_connector.connect = AsyncMock()
        mock_connector.discover_assets = AsyncMock(return_value=[])

        with patch(
            "server.services.tools._CONNECTOR_FACTORIES",
            {"simulator": lambda cfg: mock_connector},
        ):
            result = execute_tool(
                "query_live_data",
                {"connector_type": "simulator", "config": {"assets": []}},
                user_id="u1",
                db=MagicMock(),
            )

        mock_connector.connect.assert_awaited_once()
        mock_connector.discover_assets.assert_awaited_once()
        assert isinstance(result, str)

    def test_query_live_data_unknown_connector_returns_error(self):
        from server.services.tools import execute_tool
        result = execute_tool(
            "query_live_data",
            {"connector_type": "mqtt_unknown", "config": {}},
            user_id="u1",
            db=MagicMock(),
        )
        assert "unknown" in result.lower() or "error" in result.lower()

    def test_get_fleet_summary_calls_asset_service(self):
        from server.services.tools import execute_tool

        fake_summary = {
            "total_assets": 3,
            "by_severity": {"ok": 2, "warning": 1},
            "critical_assets": [],
            "degrading_assets": [],
            "last_updated": None,
        }
        with patch(
            "server.services.tools.asset_service.get_fleet_summary",
            return_value=fake_summary,
        ):
            result = execute_tool(
                "get_fleet_summary",
                {"environment_id": "env-123"},
                user_id="u1",
                db=MagicMock(),
            )

        data = json.loads(result)
        assert data["total_assets"] == 3

    def test_get_fleet_summary_access_denied_returns_error(self):
        from server.services.tools import execute_tool
        from server.core.exceptions import AccessDeniedError

        with patch(
            "server.services.tools.asset_service.get_fleet_summary",
            side_effect=AccessDeniedError("You do not own this environment."),
        ):
            result = execute_tool(
                "get_fleet_summary",
                {"environment_id": "env-other"},
                user_id="u1",
                db=MagicMock(),
            )

        assert "not own" in result.lower() or "denied" in result.lower() or "error" in result.lower()

    def test_query_live_data_returns_formatted_readings(self):
        from server.services.tools import execute_tool
        from server.services.connectors.base import HealthReading

        readings = [
            HealthReading(
                external_id="p-01",
                name="Pump A",
                asset_type="pump",
                source="simulator",
                health_score=85.0,
                severity="ok",
                message="Normal operation",
            )
        ]
        mock_connector = MagicMock()
        mock_connector.is_connected.return_value = False
        mock_connector.connect = AsyncMock()
        mock_connector.discover_assets = AsyncMock(return_value=readings)

        with patch(
            "server.services.tools._CONNECTOR_FACTORIES",
            {"simulator": lambda cfg: mock_connector},
        ):
            result = execute_tool(
                "query_live_data",
                {"connector_type": "simulator", "config": {"assets": []}},
                user_id="u1",
                db=MagicMock(),
            )

        assert "Pump A" in result
        assert "85" in result


# ── Context builder ────────────────────────────────────────────────────────────


class TestConnectorContext:
    def test_returns_string(self):
        from server.services.tools import build_connector_context
        with patch(
            "server.services.tools.asset_service.get_environments",
            return_value=[],
        ):
            result = build_connector_context(MagicMock(), user_id="u1")
        assert isinstance(result, str)

    def test_includes_environment_name(self):
        from server.services.tools import build_connector_context

        env = MagicMock()
        env.id = "env-111"
        env.name = "Plant Floor"
        env.env_type = "factory"

        with patch("server.services.tools.asset_service.get_environments", return_value=[env]):
            with patch("server.services.tools.asset_service.list_connectors", return_value=[]):
                result = build_connector_context(MagicMock(), user_id="u1")

        assert "Plant Floor" in result

    def test_includes_connector_type(self):
        from server.services.tools import build_connector_context

        env = MagicMock()
        env.id = "env-111"
        env.name = "Env A"
        env.env_type = "factory"

        conn = MagicMock()
        conn.connector_type = "opcua"
        conn.name = "Ignition OPC-UA"
        conn.config_json = '{"endpoint_url": "opc.tcp://localhost:62541"}'

        with patch("server.services.tools.asset_service.get_environments", return_value=[env]):
            with patch("server.services.tools.asset_service.list_connectors", return_value=[conn]):
                result = build_connector_context(MagicMock(), user_id="u1")

        assert "opcua" in result.lower()
        assert "opc.tcp://localhost:62541" in result

    def test_no_environments_returns_brief_message(self):
        from server.services.tools import build_connector_context
        with patch("server.services.tools.asset_service.get_environments", return_value=[]):
            result = build_connector_context(MagicMock(), user_id="u1")
        assert result  # non-empty, tells model there are no connectors


# ── Tool-call loop (chat router level) ────────────────────────────────────────


class TestToolCallLoop:
    """Tests the run_tool_loop helper that drives multi-turn tool execution."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_response_unchanged(self):
        from server.services.tools import run_tool_loop

        resp = _final_response("All good.")
        result = await run_tool_loop(resp, messages=[], user_id="u1", db=MagicMock())
        assert result["choices"][0]["message"]["content"] == "All good."

    @pytest.mark.asyncio
    async def test_tool_call_is_executed_and_loop_continues(self):
        from server.services.tools import run_tool_loop

        first = _tool_call_response("query_live_data", {
            "connector_type": "simulator",
            "config": {"assets": []},
        })
        final = _final_response("Pump A is healthy.")

        call_count = 0

        async def fake_vllm(messages):
            nonlocal call_count
            call_count += 1
            return final

        with patch("server.services.tools.execute_tool", return_value='{"readings": []}'):
            result = await run_tool_loop(
                first,
                messages=[],
                user_id="u1",
                db=MagicMock(),
                vllm_call=fake_vllm,
            )

        assert call_count == 1  # one follow-up call after tool execution
        assert result["choices"][0]["message"]["content"] == "Pump A is healthy."

    @pytest.mark.asyncio
    async def test_max_iterations_prevents_infinite_loop(self):
        """If the model keeps calling tools, stop after max_tool_iterations."""
        from server.services.tools import run_tool_loop

        always_tool = _tool_call_response("query_live_data", {
            "connector_type": "simulator", "config": {}
        })

        async def always_calls_tool(messages):
            return always_tool

        with patch("server.services.tools.execute_tool", return_value="{}"):
            result = await run_tool_loop(
                always_tool,
                messages=[],
                user_id="u1",
                db=MagicMock(),
                vllm_call=always_calls_tool,
                max_iterations=3,
            )

        # Should return something (the last tool-call response) rather than looping forever
        assert result is not None
