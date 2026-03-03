"""TDD tests for `fmind connectors query` CLI command.

Coverage:
  Happy paths   — simulator query, opcua shorthand, store-to-db
  Error cases   — not logged in, 400 bad type, 503 unreachable, bad JSON
  Output format — table rendered, severity colours, stored count shown
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _reading(
    external_id="pump-01",
    name="Pump A",
    asset_type="pump",
    health_score=85.0,
    severity="ok",
    source="simulator",
    failure_mode=None,
    raw_value=None,
    raw_unit=None,
    message=None,
):
    return {
        "external_id": external_id,
        "name": name,
        "asset_type": asset_type,
        "health_score": health_score,
        "severity": severity,
        "source": source,
        "failure_mode": failure_mode,
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "message": message,
    }


def _response(readings=None, stored=0, connector_type="simulator"):
    readings = readings or [_reading()]
    return {
        "connector_type": connector_type,
        "asset_count": len(readings),
        "readings": readings,
        "stored": stored,
        "queried_at": "2026-03-03T10:00:00",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_client(response: dict):
    """Return a mock APIClient whose query_connector returns *response*."""
    client = MagicMock()
    client.query_connector.return_value = response
    return client


# ── Import the connectors app ─────────────────────────────────────────────────


from client.commands.connectors import app  # noqa: E402


# ── Not logged in ─────────────────────────────────────────────────────────────


class TestNotLoggedIn:
    def test_query_exits_with_error_when_not_logged_in(self):
        from client.api import APIError

        with patch("client.commands.connectors.get_client", side_effect=APIError(0, "Not logged in")):
            result = runner.invoke(app, ["--type", "simulator", "--config", "{}"])
        assert result.exit_code != 0
        assert "logged in" in result.output.lower() or "not logged" in result.output.lower()


# ── Simulator happy path ──────────────────────────────────────────────────────


class TestSimulatorQuery:
    def _run(self, extra_args=None):
        config = json.dumps({
            "assets": [
                {"external_id": "p-01", "name": "Pump A", "asset_type": "pump", "base_score": 85.0},
            ]
        })
        args = ["--type", "simulator", "--config", config]
        if extra_args:
            args += extra_args

        mock = _mock_client(_response([_reading("p-01", "Pump A")]))
        with patch("client.commands.connectors.get_client", return_value=mock):
            return runner.invoke(app, args)

    def test_exits_zero(self):
        assert self._run().exit_code == 0

    def test_shows_asset_name(self):
        assert "Pump A" in self._run().output

    def test_shows_health_score(self):
        assert "85" in self._run().output

    def test_shows_severity(self):
        assert "ok" in self._run().output.lower()

    def test_shows_connector_type_header(self):
        assert "simulator" in self._run().output.lower()

    def test_json_flag_outputs_valid_json(self):
        result = self._run(["--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["connector_type"] == "simulator"
        assert len(data["readings"]) == 1


# ── OPC-UA shorthand (--url) ──────────────────────────────────────────────────


class TestOpcUaShorthand:
    def test_url_flag_builds_correct_config(self):
        """--url opc.tcp://... should construct the opcua config automatically."""
        endpoint = "opc.tcp://localhost:62541"
        captured = {}

        def fake_query(connector_type, config, **kwargs):
            captured["connector_type"] = connector_type
            captured["config"] = config
            return _response(connector_type="opcua")

        mock = MagicMock()
        mock.query_connector.side_effect = fake_query

        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(app, ["--type", "opcua", "--url", endpoint])

        assert result.exit_code == 0
        assert captured["connector_type"] == "opcua"
        assert captured["config"]["endpoint_url"] == endpoint

    def test_url_and_type_opcua_required(self):
        """--url without --type opcua should show a helpful error."""
        with patch("client.commands.connectors.get_client", return_value=MagicMock()):
            result = runner.invoke(app, ["--type", "simulator", "--url", "opc.tcp://x"])
        assert result.exit_code != 0
        assert "--url" in result.output or "opcua" in result.output.lower()


# ── Multiple assets rendering ─────────────────────────────────────────────────


class TestMultipleAssets:
    def test_all_assets_shown(self):
        readings = [
            _reading("p-01", "Pump A", health_score=90.0, severity="ok"),
            _reading("m-01", "Motor B", health_score=45.0, severity="warning"),
            _reading("c-01", "Compressor C", health_score=20.0, severity="critical"),
        ]
        mock = _mock_client(_response(readings))
        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(
                app,
                ["--type", "simulator", "--config", '{"assets": []}'],
            )
        assert result.exit_code == 0
        assert "Pump A" in result.output
        assert "Motor B" in result.output
        assert "Compressor C" in result.output

    def test_asset_count_shown(self):
        readings = [_reading(f"a-{i}", f"Asset {i}") for i in range(4)]
        mock = _mock_client(_response(readings))
        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(
                app,
                ["--type", "simulator", "--config", '{"assets": []}'],
            )
        assert "4" in result.output


# ── Store to DB ───────────────────────────────────────────────────────────────


class TestStoreToDb:
    def test_store_flag_passes_env_id_and_store_flag(self):
        env_id = "env-abc-123"
        captured = {}

        def fake_query(connector_type, config, **kwargs):
            captured.update(kwargs)
            return _response(stored=1)

        mock = MagicMock()
        mock.query_connector.side_effect = fake_query

        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(
                app,
                [
                    "--type", "simulator",
                    "--config", '{"assets": []}',
                    "--env-id", env_id,
                    "--store",
                ],
            )

        assert result.exit_code == 0
        assert captured.get("environment_id") == env_id
        assert captured.get("store_readings") is True

    def test_stored_count_shown(self):
        mock = _mock_client(_response(stored=3))
        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(
                app,
                [
                    "--type", "simulator",
                    "--config", '{"assets": []}',
                    "--env-id", "env-x",
                    "--store",
                ],
            )
        assert "3" in result.output


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_400_unknown_type_shows_message(self):
        from client.api import APIError

        mock = MagicMock()
        mock.query_connector.side_effect = APIError(400, "Unknown connector_type 'mqtt_v99'")
        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(app, ["--type", "mqtt_v99", "--config", "{}"])
        assert result.exit_code != 0
        assert "400" in result.output or "Unknown" in result.output

    def test_503_unreachable_shows_message(self):
        from client.api import APIError

        mock = MagicMock()
        mock.query_connector.side_effect = APIError(503, "Could not connect to opcua data source")
        with patch("client.commands.connectors.get_client", return_value=mock):
            result = runner.invoke(app, ["--type", "opcua", "--config", "{}"])
        assert result.exit_code != 0
        assert "503" in result.output or "connect" in result.output.lower()

    def test_invalid_json_config_exits_with_error(self):
        with patch("client.commands.connectors.get_client", return_value=MagicMock()):
            result = runner.invoke(app, ["--type", "simulator", "--config", "{bad json}"])
        assert result.exit_code != 0
        assert "json" in result.output.lower() or "invalid" in result.output.lower()
