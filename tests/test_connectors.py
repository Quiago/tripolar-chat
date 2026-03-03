"""TDD tests for the connector stack.

Covers (in order of complexity):
  _map_asset_type         — pure function, no mocking
  _compute_health         — pure function, no mocking
  HttpPushConnector       — passive connector, no I/O
  SimulatorConnector      — in-process random walk, no I/O
  OpcUaConnector          — asyncua fully mocked
  ConnectorManager        — connectors mocked
  Discovery HTTP endpoint — integration via TestClient

OPC-UA tests never touch a real server; asyncua.Client is patched
at the module level so the dependency doesn't need to be importable.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_node_class(value: int):
    """Return a mock object whose .value matches an OPC-UA NodeClass int."""
    nc = MagicMock()
    nc.value = value
    return nc


def _make_opcua_node(name: str, node_id_str: str, node_class_value: int = 1,
                     children=None):
    """Build a realistic asyncua node mock."""
    browse_name = MagicMock()
    browse_name.Name = name

    node = MagicMock()
    node.nodeid = MagicMock()
    node.nodeid.__str__ = lambda self: node_id_str

    node.read_browse_name = AsyncMock(return_value=browse_name)
    node.read_node_class = AsyncMock(return_value=_make_node_class(node_class_value))
    node.get_children = AsyncMock(return_value=children or [])
    node.read_value = AsyncMock(return_value=42.0)
    return node


# ── _map_asset_type ───────────────────────────────────────────────────────────


class TestMapAssetType:
    def test_pump(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Centrifugal Pump A") == "pump"

    def test_motor(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Motor Drive 3") == "motor"

    def test_compressor(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Air Compressor") == "compressor"

    def test_fan(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Cooling Fan Unit") == "fan"

    def test_chiller(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Water Chiller") == "chiller"

    def test_conveyor(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Belt Conveyor 2") == "conveyor"

    def test_unknown_defaults_to_equipment(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("Mystery Device") == "equipment"

    def test_case_insensitive(self):
        from server.services.connectors.opcua_connector import _map_asset_type
        assert _map_asset_type("PUMP-101") == "pump"


# ── _compute_health ───────────────────────────────────────────────────────────


class TestComputeHealth:
    def test_no_variables_returns_default_75(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({}, {})
        assert score == 75.0
        assert sev == "info"
        assert fm is None

    def test_explicit_health_variable_used_directly(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({"HealthScore": 92.0}, {})
        assert score == 92.0
        assert sev == "ok"

    def test_temperature_below_healthy_max_gives_ok(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({"Temperature": 70.0}, {})
        assert score == 90.0
        assert sev == "ok"

    def test_temperature_above_healthy_max_gives_warning(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({"Temperature": 85.0}, {})
        assert score == 60.0
        assert sev == "info"
        assert fm is not None

    def test_temperature_above_critical_max_gives_critical(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({"Temperature": 100.0}, {})
        assert score == 20.0
        assert sev == "critical"
        assert "overheat" in fm

    def test_vibration_high_gives_critical(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({"Vibration": 12.0}, {})
        assert score == 20.0
        assert sev == "critical"

    def test_vibration_medium_gives_info(self):
        from server.services.connectors.opcua_connector import _compute_health
        score, sev, fm = _compute_health({"Vibration": 7.0}, {})
        assert score == 60.0

    def test_multiple_vars_takes_minimum(self):
        from server.services.connectors.opcua_connector import _compute_health
        # Temp OK (90) + Vibration critical (20) → min is 20
        score, sev, fm = _compute_health({"Temperature": 70.0, "Vibration": 12.0}, {})
        assert score == 20.0

    def test_custom_healthy_max_from_mappings(self):
        from server.services.connectors.opcua_connector import _compute_health
        mappings = {"Temperature": {"healthy_max": 60, "critical_max": 75}}
        # 70°C is above healthy_max=60 but below critical_max=75 → warning score
        score, sev, fm = _compute_health({"Temperature": 70.0}, mappings)
        assert score == 60.0


# ── HttpPushConnector ─────────────────────────────────────────────────────────


class TestHttpPushConnector:
    @pytest.mark.asyncio
    async def test_connect_is_noop(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        c = HttpPushConnector()
        await c.connect()  # must not raise

    @pytest.mark.asyncio
    async def test_disconnect_is_noop(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        c = HttpPushConnector()
        await c.disconnect()  # must not raise

    @pytest.mark.asyncio
    async def test_discover_returns_empty(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        c = HttpPushConnector()
        assert await c.discover_assets() == []

    @pytest.mark.asyncio
    async def test_poll_returns_empty(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        c = HttpPushConnector()
        assert await c.poll() == []

    def test_is_connected_always_true(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        c = HttpPushConnector()
        assert c.is_connected() is True

    def test_parse_tractian_payload_maps_fields(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        payload = {
            "asset_id": "TRACT-001",
            "asset_name": "Motor Alpha",
            "health_score": 72.5,
            "asset_type": "motor",
            "vibration": 3.2,
            "message": "Normal operation",
        }
        reading = HttpPushConnector.parse_tractian_payload(payload)
        assert reading.external_id == "TRACT-001"
        assert reading.name == "Motor Alpha"
        assert reading.health_score == 72.5
        assert reading.raw_value == 3.2
        assert reading.raw_unit == "mm/s"
        assert reading.source == "tractian_api"

    def test_parse_tractian_severity_ok(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        reading = HttpPushConnector.parse_tractian_payload(
            {"asset_id": "x", "health_score": 90.0}
        )
        assert reading.severity == "ok"

    def test_parse_tractian_severity_critical(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        reading = HttpPushConnector.parse_tractian_payload(
            {"asset_id": "x", "health_score": 25.0}
        )
        assert reading.severity == "critical"

    def test_parse_generic_payload_uses_provided_severity(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        payload = {
            "external_id": "P-101",
            "health_score": 55.0,
            "severity": "warning",
            "source": "modbus",
        }
        reading = HttpPushConnector.parse_generic_payload(payload)
        assert reading.external_id == "P-101"
        assert reading.severity == "warning"
        assert reading.source == "modbus"

    def test_parse_generic_payload_infers_severity_when_missing(self):
        from server.services.connectors.http_push_connector import HttpPushConnector
        reading = HttpPushConnector.parse_generic_payload(
            {"external_id": "P-102", "health_score": 25.0}
        )
        assert reading.severity == "critical"


# ── SimulatorConnector ────────────────────────────────────────────────────────


def _make_simulator(n_assets: int = 3):
    from server.services.connectors.simulator_connector import (
        AssetDefinition, SimulatorConfig, SimulatorConnector,
    )
    assets = [
        AssetDefinition(external_id=f"sim-{i}", name=f"Asset {i}",
                        asset_type="pump", base_score=85.0)
        for i in range(n_assets)
    ]
    return SimulatorConnector(SimulatorConfig(assets=assets))


class TestSimulatorConnector:
    def test_is_connected_false_before_connect(self):
        sim = _make_simulator()
        assert sim.is_connected() is False

    @pytest.mark.asyncio
    async def test_is_connected_true_after_connect(self):
        sim = _make_simulator()
        await sim.connect()
        assert sim.is_connected() is True

    @pytest.mark.asyncio
    async def test_disconnect_sets_connected_false(self):
        sim = _make_simulator()
        await sim.connect()
        await sim.disconnect()
        assert sim.is_connected() is False

    @pytest.mark.asyncio
    async def test_discover_returns_all_assets(self):
        sim = _make_simulator(n_assets=4)
        await sim.connect()
        readings = await sim.discover_assets()
        assert len(readings) == 4

    @pytest.mark.asyncio
    async def test_discover_reading_has_required_fields(self):
        sim = _make_simulator(n_assets=1)
        await sim.connect()
        reading = (await sim.discover_assets())[0]
        assert reading.external_id == "sim-0"
        assert reading.name == "Asset 0"
        assert reading.asset_type == "pump"
        assert reading.source == "simulator"
        assert 0.0 <= reading.health_score <= 100.0
        assert reading.severity in ("ok", "info", "warning", "critical")

    @pytest.mark.asyncio
    async def test_poll_returns_readings_for_all_assets(self):
        sim = _make_simulator(n_assets=3)
        await sim.connect()
        readings = await sim.poll()
        assert len(readings) == 3

    @pytest.mark.asyncio
    async def test_poll_scores_stay_in_range(self):
        sim = _make_simulator(n_assets=2)
        await sim.connect()
        for _ in range(50):
            readings = await sim.poll()
            for r in readings:
                assert 0.0 <= r.health_score <= 100.0

    @pytest.mark.asyncio
    async def test_recovery_after_5_consecutive_critical_polls(self):
        """An asset stuck below 40 for 5 polls must recover to ~65."""
        from server.services.connectors.simulator_connector import (
            AssetDefinition, SimulatorConfig, SimulatorConnector,
        )
        sim = SimulatorConnector(SimulatorConfig(assets=[
            AssetDefinition("sim-0", "Asset 0", "pump", base_score=85.0),
        ]))
        await sim.connect()
        # Force the score below 40
        sim._scores["sim-0"] = 20.0
        sim._consecutive_low["sim-0"] = 4

        # 5th consecutive low poll triggers recovery
        readings = await sim.poll()
        assert readings[0].health_score == 65.0

    @pytest.mark.asyncio
    async def test_failure_mode_set_for_degraded_asset(self):
        from server.services.connectors.simulator_connector import (
            AssetDefinition, SimulatorConfig, SimulatorConnector,
        )
        sim = SimulatorConnector(SimulatorConfig(assets=[
            AssetDefinition("s0", "Pump", "pump", base_score=85.0),
        ]))
        await sim.connect()
        sim._scores["s0"] = 35.0  # critical
        readings = await sim.poll()
        assert readings[0].failure_mode is not None


# ── OpcUaConnector ────────────────────────────────────────────────────────────


@pytest.fixture()
def opcua_config():
    from server.services.connectors.opcua_connector import OpcUaConfig
    return OpcUaConfig(endpoint_url="opc.tcp://localhost:4840")


@pytest.fixture()
def mock_asyncua_client():
    """Patch asyncua.Client so no real network calls happen."""
    with patch("server.services.connectors.opcua_connector.asyncua") as mock_asyncua:
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_asyncua.Client.return_value = mock_client
        yield mock_asyncua, mock_client


class TestOpcUaConnector:
    @pytest.mark.asyncio
    async def test_connect_creates_client_with_endpoint(
        self, opcua_config, mock_asyncua_client
    ):
        mock_asyncua, mock_client = mock_asyncua_client
        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        await conn.connect()
        # Client is now constructed with url only; session_timeout is set as a property
        mock_asyncua.Client.assert_called_once_with(url="opc.tcp://localhost:4840")
        assert mock_client.session_timeout == 120_000
        mock_client.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_sets_credentials_when_provided(self, mock_asyncua_client):
        mock_asyncua, mock_client = mock_asyncua_client
        from server.services.connectors.opcua_connector import OpcUaConfig, OpcUaConnector
        cfg = OpcUaConfig(
            endpoint_url="opc.tcp://localhost:4840",
            username="admin",
            password="secret",
        )
        conn = OpcUaConnector(cfg)
        await conn.connect()
        mock_client.set_user.assert_called_once_with("admin")
        mock_client.set_password.assert_called_once_with("secret")

    @pytest.mark.asyncio
    async def test_discover_assets_browses_objects_node(
        self, opcua_config, mock_asyncua_client
    ):
        mock_asyncua, mock_client = mock_asyncua_client
        # Build: Objects → [Pump A (Object), Temp Var (Variable)]
        var_node = _make_opcua_node("Temperature", "ns=2;i=1002", node_class_value=2)
        var_node.read_value = AsyncMock(return_value=72.5)

        pump_node = _make_opcua_node(
            "Pump A", "ns=2;i=1001", node_class_value=1, children=[var_node]
        )
        objects_node = MagicMock()
        objects_node.get_children = AsyncMock(return_value=[pump_node])
        mock_client.get_objects_node = MagicMock(return_value=objects_node)

        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        conn._client = mock_client  # inject mock directly

        readings = await conn.discover_assets()

        assert len(readings) == 1
        assert readings[0].external_id == "ns=2;i=1001"
        assert readings[0].name == "Pump A"
        assert readings[0].asset_type == "pump"
        assert readings[0].source == "opcua"

    @pytest.mark.asyncio
    async def test_discover_skips_variable_nodes_at_top_level(
        self, opcua_config, mock_asyncua_client
    ):
        mock_asyncua, mock_client = mock_asyncua_client
        # A Variable node at the top level — must be skipped as an "asset"
        var_node = _make_opcua_node("SomeVar", "ns=2;i=9000", node_class_value=2)
        objects_node = MagicMock()
        objects_node.get_children = AsyncMock(return_value=[var_node])
        mock_client.get_objects_node = MagicMock(return_value=objects_node)

        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        conn._client = mock_client

        readings = await conn.discover_assets()
        assert readings == []

    @pytest.mark.asyncio
    async def test_disconnect_calls_client_disconnect(
        self, opcua_config, mock_asyncua_client
    ):
        mock_asyncua, mock_client = mock_asyncua_client
        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        conn._client = mock_client
        await conn.disconnect()
        mock_client.disconnect.assert_awaited_once()

    def test_is_connected_false_when_client_none(self, opcua_config):
        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        assert conn.is_connected() is False

    def test_is_connected_true_when_client_alive(self, opcua_config, mock_asyncua_client):
        mock_asyncua, mock_client = mock_asyncua_client
        mock_client.uaclient.is_connected.return_value = True
        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        conn._client = mock_client
        assert conn.is_connected() is True

    def test_is_connected_false_on_exception(self, opcua_config, mock_asyncua_client):
        mock_asyncua, mock_client = mock_asyncua_client
        mock_client.uaclient.is_connected.side_effect = RuntimeError("dead")
        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        conn._client = mock_client
        assert conn.is_connected() is False

    @pytest.mark.asyncio
    async def test_poll_uses_discovered_nodes(self, opcua_config, mock_asyncua_client):
        mock_asyncua, mock_client = mock_asyncua_client
        var_node = _make_opcua_node("Vibration", "ns=2;i=1002", node_class_value=2)
        var_node.read_value = AsyncMock(return_value=3.0)

        pump_node = _make_opcua_node(
            "Pump B", "ns=2;i=1001", node_class_value=1, children=[var_node]
        )
        objects_node = MagicMock()
        objects_node.get_children = AsyncMock(return_value=[pump_node])
        mock_client.get_objects_node = MagicMock(return_value=objects_node)
        mock_client.get_node = MagicMock(return_value=pump_node)

        from server.services.connectors.opcua_connector import OpcUaConnector
        conn = OpcUaConnector(opcua_config)
        conn._client = mock_client

        # First discover
        await conn.discover_assets()
        # Then poll
        readings = await conn.poll()
        assert len(readings) == 1
        assert readings[0].name == "Pump B"

    @pytest.mark.asyncio
    async def test_opcua_config_from_json(self):
        from server.services.connectors.opcua_connector import OpcUaConfig
        cfg = OpcUaConfig.from_json(
            json.dumps({
                "endpoint_url": "opc.tcp://192.168.1.10:4840",
                "username": "user",
                "password": "pass",
                "poll_interval": 15,
            })
        )
        assert cfg.endpoint_url == "opc.tcp://192.168.1.10:4840"
        assert cfg.poll_interval == 15


# ── ConnectorManager ──────────────────────────────────────────────────────────


class TestConnectorManager:
    @pytest.mark.asyncio
    async def test_run_discovery_calls_connector_discover(self, client, auth_headers):
        """ConnectorManager.run_discovery must call discover_assets and upsert assets."""
        import json
        from server.services.connector_manager import ConnectorManager
        from server.services.connectors.base import HealthReading
        from server.models_assets import ConnectorConfig
        from sqlmodel import Session
        from server.database import engine

        mock_connector = MagicMock()
        mock_connector.is_connected.return_value = False
        mock_connector.connect = AsyncMock()
        mock_connector.discover_assets = AsyncMock(return_value=[
            HealthReading(
                external_id="mgr-pump-001",
                name="Manager Pump",
                asset_type="pump",
                source="simulator",
                health_score=80.0,
                severity="ok",
            )
        ])

        # Create env + real connector config row (so db.add works in run_discovery)
        env_resp = client.post(
            "/assets/environments",
            json={"name": "Manager Test Env", "env_type": "factory"},
            headers=auth_headers,
        )
        assert env_resp.status_code == 201
        env_id = env_resp.json()["id"]

        conn_resp = client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "simulator",
                "name": "Test Sim",
                "config_json": json.dumps({"assets": []}),
            },
            headers=auth_headers,
        )
        conn_id = conn_resp.json()["id"]

        with Session(engine) as db:
            real_cfg = db.get(ConnectorConfig, conn_id)
            manager = ConnectorManager(env_id)
            # Pair the mock connector with the real config row
            manager._connectors[conn_id] = (mock_connector, real_cfg)
            report = await manager.run_discovery(db)

        assert mock_connector.connect.await_count >= 1
        assert mock_connector.discover_assets.await_count == 1
        assert report["discovered"] == 1
        assert report["errors"] == []

    def test_stop_cancels_tasks(self):
        from server.services.connector_manager import ConnectorManager
        manager = ConnectorManager("env-stop-test")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        manager._tasks["t1"] = mock_task
        manager.stop()
        mock_task.cancel.assert_called_once()

    def test_load_from_db_skips_disabled_connectors(self, client, auth_headers):
        """load_from_db must only instantiate enabled connectors."""
        from server.services.connector_manager import ConnectorManager

        resp = client.post(
            "/assets/environments",
            json={"name": "Load Test Env", "env_type": "factory"},
            headers=auth_headers,
        )
        env_id = resp.json()["id"]

        # Add a disabled connector
        client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "simulator",
                "name": "Disabled Sim",
                "config_json": json.dumps({"assets": []}),
                "enabled": False,
            },
            headers=auth_headers,
        )

        from sqlmodel import Session
        from server.database import engine
        manager = ConnectorManager(env_id)
        with Session(engine) as db:
            manager.load_from_db(db)

        assert len(manager._connectors) == 0


# ── Discovery HTTP endpoint ───────────────────────────────────────────────────


class TestDiscoveryEndpoint:
    @pytest.mark.asyncio
    async def test_discover_endpoint_returns_report(self, client, auth_headers):
        """POST /assets/environments/{env_id}/discover must return a discovery report."""
        # Create env + simulator connector
        env_resp = client.post(
            "/assets/environments",
            json={"name": "Discovery EP Test", "env_type": "factory"},
            headers=auth_headers,
        )
        env_id = env_resp.json()["id"]

        client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "simulator",
                "name": "Test Sim",
                "config_json": json.dumps({
                    "assets": [
                        {"external_id": "ep-pump-1", "name": "EP Pump 1",
                         "asset_type": "pump", "base_score": 85.0},
                    ]
                }),
                "enabled": True,
            },
            headers=auth_headers,
        )

        resp = client.post(
            f"/assets/environments/{env_id}/discover",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "discovered" in body
        assert body["discovered"] >= 1
        assert body["errors"] == []

    def test_discover_endpoint_requires_auth(self, client):
        resp = client.post("/assets/environments/some-id/discover")
        assert resp.status_code in (401, 403)

    def test_discover_endpoint_unknown_env_returns_404(self, client, auth_headers):
        resp = client.post(
            "/assets/environments/nonexistent-env/discover",
            headers=auth_headers,
        )
        assert resp.status_code == 404
