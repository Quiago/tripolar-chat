"""TDD tests for POST /connectors/query — the live data gateway endpoint.

This endpoint is stateless: the connector config is passed inline, data is
fetched on-demand, cleaned, and returned.  DB writes are opt-in via the
store_readings flag + environment_id.

Coverage:
  Schema validation   — missing required fields, unknown connector type
  Simulator query     — happy path, returns clean readings
  Smart auto          — discovers on first call, polls on subsequent
  Optional DB store   — readings stored when environment_id + store_readings
  OPC-UA (mocked)     — wires through real OpcUaConnector with asyncua mocked
  Auth                — 401/403 without valid API key
  Access control      — cannot store to another user's environment
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _simulator_config(n: int = 2) -> dict:
    return {
        "assets": [
            {"external_id": f"q-pump-{i}", "name": f"Query Pump {i}",
             "asset_type": "pump", "base_score": 85.0}
            for i in range(n)
        ]
    }


# ── Schema validation ─────────────────────────────────────────────────────────


class TestConnectorQuerySchema:
    def test_missing_connector_type_returns_422(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"config": {"assets": []}},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_missing_config_returns_422(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_unknown_connector_type_returns_400(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "mqtt_v99_unknown", "config": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_requires_auth(self, client):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator", "config": _simulator_config()},
        )
        assert resp.status_code in (401, 403)


# ── Simulator query — happy path ──────────────────────────────────────────────


class TestSimulatorQuery:
    def test_returns_200(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator", "config": _simulator_config(3)},
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator", "config": _simulator_config(2)},
            headers=auth_headers,
        )
        body = resp.json()
        assert body["connector_type"] == "simulator"
        assert body["asset_count"] == 2
        assert len(body["readings"]) == 2
        assert "queried_at" in body
        assert isinstance(body["stored"], int)

    def test_readings_have_clean_fields(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator", "config": _simulator_config(1)},
            headers=auth_headers,
        )
        reading = resp.json()["readings"][0]
        assert "external_id" in reading
        assert "name" in reading
        assert "asset_type" in reading
        assert "health_score" in reading
        assert "severity" in reading
        assert reading["severity"] in ("ok", "info", "warning", "critical")
        assert 0.0 <= reading["health_score"] <= 100.0

    def test_no_environment_id_does_not_store(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator", "config": _simulator_config(2)},
            headers=auth_headers,
        )
        assert resp.json()["stored"] == 0

    def test_empty_asset_list_returns_empty_readings(self, client, auth_headers):
        resp = client.post(
            "/connectors/query",
            json={"connector_type": "simulator", "config": {"assets": []}},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["asset_count"] == 0
        assert resp.json()["readings"] == []


# ── Smart auto: discover vs poll ──────────────────────────────────────────────


class TestSmartAuto:
    def test_without_env_id_always_discovers(self, client, auth_headers):
        """Without environment_id there's no cached state → always discovers."""
        payload = {
            "connector_type": "simulator",
            "config": _simulator_config(2),
        }
        r1 = client.post("/connectors/query", json=payload, headers=auth_headers)
        r2 = client.post("/connectors/query", json=payload, headers=auth_headers)
        # Both should succeed and return the same asset count
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["asset_count"] == r2.json()["asset_count"]

    def test_with_env_id_second_call_uses_poll(self, client, auth_headers):
        """With environment_id the manager caches the connector; second call polls."""
        env_resp = client.post(
            "/assets/environments",
            json={"name": "Smart Auto Env", "env_type": "factory"},
            headers=auth_headers,
        )
        env_id = env_resp.json()["id"]

        payload = {
            "connector_type": "simulator",
            "config": _simulator_config(2),
            "environment_id": env_id,
            "store_readings": False,
        }
        r1 = client.post("/connectors/query", json=payload, headers=auth_headers)
        r2 = client.post("/connectors/query", json=payload, headers=auth_headers)

        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both return the same number of assets
        assert r1.json()["asset_count"] == r2.json()["asset_count"] == 2
        # First call is discover, second is poll — scores may differ slightly
        # (random walk), but both are valid
        for reading in r2.json()["readings"]:
            assert 0.0 <= reading["health_score"] <= 100.0


# ── Optional DB store ──────────────────────────────────────────────────────────


class TestOptionalStore:
    def test_store_readings_true_writes_to_db(self, client, auth_headers):
        """store_readings=True + environment_id must persist readings to DB."""
        env_resp = client.post(
            "/assets/environments",
            json={"name": "Store Test Env", "env_type": "factory"},
            headers=auth_headers,
        )
        env_id = env_resp.json()["id"]

        resp = client.post(
            "/connectors/query",
            json={
                "connector_type": "simulator",
                "config": _simulator_config(2),
                "environment_id": env_id,
                "store_readings": True,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stored"] == 2

        # Assets must appear in the registry
        assets_resp = client.get(
            f"/assets/environments/{env_id}/assets",
            headers=auth_headers,
        )
        assert assets_resp.status_code == 200
        assert len(assets_resp.json()) >= 2

    def test_store_readings_false_does_not_write(self, client, auth_headers):
        """store_readings=False (default) must never write to DB."""
        env_resp = client.post(
            "/assets/environments",
            json={"name": "No Store Env", "env_type": "factory"},
            headers=auth_headers,
        )
        env_id = env_resp.json()["id"]

        resp = client.post(
            "/connectors/query",
            json={
                "connector_type": "simulator",
                "config": _simulator_config(2),
                "environment_id": env_id,
                "store_readings": False,
            },
            headers=auth_headers,
        )
        assert resp.json()["stored"] == 0

        assets_resp = client.get(
            f"/assets/environments/{env_id}/assets",
            headers=auth_headers,
        )
        assert assets_resp.json() == []

    def test_store_requires_environment_id(self, client, auth_headers):
        """store_readings=True without environment_id must return 422."""
        resp = client.post(
            "/connectors/query",
            json={
                "connector_type": "simulator",
                "config": _simulator_config(1),
                "store_readings": True,
                # environment_id intentionally missing
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_store_to_foreign_environment_returns_403(self, client, auth_headers):
        """Cannot store readings to another user's environment."""
        # Register a second user
        other = client.post(
            "/auth/register",
            json={"username": "cq_other", "email": "cq_other@test.com",
                  "password": "pass123456"},
        )
        other_key = {"X-API-Key": other.json()["api_key"]}

        # Other user creates an environment
        other_env = client.post(
            "/assets/environments",
            json={"name": "Other Env", "env_type": "factory"},
            headers=other_key,
        )
        other_env_id = other_env.json()["id"]

        # First user tries to store to other's environment
        resp = client.post(
            "/connectors/query",
            json={
                "connector_type": "simulator",
                "config": _simulator_config(1),
                "environment_id": other_env_id,
                "store_readings": True,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 403


# ── OPC-UA (mocked asyncua) ───────────────────────────────────────────────────


class TestOpcUaQuery:
    def test_opcua_query_returns_readings(self, client, auth_headers):
        """OPC-UA connector must be invoked and readings returned (asyncua mocked)."""
        from tests.test_connectors import _make_opcua_node

        var_node = _make_opcua_node("Temperature", "ns=2;i=1002", node_class_value=2)
        var_node.read_value = AsyncMock(return_value=72.0)
        pump_node = _make_opcua_node(
            "Pump Alpha", "ns=2;i=1001", node_class_value=1, children=[var_node]
        )
        objects_node = MagicMock()
        objects_node.get_children = AsyncMock(return_value=[pump_node])

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.get_objects_node = MagicMock(return_value=objects_node)
        mock_client.uaclient.is_connected.return_value = True

        with patch(
            "server.services.connectors.opcua_connector.asyncua"
        ) as mock_asyncua:
            mock_asyncua.Client.return_value = mock_client

            resp = client.post(
                "/connectors/query",
                json={
                    "connector_type": "opcua",
                    "config": {"endpoint_url": "opc.tcp://localhost:4840"},
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["connector_type"] == "opcua"
        assert body["asset_count"] == 1
        assert body["readings"][0]["name"] == "Pump Alpha"
        assert body["readings"][0]["asset_type"] == "pump"

    def test_opcua_connection_failure_returns_503(self, client, auth_headers):
        """If the OPC-UA server is unreachable, return 503 with a clear message."""
        with patch(
            "server.services.connectors.opcua_connector.asyncua"
        ) as mock_asyncua:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock(
                side_effect=ConnectionRefusedError("Connection refused")
            )
            mock_asyncua.Client.return_value = mock_client

            resp = client.post(
                "/connectors/query",
                json={
                    "connector_type": "opcua",
                    "config": {"endpoint_url": "opc.tcp://unreachable:4840"},
                },
                headers=auth_headers,
            )

        assert resp.status_code == 503
        assert "detail" in resp.json()
