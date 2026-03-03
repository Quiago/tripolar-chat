"""Test suite for the Asset Registry + Health Ingestion feature (Prompt 1).

Coverage:
  Environments  — create, list, get, ownership enforcement
  Connectors    — add, list, partial update (enable/disable)
  Ingest        — single event, auto-discovery, batch (happy + edge cases)
  Assets        — list with status, per-asset status, history, fleet summary
  Acknowledge   — marks event, rejects cross-asset mismatch
  Access control— 403 on resources owned by another user
"""

import pytest


# ── Module-scoped state ───────────────────────────────────────────────────────
# Stored as module globals so session-scoped fixtures can share IDs across tests.
# The conftest `client` and `auth_headers` fixtures are session-scoped, meaning
# one DB and one registered user for the whole module run.

_ENV_ID: str = ""
_CONN_ID: str = ""
_ASSET_ID: str = ""
_EVENT_ID: str = ""


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env_id(client, auth_headers):
    """Create one environment for all tests in this module."""
    resp = client.post(
        "/assets/environments",
        json={
            "name": "Test Factory Floor",
            "env_type": "factory",
            "location": "Dubai, UAE",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    global _ENV_ID
    _ENV_ID = resp.json()["id"]
    return _ENV_ID


@pytest.fixture(scope="module")
def ingested_event(client, auth_headers, env_id):
    """Ingest one event (auto-creates asset) and return the full response."""
    resp = client.post(
        "/assets/ingest",
        json={
            "environment_id": env_id,
            "external_id": "pump-001",
            "source": "opcua",
            "health_score": 85.0,
            "severity": "ok",
            "asset_name": "Centrifugal Pump A",
            "asset_type": "pump",
            "vendor": "Siemens",
            "raw_value": 2.3,
            "raw_unit": "mm/s",
            "message": "Nominal vibration",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture(scope="module")
def asset_id(client, auth_headers, env_id, ingested_event):
    """Return the asset_id for pump-001 created by the first ingest."""
    resp = client.get(
        f"/assets/environments/{env_id}/assets",
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assets = resp.json()
    # Locate the specific asset created by ingested_event (external_id="pump-001")
    asset = next(a for a in assets if a["external_id"] == "pump-001")
    global _ASSET_ID
    _ASSET_ID = asset["asset_id"]
    return _ASSET_ID


@pytest.fixture(scope="module")
def event_id(ingested_event):
    """Return the event_id from the first ingest."""
    global _EVENT_ID
    _EVENT_ID = ingested_event["event_ids"][0]
    return _EVENT_ID


# ── Environments ──────────────────────────────────────────────────────────────


class TestEnvironments:
    def test_create_environment_returns_201(self, client, auth_headers):
        resp = client.post(
            "/assets/environments",
            json={"name": "Throwaway Env", "env_type": "custom"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Throwaway Env"
        assert body["env_type"] == "custom"
        assert "id" in body

    def test_create_environment_persists_location(self, client, auth_headers):
        resp = client.post(
            "/assets/environments",
            json={"name": "DC East", "env_type": "datacenter", "location": "Abu Dhabi"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["location"] == "Abu Dhabi"

    def test_list_environments_includes_created(self, client, auth_headers, env_id):
        resp = client.get("/assets/environments", headers=auth_headers)
        assert resp.status_code == 200
        ids = [e["id"] for e in resp.json()]
        assert env_id in ids

    def test_get_environment_by_id(self, client, auth_headers, env_id):
        resp = client.get(f"/assets/environments/{env_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == env_id
        assert resp.json()["name"] == "Test Factory Floor"

    def test_get_environment_not_found_returns_404(self, client, auth_headers):
        resp = client.get("/assets/environments/nonexistent-id-xyz", headers=auth_headers)
        assert resp.status_code == 404

    def test_environment_requires_auth(self, client):
        resp = client.get("/assets/environments")
        assert resp.status_code in (401, 403)


# ── Connectors ────────────────────────────────────────────────────────────────


class TestConnectors:
    def test_add_connector_returns_201(self, client, auth_headers, env_id):
        resp = client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "opcua",
                "name": "Plant Floor OPC-UA",
                "config_json": '{"host": "192.168.1.10", "port": 4840}',
                "poll_interval_seconds": 10,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["connector_type"] == "opcua"
        assert body["enabled"] is True
        global _CONN_ID
        _CONN_ID = body["id"]

    def test_list_connectors_includes_added(self, client, auth_headers, env_id):
        resp = client.get(
            f"/assets/environments/{env_id}/connectors",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_update_connector_disable(self, client, auth_headers, env_id):
        # Re-create a connector to avoid depending on _CONN_ID ordering
        add_resp = client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "modbus",
                "name": "Modbus RTU Bus",
                "config_json": '{"port": "/dev/ttyUSB0"}',
            },
            headers=auth_headers,
        )
        conn_id = add_resp.json()["id"]

        patch_resp = client.patch(
            f"/assets/environments/{env_id}/connectors/{conn_id}",
            json={"enabled": False},
            headers=auth_headers,
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["enabled"] is False

    def test_update_connector_change_poll_interval(self, client, auth_headers, env_id):
        add_resp = client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "snmp",
                "name": "SNMP Poller",
                "config_json": '{"community": "public"}',
                "poll_interval_seconds": 60,
            },
            headers=auth_headers,
        )
        conn_id = add_resp.json()["id"]

        patch_resp = client.patch(
            f"/assets/environments/{env_id}/connectors/{conn_id}",
            json={"poll_interval_seconds": 120},
            headers=auth_headers,
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["poll_interval_seconds"] == 120

    def test_update_nonexistent_connector_returns_404(self, client, auth_headers, env_id):
        resp = client.patch(
            f"/assets/environments/{env_id}/connectors/bad-id",
            json={"enabled": False},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_connector_config_json_not_exposed(self, client, auth_headers, env_id):
        """ConnectorConfigPublic must never include config_json (credentials risk)."""
        add_resp = client.post(
            f"/assets/environments/{env_id}/connectors",
            json={
                "connector_type": "http_push",
                "name": "Webhook Receiver",
                "config_json": '{"token": "super_secret_token"}',
            },
            headers=auth_headers,
        )
        body = add_resp.json()
        assert "config_json" not in body


# ── Ingest ────────────────────────────────────────────────────────────────────


class TestIngest:
    def test_ingest_single_event_returns_201(self, client, auth_headers, env_id):
        resp = client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": "crac-unit-3",
                "source": "bacnet",
                "health_score": 92.5,
                "severity": "ok",
                "asset_name": "CRAC Unit 3",
                "asset_type": "crac_unit",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["ingested"] == 1
        assert body["failed"] == 0
        assert len(body["event_ids"]) == 1

    def test_ingest_auto_discovers_new_asset(self, client, auth_headers, env_id):
        """An unknown external_id must create a new asset automatically."""
        external_id = "motor-auto-discovered-xyz"
        client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": external_id,
                "source": "simulator",
                "health_score": 70.0,
                "severity": "info",
            },
            headers=auth_headers,
        )
        assets_resp = client.get(
            f"/assets/environments/{env_id}/assets",
            headers=auth_headers,
        )
        external_ids = [a["external_id"] for a in assets_resp.json()]
        assert external_id in external_ids

    def test_ingest_idempotent_asset_upsert(self, client, auth_headers, env_id):
        """Ingesting the same external_id twice must not create duplicate assets."""
        payload = {
            "environment_id": env_id,
            "external_id": "pump-idempotent-test",
            "source": "simulator",
            "health_score": 80.0,
            "severity": "ok",
        }
        client.post("/assets/ingest", json=payload, headers=auth_headers)
        client.post("/assets/ingest", json=payload, headers=auth_headers)

        assets_resp = client.get(
            f"/assets/environments/{env_id}/assets",
            headers=auth_headers,
        )
        count = sum(
            1 for a in assets_resp.json() if a["external_id"] == "pump-idempotent-test"
        )
        assert count == 1

    def test_ingest_to_unknown_environment_returns_404(self, client, auth_headers):
        resp = client.post(
            "/assets/ingest",
            json={
                "environment_id": "does-not-exist",
                "external_id": "pump-x",
                "source": "opcua",
                "health_score": 50.0,
                "severity": "warning",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_ingest_batch_returns_all_event_ids(self, client, auth_headers, env_id):
        events = [
            {
                "environment_id": env_id,
                "external_id": f"batch-asset-{i}",
                "source": "simulator",
                "health_score": float(80 - i * 5),
                "severity": "ok" if i < 3 else "warning",
            }
            for i in range(5)
        ]
        resp = client.post(
            "/assets/ingest/batch",
            json={"events": events},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["ingested"] == 5
        assert body["failed"] == 0
        assert len(body["event_ids"]) == 5

    def test_ingest_batch_exceeds_limit_returns_422(self, client, auth_headers, env_id):
        events = [
            {
                "environment_id": env_id,
                "external_id": f"overflow-{i}",
                "source": "simulator",
                "health_score": 80.0,
                "severity": "ok",
            }
            for i in range(501)
        ]
        resp = client.post(
            "/assets/ingest/batch",
            json={"events": events},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_ingest_requires_auth(self, client, env_id):
        resp = client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": "pump-no-auth",
                "source": "opcua",
                "health_score": 80.0,
                "severity": "ok",
            },
        )
        assert resp.status_code in (401, 403)


# ── Assets & Status ───────────────────────────────────────────────────────────


class TestAssetStatus:
    def test_list_assets_returns_status_fields(self, client, auth_headers, env_id, ingested_event):
        resp = client.get(
            f"/assets/environments/{env_id}/assets",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assets = resp.json()
        assert len(assets) >= 1
        first = assets[0]
        assert "asset_id" in first
        assert "latest_severity" in first
        assert "latest_health_score" in first

    def test_get_asset_status_by_id(self, client, auth_headers, asset_id):
        resp = client.get(f"/assets/{asset_id}/status", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["asset_id"] == asset_id
        assert body["latest_health_score"] is not None

    def test_get_asset_status_not_found(self, client, auth_headers):
        resp = client.get("/assets/nonexistent-asset-id/status", headers=auth_headers)
        assert resp.status_code == 404

    def test_get_asset_history_returns_events(self, client, auth_headers, asset_id):
        resp = client.get(f"/assets/{asset_id}/history", headers=auth_headers)
        assert resp.status_code == 200
        events = resp.json()
        assert isinstance(events, list)
        assert len(events) >= 1
        assert "health_score" in events[0]
        assert "severity" in events[0]

    def test_get_asset_history_limit_param(self, client, auth_headers, env_id):
        """History endpoint must respect ?limit= query parameter."""
        # Ingest 5 events for a dedicated asset
        ext_id = "asset-history-limit-test"
        for i in range(5):
            client.post(
                "/assets/ingest",
                json={
                    "environment_id": env_id,
                    "external_id": ext_id,
                    "source": "simulator",
                    "health_score": float(90 - i),
                    "severity": "ok",
                },
                headers=auth_headers,
            )
        # Get the asset id
        assets = client.get(
            f"/assets/environments/{env_id}/assets", headers=auth_headers
        ).json()
        aid = next(a["asset_id"] for a in assets if a["external_id"] == ext_id)

        resp = client.get(f"/assets/{aid}/history?limit=2", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ── Fleet Summary ──────────────────────────────────────────────────────────────


class TestFleetSummary:
    def test_fleet_summary_returns_total_assets(self, client, auth_headers, env_id, asset_id):
        resp = client.get(
            f"/assets/environments/{env_id}/summary",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_assets"] >= 1
        assert isinstance(body["by_severity"], dict)
        assert isinstance(body["critical_assets"], list)
        assert isinstance(body["degrading_assets"], list)

    def test_fleet_summary_counts_severity_correctly(self, client, auth_headers, env_id):
        """Ingest one critical and one warning event; summary must reflect both."""
        client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": "pump-critical-test",
                "source": "simulator",
                "health_score": 10.0,
                "severity": "critical",
                "failure_mode": "bearing_fault",
            },
            headers=auth_headers,
        )
        client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": "pump-warning-test",
                "source": "simulator",
                "health_score": 55.0,
                "severity": "warning",
            },
            headers=auth_headers,
        )
        resp = client.get(
            f"/assets/environments/{env_id}/summary",
            headers=auth_headers,
        )
        body = resp.json()
        assert body["by_severity"].get("critical", 0) >= 1
        assert body["by_severity"].get("warning", 0) >= 1

        critical_ids = {a["external_id"] for a in body["critical_assets"]}
        assert "pump-critical-test" in critical_ids

        degrading_ids = {a["external_id"] for a in body["degrading_assets"]}
        assert "pump-warning-test" in degrading_ids

    def test_fleet_summary_not_found_returns_404(self, client, auth_headers):
        resp = client.get(
            "/assets/environments/no-such-env/summary",
            headers=auth_headers,
        )
        assert resp.status_code == 404


# ── Acknowledge ───────────────────────────────────────────────────────────────


class TestAcknowledge:
    def test_acknowledge_event_sets_flag(self, client, auth_headers, asset_id, event_id):
        resp = client.post(
            f"/assets/{asset_id}/acknowledge",
            json={"event_id": event_id},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["acknowledged"] is True
        assert body["acknowledged_by"] is not None
        assert body["acknowledged_at"] is not None

    def test_acknowledge_nonexistent_event_returns_404(self, client, auth_headers, asset_id):
        resp = client.post(
            f"/assets/{asset_id}/acknowledge",
            json={"event_id": "bad-event-id"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_acknowledge_event_wrong_asset_returns_422(
        self, client, auth_headers, env_id, asset_id
    ):
        """Acknowledging an event via the wrong asset path must return 422."""
        # Ingest a second asset and get its event
        ingest_resp = client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": "pump-ack-wrong-asset",
                "source": "simulator",
                "health_score": 60.0,
                "severity": "warning",
            },
            headers=auth_headers,
        )
        other_event_id = ingest_resp.json()["event_ids"][0]

        # Try to acknowledge it via a different asset's path
        resp = client.post(
            f"/assets/{asset_id}/acknowledge",
            json={"event_id": other_event_id},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ── Access control ────────────────────────────────────────────────────────────


class TestAccessControl:
    @pytest.fixture(scope="class")
    def other_user_headers(self, client):
        """Register a second user and return their auth headers."""
        resp = client.post(
            "/auth/register",
            json={
                "username": "otherassetuser",
                "email": "other_asset@example.com",
                "password": "otherpass123",
            },
        )
        assert resp.status_code == 201, resp.text
        return {"X-API-Key": resp.json()["api_key"]}

    def test_other_user_cannot_list_assets(
        self, client, other_user_headers, env_id
    ):
        resp = client.get(
            f"/assets/environments/{env_id}/assets",
            headers=other_user_headers,
        )
        assert resp.status_code == 403

    def test_other_user_cannot_get_environment(
        self, client, other_user_headers, env_id
    ):
        resp = client.get(
            f"/assets/environments/{env_id}",
            headers=other_user_headers,
        )
        assert resp.status_code == 403

    def test_other_user_cannot_get_asset_status(
        self, client, other_user_headers, asset_id
    ):
        resp = client.get(
            f"/assets/{asset_id}/status",
            headers=other_user_headers,
        )
        assert resp.status_code == 403

    def test_other_user_cannot_ingest_to_foreign_environment(
        self, client, other_user_headers, env_id
    ):
        resp = client.post(
            "/assets/ingest",
            json={
                "environment_id": env_id,
                "external_id": "hacker-pump",
                "source": "simulator",
                "health_score": 50.0,
                "severity": "warning",
            },
            headers=other_user_headers,
        )
        assert resp.status_code == 403
