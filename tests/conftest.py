"""Shared pytest fixtures for FactoryMind test suite.

Environment variables are set at the TOP of this module – before any server
imports – so that pydantic-settings picks up the test values when
server/config.py is first imported.
"""

import os
import tempfile

# ── Must be set BEFORE any server imports ────────────────────────────────────
_db_dir = tempfile.mkdtemp(prefix="fmind_test_")
os.environ.setdefault("SECRET_KEY", "test-secret-key-32-chars-minimum!!")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_dir}/test.db")
os.environ.setdefault("ENABLE_REGISTRATION", "true")

# ── Now safe to import server modules ────────────────────────────────────────
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server.services.vllm_manager as _vllm_mod  # noqa: E402


def _make_mock_manager():
    """Return a MagicMock that satisfies VLLMModelManager's interface."""
    m = MagicMock()
    m.get_info.return_value = {
        "current_model": "llama-3.1-8b-awq",
        "is_ready": True,
        "catalog": list(_vllm_mod.MODEL_CATALOG.keys()),
    }
    m.ensure_loaded = MagicMock()  # no-op by default (success)
    m.aforward_chat = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "Test response from mock"}}],
        }
    )
    m.is_cached = MagicMock(return_value=False)
    return m


# Inject mock before the app's lifespan touches vLLM
_vllm_mod._instance = _make_mock_manager()


from server.main import app  # noqa: E402  (import after patching)


@pytest.fixture(scope="session")
def client():
    """HTTP test client with DB + lifespan wired up."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers(client):
    """Register once, return the X-API-Key header dict for all tests."""
    resp = client.post(
        "/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123",
        },
    )
    assert resp.status_code == 201, resp.text
    api_key = resp.json()["api_key"]
    return {"X-API-Key": api_key}


@pytest.fixture()
def mock_manager():
    """Return the module-level mock manager for per-test manipulation."""
    return _vllm_mod._instance
