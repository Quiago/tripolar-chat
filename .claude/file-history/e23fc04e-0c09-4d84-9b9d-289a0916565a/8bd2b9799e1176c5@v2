"""Integration tests for POST /v1/chat/completions.

These tests use a mocked VLLMModelManager (injected in conftest.py) so no
real GPU or vLLM process is needed.
"""

import pytest


# ── Auth guard ────────────────────────────────────────────────────────────────

def test_chat_requires_auth(client):
    # Missing header → 401 (APIKeyHeader auto_error); bad key → 403
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 401

    resp2 = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "invalid-key"},
        json={
            "model": "Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp2.status_code == 403


# ── Input validation ──────────────────────────────────────────────────────────

def test_chat_unknown_model_returns_400(client, auth_headers):
    """Requesting a model not in MODEL_CATALOG must return 400."""
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "not-a-real-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 400
    assert "not in the catalog" in resp.json()["detail"]


# ── Model loading failure ─────────────────────────────────────────────────────

def test_chat_ensure_loaded_failure_returns_503(client, auth_headers, mock_manager):
    """When ensure_loaded raises, the endpoint must return 503."""
    mock_manager.ensure_loaded.side_effect = RuntimeError("GPU OOM")

    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 503
    assert "Llama-2-7b-chat-hf" in resp.json()["detail"]

    # Restore default behavior for subsequent tests
    mock_manager.ensure_loaded.side_effect = None


# ── Non-streaming success ─────────────────────────────────────────────────────

def test_chat_non_streaming_returns_200(client, auth_headers, mock_manager):
    """Non-streaming chat must return a full completion object."""
    mock_manager.ensure_loaded.side_effect = None
    mock_manager.aforward_chat.return_value = {
        "choices": [{"message": {"content": "Hello from mock!"}}],
    }

    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    # Response must contain choices and the chat_id
    assert "choices" in body
    assert "chat_id" in body
    assert body["choices"][0]["message"]["content"] == "Hello from mock!"


# ── Streaming success ─────────────────────────────────────────────────────────

def test_chat_streaming_returns_event_stream(client, auth_headers, mock_manager):
    """When stream=True, the response Content-Type must be text/event-stream."""
    mock_manager.ensure_loaded.side_effect = None

    # Streaming path calls the real vLLM port which doesn't exist in tests.
    # We verify the 200 SSE headers are present; actual SSE content comes from
    # the vLLM mock at the transport layer (covered by test_api_client tests).
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Stream test"}],
            "stream": True,
        },
    ) as resp:
        # The response starts as 200 with text/event-stream
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "x-chat-id" in resp.headers


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_includes_model_manager_info(client, mock_manager):
    """GET /health must include database and model_manager sections."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "database" in body
    assert body["database"]["status"] == "ok"
    assert "model_manager" in body
    assert "catalog" in body["model_manager"]


# ── Models endpoint ───────────────────────────────────────────────────────────

def test_list_models_returns_catalog(client, auth_headers):
    """GET /v1/models must return the model catalog."""
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    model_ids = [m["id"] for m in body["data"]]
    assert "Llama-2-7b-chat-hf" in model_ids


# ── History endpoint ──────────────────────────────────────────────────────────

def test_history_initially_empty(client, auth_headers):
    """A brand-new user should have zero chat sessions in history."""
    # Register a fresh user so history is clean
    import uuid
    fresh_user = f"fresh_{uuid.uuid4().hex[:8]}"
    reg = client.post(
        "/auth/register",
        json={
            "username": fresh_user,
            "email": f"{fresh_user}@example.com",
            "password": "pass1234",
        },
    )
    assert reg.status_code == 201
    fresh_headers = {"X-API-Key": reg.json()["api_key"]}

    resp = client.get("/v1/history", headers=fresh_headers)
    assert resp.status_code == 200
    assert resp.json() == []
