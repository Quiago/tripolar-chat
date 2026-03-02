"""Tests for APIClient streaming methods.

Key bug covered here:
  async_chat_stream() does NOT check the HTTP status code before iterating
  SSE lines.  A 503 from the server silently produces zero output instead of
  raising APIError.

  test_async_chat_stream_raises_api_error_on_503 → FAILS before the fix
                                                  → PASSES after the fix
"""

import json
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from client.api import APIClient, APIError


# ── Helpers – mock httpx streaming transport ──────────────────────────────────

class _FakeStreamResponse:
    """Minimal mock of an httpx streaming response."""

    def __init__(self, status_code: int, lines: list[str], json_body: dict | None = None):
        self.status_code = status_code
        self.is_error = status_code >= 400
        self._lines = lines
        self._json_body = json_body or {"detail": f"HTTP {status_code}"}

    async def aread(self):
        """Called by _raise_for() after detecting is_error."""
        pass

    def json(self) -> dict:
        return self._json_body

    @property
    def text(self) -> str:
        return json.dumps(self._json_body)

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    # Async context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeAsyncHTTPClient:
    """Mock of httpx.AsyncClient that returns a pre-built response."""

    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    def stream(self, *args, **kwargs) -> _FakeStreamResponse:
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _sse_lines(chunks: list[str]) -> list[str]:
    """Build SSE data lines from a list of text chunks."""
    lines = []
    for chunk in chunks:
        payload = json.dumps({"choices": [{"delta": {"content": chunk}}]})
        lines.append(f"data: {payload}")
    lines.append("data: [DONE]")
    return lines


# ── Sync chat_stream (already correct – regression guard) ─────────────────────

def test_chat_stream_raises_api_error_on_403():
    """Sync chat_stream must raise APIError on 403."""
    api = APIClient("http://localhost:8080", "bad-key")

    class _FakeSyncStreamResponse:
        status_code = 403
        is_error = True

        def read(self):
            pass

        def json(self):
            return {"detail": "Forbidden"}

        @property
        def text(self):
            return '{"detail": "Forbidden"}'

        def iter_lines(self):
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _FakeSyncHTTPClient:
        def stream(self, *args, **kwargs):
            return _FakeSyncStreamResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("httpx.Client", return_value=_FakeSyncHTTPClient()):
        with pytest.raises(APIError) as exc_info:
            list(api.chat_stream("llama-3.1-8b-awq", [{"role": "user", "content": "hi"}]))
        assert exc_info.value.status_code == 403


def test_chat_stream_yields_content_on_200():
    """Sync chat_stream must yield text chunks from SSE lines."""
    api = APIClient("http://localhost:8080", "good-key")
    expected_chunks = ["Hello", " world", "!"]

    class _FakeSyncStreamResponse:
        status_code = 200
        is_error = False

        def read(self):
            pass

        def iter_lines(self):
            return iter(_sse_lines(expected_chunks))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _FakeSyncHTTPClient:
        def stream(self, *args, **kwargs):
            return _FakeSyncStreamResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("httpx.Client", return_value=_FakeSyncHTTPClient()):
        result = list(api.chat_stream("llama-3.1-8b-awq", [{"role": "user", "content": "hi"}]))

    assert result == expected_chunks


# ── Async chat_stream ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_chat_stream_raises_api_error_on_503():
    """async_chat_stream must raise APIError when the server responds with 503.

    BUG (before fix): The method skips the status check and silently yields
    nothing, making failures invisible to the caller (TUI shows empty output).

    This test FAILS before the fix and PASSES after.
    """
    error_response = _FakeStreamResponse(
        status_code=503,
        lines=[],
        json_body={"detail": "Database unavailable"},
    )
    mock_client = _FakeAsyncHTTPClient(error_response)

    api = APIClient("http://localhost:8080", "test-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(APIError) as exc_info:
            async for _ in api.async_chat_stream(
                "llama-3.1-8b-awq",
                [{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_async_chat_stream_raises_api_error_on_403():
    """async_chat_stream must raise APIError on 403 (expired API key)."""
    error_response = _FakeStreamResponse(
        status_code=403,
        lines=[],
        json_body={"detail": "Invalid API key"},
    )
    mock_client = _FakeAsyncHTTPClient(error_response)

    api = APIClient("http://localhost:8080", "expired-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(APIError) as exc_info:
            async for _ in api.async_chat_stream(
                "llama-3.1-8b-awq",
                [{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_async_chat_stream_yields_content_on_200():
    """async_chat_stream must yield text chunks on a successful 200 response."""
    expected_chunks = ["Hi", " there", "!"]
    ok_response = _FakeStreamResponse(
        status_code=200,
        lines=_sse_lines(expected_chunks),
    )
    mock_client = _FakeAsyncHTTPClient(ok_response)

    api = APIClient("http://localhost:8080", "good-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = [
            chunk
            async for chunk in api.async_chat_stream(
                "llama-3.1-8b-awq",
                [{"role": "user", "content": "hi"}],
            )
        ]

    assert result == expected_chunks
