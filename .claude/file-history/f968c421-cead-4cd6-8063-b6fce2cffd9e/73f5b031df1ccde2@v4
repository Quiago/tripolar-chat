"""HTTP client wrapper for the FactoryMind API.

Free functions (register, login, whoami, …) are kept for backward-compat.
New code should use the APIClient class.
"""

import json
from typing import Any, AsyncIterator, Iterator, Optional

import httpx

from client import config as cfg


class APIError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"[{status_code}] {detail}")


def _raise_for(response: httpx.Response) -> None:
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(response.status_code, detail)


# ── APIClient class ────────────────────────────────────────────────────────────

class APIClient:
    """Stateful client bound to a server URL + API key."""

    def __init__(self, server_url: str, api_key: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self._headers = {"X-API-Key": api_key}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sync_client(self, timeout: float = 30.0) -> httpx.Client:
        return httpx.Client(
            base_url=self.server_url,
            headers=self._headers,
            timeout=timeout,
        )

    def _get(self, path: str, **params) -> dict:
        with self._sync_client() as c:
            r = c.get(path, params=params)
        _raise_for(r)
        return r.json()

    def _post(self, path: str, body: dict, timeout: float = 30.0) -> dict:
        with self._sync_client(timeout) as c:
            r = c.post(path, json=body)
        _raise_for(r)
        return r.json()

    # ── auth ──────────────────────────────────────────────────────────────────

    def whoami(self) -> dict:
        return self._get("/auth/me")

    def rotate_key(self) -> dict:
        return self._post("/auth/rotate-key", {})

    # ── models ────────────────────────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        return self._get("/v1/models").get("data", [])

    def load_model(self, model_name: str) -> dict:
        return self._post(f"/v1/models/{model_name}/load", {})

    # ── chat (sync streaming – for CLI) ──────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict],
        chat_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Non-streaming chat; returns the full OpenAI-style response."""
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        if chat_id:
            body["chat_id"] = chat_id
        # Long timeout: model may need to load first
        return self._post("/v1/chat/completions", body, timeout=300.0)

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        chat_id: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Sync generator that yields text chunks as they stream in.

        Raises APIError on non-200 responses.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if chat_id:
            body["chat_id"] = chat_id

        with httpx.Client(
            base_url=self.server_url,
            headers=self._headers,
            timeout=300.0,
        ) as client:
            with client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as response:
                if response.is_error:
                    response.read()
                    _raise_for(response)
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = (
                            chunk["choices"][0]["delta"].get("content") or ""
                        )
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ── chat (async streaming – for TUI) ─────────────────────────────────────

    async def async_chat_stream(
        self,
        model: str,
        messages: list[dict],
        chat_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async generator that yields text chunks. Used by the Textual TUI."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if chat_id:
            body["chat_id"] = chat_id

        async with httpx.AsyncClient(
            base_url=self.server_url,
            headers=self._headers,
            timeout=300.0,
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as response:
                if response.is_error:
                    await response.aread()
                    _raise_for(response)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = (
                            chunk["choices"][0]["delta"].get("content") or ""
                        )
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ── history ───────────────────────────────────────────────────────────────

    def list_history(self, limit: int = 20) -> list[dict]:
        return self._get("/v1/history", limit=limit)

    def get_chat(self, chat_id: str) -> dict:
        return self._get(f"/v1/history/{chat_id}")

    # ── health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        with httpx.Client(base_url=self.server_url, timeout=10.0) as c:
            r = c.get("/health")
        _raise_for(r)
        return r.json()


# ── Convenience factory ────────────────────────────────────────────────────────

def get_client() -> APIClient:
    """Build an APIClient from the local config file. Raises if not logged in."""
    url = cfg.get_server_url()
    key = cfg.get_api_key()
    if not url or not key:
        raise APIError(0, "Not logged in. Run `fmind login` first.")
    return APIClient(url, key)


# ── Legacy module-level functions (kept for auth command) ─────────────────────

def _mk(server_url: str, api_key: Optional[str] = None) -> httpx.Client:
    headers = {"X-API-Key": api_key} if api_key else {}
    return httpx.Client(base_url=server_url, headers=headers, timeout=30.0)


def register(
    server_url: str, username: str, email: str, password: str
) -> dict[str, Any]:
    with _mk(server_url) as c:
        r = c.post(
            "/auth/register",
            json={"username": username, "email": email, "password": password},
        )
    _raise_for(r)
    return r.json()


def login(server_url: str, username: str, password: str) -> dict[str, Any]:
    with _mk(server_url) as c:
        r = c.post("/auth/login", json={"username": username, "password": password})
    _raise_for(r)
    return r.json()


def whoami(server_url: str, api_key: str) -> dict[str, Any]:
    with _mk(server_url, api_key) as c:
        r = c.get("/auth/me")
    _raise_for(r)
    return r.json()


def rotate_key(server_url: str, api_key: str) -> dict[str, Any]:
    with _mk(server_url, api_key) as c:
        r = c.post("/auth/rotate-key")
    _raise_for(r)
    return r.json()


def health(server_url: str) -> dict[str, Any]:
    with httpx.Client(base_url=server_url, timeout=10.0) as c:
        r = c.get("/health")
    _raise_for(r)
    return r.json()
