"""HTTP client wrapper for the FactoryMind API."""

from typing import Any

import httpx

from client import config as cfg


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"[{status_code}] {detail}")


def _client(server_url: str, api_key: str | None = None) -> httpx.Client:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return httpx.Client(base_url=server_url, headers=headers, timeout=30.0)


def _raise_for(response: httpx.Response) -> None:
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(response.status_code, detail)


# ── Auth ──────────────────────────────────────────────────────────────────────

def register(server_url: str, username: str, email: str, password: str) -> dict[str, Any]:
    with _client(server_url) as c:
        r = c.post("/auth/register", json={"username": username, "email": email, "password": password})
    _raise_for(r)
    return r.json()


def login(server_url: str, username: str, password: str) -> dict[str, Any]:
    with _client(server_url) as c:
        r = c.post("/auth/login", json={"username": username, "password": password})
    _raise_for(r)
    return r.json()


def whoami(server_url: str, api_key: str) -> dict[str, Any]:
    with _client(server_url, api_key) as c:
        r = c.get("/auth/me")
    _raise_for(r)
    return r.json()


def rotate_key(server_url: str, api_key: str) -> dict[str, Any]:
    with _client(server_url, api_key) as c:
        r = c.post("/auth/rotate-key")
    _raise_for(r)
    return r.json()


def health(server_url: str) -> dict[str, Any]:
    with _client(server_url) as c:
        r = c.get("/health")
    _raise_for(r)
    return r.json()
