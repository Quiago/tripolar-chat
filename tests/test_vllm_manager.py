"""Tests for VLLMModelManager HF token propagation and error handling.

Root-cause being tested:
  vllm_manager.start() overrides HF_HOME in the subprocess env, which moves
  the token lookup path away from where `hf auth login` stored it.
  The subprocess therefore fails to authenticate with HuggingFace Hub.

  test_start_includes_hf_token_in_subprocess_env → FAILS before fix
  test_hf_token_read_from_env_var               → PASSES (sanity check)
  test_hf_token_read_from_token_file            → PASSES (sanity check)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import server.services.vllm_manager as vllm_mod
from server.services.vllm_manager import _read_hf_token


# ── _read_hf_token unit tests ─────────────────────────────────────────────────

def test_hf_token_read_from_hf_token_env(monkeypatch):
    """HF_TOKEN env var is the highest-priority source."""
    monkeypatch.setenv("HF_TOKEN", "tok_from_env")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert _read_hf_token() == "tok_from_env"


def test_hf_token_read_from_legacy_env_var(monkeypatch):
    """HUGGING_FACE_HUB_TOKEN is checked when HF_TOKEN is absent."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok_legacy")
    assert _read_hf_token() == "tok_legacy"


def test_hf_token_read_from_token_file(monkeypatch, tmp_path):
    """Falls back to reading ~/.cache/huggingface/token (or $HF_HOME/token)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    # Simulate hf auth login writing the token to the default HF_HOME
    fake_hf_home = tmp_path / ".cache" / "huggingface"
    fake_hf_home.mkdir(parents=True)
    (fake_hf_home / "token").write_text("tok_from_file\n")

    monkeypatch.setenv("HF_HOME", str(fake_hf_home))
    assert _read_hf_token() == "tok_from_file"


def test_hf_token_returns_none_when_absent(monkeypatch, tmp_path):
    """Returns None when neither env vars nor token file exist."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    empty_hf_home = tmp_path / "empty_hf"
    empty_hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(empty_hf_home))

    assert _read_hf_token() is None


# ── subprocess env token propagation ─────────────────────────────────────────

def test_start_includes_hf_token_in_subprocess_env(monkeypatch, tmp_path):
    """start() must pass HF_TOKEN to the vLLM subprocess env.

    BUG (before fix): env["HF_HOME"] is overridden without preserving the
    token, so vLLM can't authenticate to HuggingFace Hub.

    This test FAILS before the fix and PASSES after.
    """
    # Simulate a logged-in user: token lives in the real HF_HOME
    fake_hf_home = tmp_path / ".cache" / "huggingface"
    fake_hf_home.mkdir(parents=True)
    (fake_hf_home / "token").write_text("hf_test_token")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(fake_hf_home))

    captured_env: dict = {}

    def _fake_popen(cmd, env, **kwargs):
        captured_env.update(env)
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        return proc

    manager = vllm_mod.VLLMModelManager(port=9999, cache_dir=str(tmp_path / "cache"))

    with patch("subprocess.Popen", side_effect=_fake_popen):
        # is_ready() must return True so start() returns immediately
        with patch.object(manager, "is_ready", return_value=True):
            manager.start("Llama-2-7b-chat-hf")

    assert "HF_TOKEN" in captured_env, (
        "HF_TOKEN was NOT passed to the vLLM subprocess env. "
        "vLLM cannot authenticate to HuggingFace Hub."
    )
    assert captured_env["HF_TOKEN"] == "hf_test_token"


def test_start_sets_custom_hf_home(tmp_path):
    """start() must still set HF_HOME to the cache_dir for weight storage."""
    captured_env: dict = {}

    def _fake_popen(cmd, env, **kwargs):
        captured_env.update(env)
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    cache_dir = tmp_path / "models"
    manager = vllm_mod.VLLMModelManager(port=9999, cache_dir=str(cache_dir))

    with patch("subprocess.Popen", side_effect=_fake_popen):
        with patch.object(manager, "is_ready", return_value=True):
            manager.start("Llama-2-7b-chat-hf")

    assert captured_env.get("HF_HOME") == str(cache_dir)


# ── 503 detail is brief ───────────────────────────────────────────────────────

def test_chat_503_detail_is_brief(client, auth_headers, mock_manager):
    """The 503 detail returned to the client must be a short human message,
    not a multi-line Python traceback.

    BUG (before fix): detail contains the full stderr from vLLM including
    file paths, line numbers, and exception chains.
    """
    # Simulate what happens when start() raises after reading a long stderr
    long_traceback = (
        "vLLM exited unexpectedly:\n"
        'Traceback (most recent call last):\n'
        '  File "/some/deep/path/transformers/utils/hub.py", line 511, in cached_files\n'
        "    raise OSError(\n"
        "OSError: casperhansen/llama-3.1-8b-instruct-awq is not a local folder "
        "and is not a valid model identifier listed on "
        "'https://huggingface.co/models'\n"
        "If this is a private repository, make sure to pass a token...\n"
    )
    mock_manager.ensure_loaded.side_effect = RuntimeError(long_traceback)

    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 503
    detail = resp.json()["detail"]

    # The detail must be a single short line, not a multi-line stacktrace
    assert "\n" not in detail, (
        "503 detail contains newlines – the full traceback is leaking to the client."
    )
    assert len(detail) < 120, (
        f"503 detail is too long ({len(detail)} chars): {detail!r}"
    )

    # Restore
    mock_manager.ensure_loaded.side_effect = None
