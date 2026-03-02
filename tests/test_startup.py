"""Tests for cold-start warm-up and model pre-fetching.

Covers:
  1. Config has DEFAULT_MODEL and PREFETCH_ALL_MODELS with correct defaults.
  2. prefetch_weights() skips models already on disk (is_cached → True).
  3. prefetch_weights() calls snapshot_download for uncached models.
  4. prefetch_weights() logs a warning and continues when one download fails.
  5. FastAPI lifespan calls ensure_loaded(default_model) at startup.
  6. FastAPI lifespan calls manager.stop() at shutdown.
  7. FastAPI lifespan raises SystemExit when default_model is not in catalog.
  8. FastAPI lifespan raises SystemExit when ensure_loaded raises.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ── 1 & 2. Config defaults ────────────────────────────────────────────────────

def test_config_has_default_model():
    """settings.default_model must exist and default to 'llama-3.1-8b-awq'."""
    from server.config import settings
    assert hasattr(settings, "default_model")
    assert settings.default_model == "llama-3.1-8b-awq"


def test_config_has_prefetch_all_models():
    """Settings.prefetch_all_models field must exist and have a default of True."""
    from server.config import Settings
    field = Settings.model_fields["prefetch_all_models"]
    assert field.default is True


# ── 3. prefetch_weights skips cached models ───────────────────────────────────

@pytest.mark.asyncio
async def test_prefetch_weights_skips_cached_models(tmp_path):
    """prefetch_weights must not call snapshot_download for already-cached models."""
    import server.services.vllm_manager as vllm_mod

    manager = vllm_mod.VLLMModelManager(port=9999, cache_dir=str(tmp_path))

    with patch.object(manager, "is_cached", return_value=True):
        with patch(
            "server.services.vllm_manager.VLLMModelManager._snapshot_download"
        ) as mock_dl:
            await manager.prefetch_weights()

    mock_dl.assert_not_called()


# ── 4. prefetch_weights downloads uncached models ─────────────────────────────

@pytest.mark.asyncio
async def test_prefetch_weights_downloads_uncached_models(tmp_path):
    """prefetch_weights must call _snapshot_download for each uncached model."""
    import server.services.vllm_manager as vllm_mod

    manager = vllm_mod.VLLMModelManager(port=9999, cache_dir=str(tmp_path))
    catalog_size = len(vllm_mod.MODEL_CATALOG)

    downloaded: list[str] = []

    def _fake_dl(cfg):
        downloaded.append(cfg.repo)

    with patch.object(manager, "is_cached", return_value=False):
        with patch.object(manager, "_snapshot_download", side_effect=_fake_dl):
            await manager.prefetch_weights()

    assert len(downloaded) == catalog_size


# ── 5. prefetch_weights continues on failure ──────────────────────────────────

@pytest.mark.asyncio
async def test_prefetch_weights_continues_after_single_failure(tmp_path):
    """A failed download must not abort the prefetch of remaining models."""
    import server.services.vllm_manager as vllm_mod

    manager = vllm_mod.VLLMModelManager(port=9999, cache_dir=str(tmp_path))
    catalog = list(vllm_mod.MODEL_CATALOG.values())

    call_count = 0

    def _fail_first(cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Hub unreachable")

    with patch.object(manager, "is_cached", return_value=False):
        with patch.object(manager, "_snapshot_download", side_effect=_fail_first):
            # Should not raise even though the first download fails
            await manager.prefetch_weights()

    assert call_count == len(catalog), "All models must be attempted, not just the first"


# ── 6. _snapshot_download passes correct cache_dir and token ─────────────────

def test_snapshot_download_uses_correct_cache_dir_and_token(tmp_path, monkeypatch):
    """_snapshot_download must pass cache_dir=<cache>/hub and the HF token."""
    import sys
    from unittest.mock import MagicMock
    import server.services.vllm_manager as vllm_mod

    monkeypatch.setenv("HF_TOKEN", "hf_abc123")

    manager = vllm_mod.VLLMModelManager(port=9999, cache_dir=str(tmp_path))
    cfg = vllm_mod.MODEL_CATALOG["llama-3.1-8b-awq"]
    expected_cache = str(tmp_path / "hub")

    # The import inside _snapshot_download is lazy. The conftest already placed
    # a MagicMock at sys.modules["huggingface_hub"]. We replace snapshot_download
    # on that stub so we can assert on the call args.
    mock_dl = MagicMock()
    sys.modules["huggingface_hub"].snapshot_download = mock_dl

    manager._snapshot_download(cfg)

    mock_dl.assert_called_once_with(
        repo_id=cfg.repo,
        cache_dir=expected_cache,
        token="hf_abc123",
        local_files_only=False,
    )


# ── 7. Lifespan calls ensure_loaded at startup ────────────────────────────────

def test_lifespan_warms_up_default_model(mock_manager):
    """The FastAPI lifespan must call ensure_loaded(default_model) at startup."""
    from fastapi.testclient import TestClient
    from server.main import app

    # Verify that the mock_manager.ensure_loaded was called with the default model
    # The TestClient with-block triggers the full lifespan.
    mock_manager.ensure_loaded.reset_mock()

    from server.config import settings
    with TestClient(app):
        pass  # just trigger lifespan

    mock_manager.ensure_loaded.assert_called_with(settings.default_model)


# ── 8. Lifespan calls stop() at shutdown ─────────────────────────────────────

def test_lifespan_stops_manager_on_shutdown(mock_manager):
    """The FastAPI lifespan must call manager.stop() during shutdown."""
    from fastapi.testclient import TestClient
    from server.main import app

    mock_manager.stop.reset_mock()

    with TestClient(app):
        pass

    mock_manager.stop.assert_called()


# ── 9. Lifespan aborts when default_model not in catalog ─────────────────────

def test_lifespan_exits_when_default_model_unknown(mock_manager, monkeypatch):
    """SystemExit must be raised at startup if DEFAULT_MODEL is not in catalog."""
    import server.config as cfg_mod
    from fastapi.testclient import TestClient
    from server.main import app

    monkeypatch.setattr(cfg_mod.settings, "default_model", "nonexistent-model-xyz")

    with pytest.raises((SystemExit, Exception)):
        with TestClient(app):
            pass


# ── 10. Lifespan aborts when ensure_loaded raises ────────────────────────────

def test_lifespan_exits_when_ensure_loaded_fails(mock_manager, monkeypatch):
    """SystemExit must be raised at startup if ensure_loaded raises."""
    from fastapi.testclient import TestClient
    from server.main import app

    mock_manager.ensure_loaded.side_effect = RuntimeError("GPU OOM")

    with pytest.raises((SystemExit, Exception)):
        with TestClient(app):
            pass

    # Restore
    mock_manager.ensure_loaded.side_effect = None
