"""GET /v1/models and POST /v1/models/{name}/load endpoints."""

import threading

from fastapi import APIRouter, Depends, HTTPException

from server.models import User
from server.routers.auth import get_current_user
from server.services.vllm_manager import MODEL_CATALOG, get_vllm_manager

router = APIRouter(tags=["models"])


@router.get("/v1/models")
def list_models(current_user: User = Depends(get_current_user)):
    """Return catalog with cached/loaded status for each model."""
    manager = get_vllm_manager()
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": "factorymind",
                "description": config.description,
                "cached": manager.is_cached(name),
                "loaded": manager.current_model == name,
            }
            for name, config in MODEL_CATALOG.items()
        ],
    }


@router.post("/v1/models/{model_name}/load", status_code=202)
def load_model(
    model_name: str,
    current_user: User = Depends(get_current_user),
):
    """Trigger model loading in a background thread.

    Returns immediately with status=loading. Poll GET /v1/models or GET /health
    to check when the model becomes ready (loaded=true).
    """
    if model_name not in MODEL_CATALOG:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not in catalog. Available: {list(MODEL_CATALOG)}",
        )

    manager = get_vllm_manager()
    if manager.current_model == model_name and manager.is_ready():
        return {"status": "already_loaded", "model": model_name}

    def _load() -> None:
        try:
            manager.ensure_loaded(model_name)
        except Exception:
            pass  # errors surface on next /v1/models poll

    threading.Thread(target=_load, daemon=True).start()

    return {
        "status": "loading",
        "model": model_name,
        "message": "Model loading started. Poll GET /v1/models for status.",
    }
