from datetime import datetime

from fastapi import APIRouter
from sqlalchemy import text

from server.database import engine
from server.services.vllm_manager import get_vllm_manager

router = APIRouter(tags=["health"])


def _db_status() -> dict:
    """Return a brief database health snapshot."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": _db_status(),
        "model_manager": get_vllm_manager().get_info(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
