import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from server.database import create_db_and_tables
from server.routers import auth, chat, health, history, models

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    try:
        create_db_and_tables()
    except RuntimeError as exc:
        # Catches _check_writable() errors (read-only DB, bad permissions, …)
        log.critical("Database initialisation failed: %s", exc)
        raise SystemExit(f"\n[FATAL] {exc}\n") from exc
    except OperationalError as exc:
        log.critical("Could not connect to database: %s", exc)
        raise SystemExit(f"\n[FATAL] Cannot connect to database: {exc}\n") from exc

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    from server.services.vllm_manager import get_vllm_manager
    get_vllm_manager().stop()


app = FastAPI(
    title="FactoryMind AI API",
    description="Enterprise LLM inference platform for industrial environments.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(OperationalError)
async def db_operational_error(request: Request, exc: OperationalError):
    """Catch SQLite/Postgres operational errors (locked, readonly, down, …)."""
    msg = str(exc.orig) if exc.orig else str(exc)
    log.error("Database operational error on %s %s: %s", request.method, request.url.path, msg)
    return JSONResponse(
        status_code=503,
        content={"detail": f"Database unavailable: {msg}"},
    )


@app.exception_handler(SQLAlchemyError)
async def db_generic_error(request: Request, exc: SQLAlchemyError):
    """Catch any other unexpected SQLAlchemy error and return 503."""
    log.error("Unexpected database error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "A database error occurred. Check server logs."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(history.router)
