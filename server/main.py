import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from contextlib import contextmanager

from server.config import settings
from server.database import create_db_and_tables, engine
import server.models_assets  # noqa: F401 — registers SQLModel tables before create_all()
from server.routers import auth, chat, health, history, models
from server.routers import assets
from server.routers import connectors
from server.services import connector_manager as cm


@contextmanager
def _db_factory():
    """Yield a fresh Session; used by connector polling loops."""
    from sqlmodel import Session
    with Session(engine) as session:
        yield session

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Database ───────────────────────────────────────────────────────────
    try:
        create_db_and_tables()
    except RuntimeError as exc:
        log.critical("Database initialisation failed: %s", exc)
        raise SystemExit(f"\n[FATAL] {exc}\n") from exc
    except OperationalError as exc:
        log.critical("Could not connect to database: %s", exc)
        raise SystemExit(f"\n[FATAL] Cannot connect to database: {exc}\n") from exc

    # ── 2. vLLM warm-up ───────────────────────────────────────────────────────
    from server.services.vllm_manager import MODEL_CATALOG, get_vllm_manager

    manager = get_vllm_manager()

    if settings.default_model not in MODEL_CATALOG:
        raise SystemExit(
            f"\n[FATAL] DEFAULT_MODEL='{settings.default_model}' is not in the "
            f"model catalog. Available: {list(MODEL_CATALOG)}\n"
        )

    log.info(
        "Warming up default model '%s' – server will accept traffic once ready…",
        settings.default_model,
    )
    try:
        await run_in_threadpool(manager.ensure_loaded, settings.default_model)
        log.info("Model '%s' loaded and ready.", settings.default_model)
    except Exception as exc:
        log.critical(
            "Failed to load default model '%s': %s", settings.default_model, exc
        )
        manager.stop()
        raise SystemExit(
            f"\n[FATAL] Cannot start vLLM for model '{settings.default_model}': {exc}\n"
        ) from exc

    # ── 3. Connector polling loops ────────────────────────────────────────────
    try:
        cm.load_and_start_all(_db_factory)
        log.info("Connector polling loops started.")
    except Exception as exc:
        log.warning("Connector startup non-fatal error: %s", exc)

    # ── 4. Background prefetch (disk only – no GPU) ───────────────────────────
    prefetch_task: Optional[asyncio.Task] = None
    if settings.prefetch_all_models:
        log.info(
            "PREFETCH_ALL_MODELS=True – downloading uncached weights in background…"
        )
        prefetch_task = asyncio.create_task(manager.prefetch_weights())

    # ── Server is now ready ───────────────────────────────────────────────────
    yield

    # ── 4. Shutdown ───────────────────────────────────────────────────────────
    if prefetch_task and not prefetch_task.done():
        log.info("Cancelling background prefetch task…")
        prefetch_task.cancel()
        try:
            await prefetch_task
        except asyncio.CancelledError:
            pass

    log.info("Shutting down – stopping connector polling loops…")
    cm.stop_all()

    log.info("Shutting down – stopping vLLM process…")
    manager.stop()


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
app.include_router(assets.router)
app.include_router(connectors.router)
