"""Connector query router — live data gateway.

POST /connectors/query
    Stateless: connector config is passed inline, data is fetched on-demand,
    cleaned, and returned.  DB writes are opt-in via store_readings + environment_id.

Smart-auto behaviour:
    - No environment_id  → always runs discover_assets() (no cached state).
    - With environment_id → connector is cached; first call runs discover_assets(),
      subsequent calls run poll() (cheaper, avoids re-browsing the source).

Error mapping:
    - Unknown connector_type            → 400
    - store_readings without env_id     → 422
    - Foreign / missing environment     → 403 / 404
    - Data source unreachable           → 503
"""

import asyncio
import json as _json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from server.core.exceptions import AccessDeniedError, EnvironmentNotFoundError
from server.database import get_db
from server.models import User
from server.models_assets import (
    ConnectorQueryRequest,
    ConnectorQueryResponse,
    ConnectorReadingPublic,
)
from server.routers.auth import get_current_user
from server.services import asset_service
from server.services.connector_manager import _CONNECTOR_FACTORIES
from server.services.connectors.base import BaseConnector

log = logging.getLogger(__name__)

router = APIRouter(prefix="/connectors", tags=["connectors"])

# ── Smart-auto connector cache ────────────────────────────────────────────────
# Key: (environment_id, connector_type) → connected BaseConnector instance.
# Only populated when environment_id is provided; cleared on disconnect.
_query_cache: dict[tuple[str, str], BaseConnector] = {}


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post("/query", response_model=ConnectorQueryResponse)
async def query_connector(
    payload: ConnectorQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConnectorQueryResponse:
    """Fetch live data from a connector (config passed inline).

    Args:
        payload: connector_type, inline config dict, optional environment_id
            and store_readings flag.
        db: Injected database session.
        current_user: Authenticated user from API key.

    Returns:
        ConnectorQueryResponse with cleaned readings and metadata.

    Raises:
        400: Unknown connector_type.
        422: store_readings=True but no environment_id.
        403: environment_id refers to another user's environment.
        404: environment_id not found.
        503: Data source unreachable or connector error.
    """
    # ── Schema validation ─────────────────────────────────────────────────────
    if payload.connector_type not in _CONNECTOR_FACTORIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown connector_type '{payload.connector_type}'. "
                f"Available: {sorted(_CONNECTOR_FACTORIES)}"
            ),
        )

    if payload.store_readings and not payload.environment_id:
        raise HTTPException(
            status_code=422,
            detail="store_readings=True requires environment_id.",
        )

    # ── Ownership check (only when storing) ───────────────────────────────────
    if payload.environment_id and payload.store_readings:
        try:
            asset_service.get_environment(db, payload.environment_id, current_user.id)
        except EnvironmentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except AccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

    # ── Build or retrieve connector ───────────────────────────────────────────
    cache_key: Optional[tuple[str, str]] = (
        (payload.environment_id, payload.connector_type)
        if payload.environment_id
        else None
    )

    connector: Optional[BaseConnector] = None
    use_poll = False

    if cache_key and cache_key in _query_cache:
        cached = _query_cache[cache_key]
        if cached.is_connected():
            connector = cached
            use_poll = True
        else:
            # Stale cache entry — discard and rebuild
            del _query_cache[cache_key]

    if connector is None:
        factory = _CONNECTOR_FACTORIES[payload.connector_type]
        connector = factory(payload.config)

    # ── Connect and fetch (with timeout) ──────────────────────────────────────
    async def _fetch() -> list:
        if not connector.is_connected():
            await connector.connect()
        return await (connector.poll() if use_poll else connector.discover_assets())

    try:
        readings = await asyncio.wait_for(_fetch(), timeout=payload.timeout_seconds)

    except asyncio.TimeoutError:
        log.warning(
            "Connector '%s' timed out after %ds",
            payload.connector_type,
            payload.timeout_seconds,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"{payload.connector_type} connector timed out after "
                f"{payload.timeout_seconds}s. Pass a larger timeout_seconds value "
                f"in the request body if the node tree is large."
            ),
        )
    except (ConnectionRefusedError, ConnectionError, OSError, TimeoutError) as exc:
        log.warning("Connector '%s' unreachable: %s", payload.connector_type, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to {payload.connector_type} data source: {exc}",
        )
    except Exception as exc:
        log.error("Connector '%s' error: %s", payload.connector_type, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Connector error ({payload.connector_type}): {exc}",
        )

    # ── Cache or disconnect ───────────────────────────────────────────────────
    if cache_key:
        _query_cache[cache_key] = connector
    else:
        try:
            await connector.disconnect()
        except Exception:
            pass  # best-effort cleanup

    # ── Optional DB store ─────────────────────────────────────────────────────
    stored = 0
    if payload.environment_id and payload.store_readings:
        for reading in readings:
            try:
                asset_service.ingest_event(
                    db,
                    environment_id=payload.environment_id,
                    external_id=reading.external_id,
                    source=reading.source,
                    health_score=reading.health_score,
                    severity=reading.severity,
                    failure_mode=reading.failure_mode,
                    raw_value=reading.raw_value,
                    raw_unit=reading.raw_unit,
                    message=reading.message,
                    raw_payload=(
                        _json.dumps(reading.raw_payload) if reading.raw_payload else None
                    ),
                    timestamp=reading.timestamp,
                    asset_name=reading.name,
                    asset_type=reading.asset_type,
                )
                stored += 1
            except Exception as exc:
                log.error(
                    "Failed to store reading for asset '%s': %s",
                    reading.external_id,
                    exc,
                )

    # ── Build response ────────────────────────────────────────────────────────
    public_readings = [
        ConnectorReadingPublic(
            external_id=r.external_id,
            name=r.name,
            asset_type=r.asset_type,
            source=r.source,
            health_score=r.health_score,
            severity=r.severity,
            failure_mode=r.failure_mode,
            raw_value=r.raw_value,
            raw_unit=r.raw_unit,
            message=r.message,
        )
        for r in readings
    ]

    return ConnectorQueryResponse(
        connector_type=payload.connector_type,
        asset_count=len(public_readings),
        readings=public_readings,
        stored=stored,
        queried_at=datetime.utcnow(),
    )
