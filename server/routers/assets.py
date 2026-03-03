"""Assets router — environments, connectors, assets, and health events.

All endpoints require authentication via get_current_user.

Ingest endpoints (/assets/ingest*) are on the hot path: they must stay
thin — validate input, call the service, return the response. No business
logic here.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from server.core.exceptions import (
    AccessDeniedError,
    AssetNotFoundError,
    ConnectorNotFoundError,
    EnvironmentNotFoundError,
    EventNotFoundError,
)
from server.database import get_db
from server.models import User
from server.models_assets import (
    AcknowledgeRequest,
    AssetStatusPublic,
    BatchIngestRequest,
    ConnectorConfigCreate,
    ConnectorConfigPublic,
    ConnectorConfigUpdate,
    EnvironmentCreate,
    EnvironmentPublic,
    FleetSummary,
    HealthEventPublic,
    IngestRequest,
    IngestResponse,
)
from server.routers.auth import get_current_user
from server.services import asset_service
from server.services import connector_manager as cm

router = APIRouter(prefix="/assets", tags=["assets"])

# Maximum events allowed per batch ingest call
_BATCH_LIMIT = 500


# ── Exception → HTTP mapping ──────────────────────────────────────────────────


def _map_domain_errors(exc: Exception) -> HTTPException:
    """Convert a domain exception to the appropriate HTTPException.

    Args:
        exc: A FactoryMind domain exception.

    Returns:
        HTTPException with the appropriate status code and detail.
    """
    if isinstance(exc, (EnvironmentNotFoundError, AssetNotFoundError,
                        ConnectorNotFoundError, EventNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AccessDeniedError):
        return HTTPException(status_code=403, detail=str(exc))
    raise exc  # unexpected — let the global handler take it


# ── Environments ──────────────────────────────────────────────────────────────


@router.post("/environments", response_model=EnvironmentPublic, status_code=201)
def create_environment(
    payload: EnvironmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new environment owned by the authenticated user."""
    env = asset_service.create_environment(
        db,
        user_id=current_user.id,
        name=payload.name,
        env_type=payload.env_type,
        location=payload.location,
        config_json=payload.config_json,
    )
    return env


@router.get("/environments", response_model=List[EnvironmentPublic])
def list_environments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all environments owned by the authenticated user."""
    return asset_service.get_environments(db, user_id=current_user.id)


@router.get("/environments/{env_id}", response_model=EnvironmentPublic)
def get_environment(
    env_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return details for a single environment."""
    try:
        return asset_service.get_environment(db, env_id=env_id, user_id=current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)


# ── Connectors ────────────────────────────────────────────────────────────────


@router.post(
    "/environments/{env_id}/connectors",
    response_model=ConnectorConfigPublic,
    status_code=201,
)
def add_connector(
    env_id: str,
    payload: ConnectorConfigCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Add a connector configuration to an environment."""
    try:
        return asset_service.add_connector(db, env_id, current_user.id, payload)
    except Exception as exc:
        raise _map_domain_errors(exc)


@router.get(
    "/environments/{env_id}/connectors",
    response_model=List[ConnectorConfigPublic],
)
def list_connectors(
    env_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all connector configs for an environment."""
    try:
        return asset_service.list_connectors(db, env_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)


@router.patch(
    "/environments/{env_id}/connectors/{conn_id}",
    response_model=ConnectorConfigPublic,
)
def update_connector(
    env_id: str,
    conn_id: str,
    patch: ConnectorConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enable, disable, or update a connector config."""
    try:
        return asset_service.update_connector(db, env_id, conn_id, current_user.id, patch)
    except Exception as exc:
        raise _map_domain_errors(exc)


# ── Discovery ─────────────────────────────────────────────────────────────────


@router.post("/environments/{env_id}/discover")
async def discover(
    env_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Trigger auto-discovery for all enabled connectors in an environment.

    Calls ConnectorManager.run_discovery(), which browses every connector
    and upserts discovered assets into the registry.  Returns a report with
    the number of assets discovered and any connector errors.

    Use this during customer onboarding to populate the asset registry
    without manual data entry.
    """
    try:
        asset_service.get_environment(db, env_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)

    manager = cm.get_manager(env_id)
    with db:
        manager.load_from_db(db)
        report = await manager.run_discovery(db)
    return report


# ── Assets ────────────────────────────────────────────────────────────────────


@router.get("/environments/{env_id}/assets", response_model=List[AssetStatusPublic])
def list_assets(
    env_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all active assets in an environment with their latest status."""
    try:
        return asset_service.list_assets_with_status(db, env_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)


@router.get("/environments/{env_id}/summary", response_model=FleetSummary)
def fleet_summary(
    env_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return an aggregated fleet health summary for an environment."""
    try:
        return asset_service.get_fleet_summary(db, env_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)


# ── Ingest ────────────────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest(
    payload: IngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ingest a single health event.

    The environment must be owned by the authenticated user.
    Target write latency: <10 ms.
    """
    try:
        asset_service.get_environment(db, payload.environment_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)

    event = asset_service.ingest_event(
        db,
        environment_id=payload.environment_id,
        external_id=payload.external_id,
        source=payload.source,
        health_score=payload.health_score,
        severity=payload.severity,
        failure_mode=payload.failure_mode,
        raw_value=payload.raw_value,
        raw_unit=payload.raw_unit,
        message=payload.message,
        raw_payload=payload.raw_payload,
        timestamp=payload.timestamp,
        asset_name=payload.asset_name,
        asset_type=payload.asset_type,
        vendor=payload.vendor,
    )
    return IngestResponse(ingested=1, failed=0, event_ids=[event.id])


@router.post("/ingest/batch", response_model=IngestResponse, status_code=201)
def ingest_batch(
    payload: BatchIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ingest up to 500 health events in a single DB transaction.

    All events must belong to environments owned by the authenticated user.
    If any environment check fails the entire batch is rejected (no partial
    writes).
    """
    if len(payload.events) > _BATCH_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(payload.events)} exceeds the limit of {_BATCH_LIMIT}.",
        )

    # Ownership check for every unique environment in the batch upfront.
    env_ids = {req.environment_id for req in payload.events}
    for env_id in env_ids:
        try:
            asset_service.get_environment(db, env_id, current_user.id)
        except Exception as exc:
            raise _map_domain_errors(exc)

    events = asset_service.ingest_events_batch(db, payload.events)
    return IngestResponse(
        ingested=len(events),
        failed=0,
        event_ids=[e.id for e in events],
    )


# ── Per-asset ─────────────────────────────────────────────────────────────────


@router.get("/{asset_id}/status", response_model=AssetStatusPublic)
def asset_status(
    asset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the latest health status for an asset."""
    try:
        return asset_service.get_asset_status(db, asset_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)

@router.get(
    "/environments/{env_id}/assets/by-external/{external_id}",
    response_model=AssetStatusPublic,
)
def asset_status_by_external_id(
    env_id: str,
    external_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return latest health status for an asset, looked up by external_id.

    This is the UX-friendly endpoint; it avoids exposing internal UUIDs.
    """
    try:
        return asset_service.get_asset_status_by_external_id(
            db, environment_id=env_id, external_id=external_id, user_id=current_user.id
        )
    except Exception as exc:
        raise _map_domain_errors(exc)



@router.get("/{asset_id}/history", response_model=List[HealthEventPublic])
def asset_history(
    asset_id: str,
    limit: int = Query(default=50, ge=1, le=1000),
    since: Optional[datetime] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return health event history for an asset (newest first).

    Args:
        asset_id: Asset primary key.
        limit: Max events to return (1–1000, default 50).
        since: Only return events after this UTC timestamp.
    """
    try:
        return asset_service.get_asset_history(
            db, asset_id, current_user.id, limit=limit, since=since
        )
    except Exception as exc:
        raise _map_domain_errors(exc)


@router.post("/{asset_id}/acknowledge", response_model=HealthEventPublic)
def acknowledge(
    asset_id: str,
    body: AcknowledgeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Acknowledge a specific health event belonging to this asset."""
    try:
        event = asset_service.acknowledge_event(db, body.event_id, current_user.id)
    except Exception as exc:
        raise _map_domain_errors(exc)

    # Ensure the event actually belongs to the asset in the path
    if event.asset_id != asset_id:
        raise HTTPException(
            status_code=422,
            detail=f"Event '{body.event_id}' does not belong to asset '{asset_id}'.",
        )
    return event
