"""Asset service — business logic for environments, assets, and health events.

This module has no FastAPI imports. All functions receive an open SQLModel
Session from the caller and assume the caller controls the transaction
boundary (commit / rollback).

Ownership semantics:
  Every environment is owned by a user. Assets belong to environments.
  HealthEvents belong to assets. Any read or write that touches an asset
  or event must first verify that the calling user owns the environment.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlmodel import Session, select

from server.core.exceptions import (
    AccessDeniedError,
    AssetNotFoundError,
    ConnectorNotFoundError,
    EnvironmentNotFoundError,
    EventNotFoundError,
)
from server.models_assets import (
    Asset,
    AssetStatusPublic,
    ConnectorConfig,
    ConnectorConfigCreate,
    ConnectorConfigUpdate,
    Environment,
    FleetSummary,
    HealthEvent,
    HealthEventPublic,
)


# ── Private helpers ────────────────────────────────────────────────────────────


def _require_environment(db: Session, env_id: str, user_id: str) -> Environment:
    """Return the environment or raise domain exceptions.

    Args:
        db: Open database session.
        env_id: Environment primary key.
        user_id: ID of the requesting user.

    Returns:
        The validated Environment instance.

    Raises:
        EnvironmentNotFoundError: env_id does not exist.
        AccessDeniedError: Environment exists but belongs to another user.
    """
    env = db.get(Environment, env_id)
    if env is None:
        raise EnvironmentNotFoundError(f"Environment '{env_id}' not found.")
    if env.owner_user_id != user_id:
        raise AccessDeniedError("You do not own this environment.")
    return env


def _require_asset(db: Session, asset_id: str, user_id: str) -> Asset:
    """Return the asset after verifying ownership through its environment.

    Args:
        db: Open database session.
        asset_id: Asset primary key.
        user_id: ID of the requesting user.

    Returns:
        The validated Asset instance.

    Raises:
        AssetNotFoundError: asset_id does not exist.
        AccessDeniedError: Asset's environment belongs to another user.
    """
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise AssetNotFoundError(f"Asset '{asset_id}' not found.")
    _require_environment(db, asset.environment_id, user_id)
    return asset


def _build_asset_status(
    asset: Asset, latest_event: Optional[HealthEvent]
) -> AssetStatusPublic:
    """Combine an Asset and its optional latest HealthEvent into a status view.

    Args:
        asset: The asset record.
        latest_event: Most-recent HealthEvent, or None if no events exist.

    Returns:
        AssetStatusPublic with all latest_* fields populated from the event.
    """
    return AssetStatusPublic(
        asset_id=asset.id,
        external_id=asset.external_id,
        environment_id=asset.environment_id,
        name=asset.name,
        asset_type=asset.asset_type,
        criticality=asset.criticality,
        vendor=asset.vendor,
        connector_type=asset.connector_type,
        last_seen=asset.last_seen,
        is_active=asset.is_active,
        latest_event_id=latest_event.id if latest_event else None,
        latest_health_score=latest_event.health_score if latest_event else None,
        latest_severity=latest_event.severity if latest_event else None,
        latest_failure_mode=latest_event.failure_mode if latest_event else None,
        latest_message=latest_event.message if latest_event else None,
        latest_event_timestamp=latest_event.timestamp if latest_event else None,
    )


def _latest_events_for_assets(
    db: Session, asset_ids: List[str]
) -> Dict[str, HealthEvent]:
    """Return the most-recent HealthEvent per asset in a single DB round-trip.

    Uses a MAX(timestamp) subquery so we never fetch the full history.

    Args:
        db: Open database session.
        asset_ids: List of asset primary keys to query.

    Returns:
        Dict mapping asset_id → HealthEvent (only assets that have events).
    """
    if not asset_ids:
        return {}

    subq = (
        select(
            HealthEvent.asset_id,
            func.max(HealthEvent.timestamp).label("max_ts"),
        )
        .where(HealthEvent.asset_id.in_(asset_ids))
        .group_by(HealthEvent.asset_id)
        .subquery()
    )

    rows = db.exec(
        select(HealthEvent).join(
            subq,
            (HealthEvent.asset_id == subq.c.asset_id)
            & (HealthEvent.timestamp == subq.c.max_ts),
        )
    ).all()

    return {e.asset_id: e for e in rows}


# ── Environment ────────────────────────────────────────────────────────────────


def create_environment(
    db: Session,
    user_id: str,
    name: str,
    env_type: str,
    location: Optional[str] = None,
    config_json: Optional[str] = None,
) -> Environment:
    """Create and persist a new environment owned by user_id.

    Args:
        db: Open database session.
        user_id: ID of the owning user.
        name: Human-readable environment name.
        env_type: One of "datacenter", "factory", "hotel", "building", "custom".
        location: Optional geographic description.
        config_json: Optional serialized connector / threshold config.

    Returns:
        The newly created Environment.
    """
    env = Environment(
        name=name,
        env_type=env_type,
        location=location,
        owner_user_id=user_id,
        config_json=config_json,
    )
    db.add(env)
    db.commit()
    db.refresh(env)
    return env


def get_environments(db: Session, user_id: str) -> List[Environment]:
    """Return all environments owned by user_id.

    Args:
        db: Open database session.
        user_id: ID of the requesting user.

    Returns:
        List of Environment records (may be empty).
    """
    return db.exec(
        select(Environment).where(Environment.owner_user_id == user_id)
    ).all()


def get_environment(db: Session, env_id: str, user_id: str) -> Environment:
    """Return a single environment by ID, verifying ownership.

    Args:
        db: Open database session.
        env_id: Environment primary key.
        user_id: ID of the requesting user.

    Returns:
        The Environment.

    Raises:
        EnvironmentNotFoundError: Environment does not exist.
        AccessDeniedError: Environment belongs to another user.
    """
    return _require_environment(db, env_id, user_id)


# ── Connector config ───────────────────────────────────────────────────────────


def add_connector(
    db: Session, env_id: str, user_id: str, payload: ConnectorConfigCreate
) -> ConnectorConfig:
    """Add a new ConnectorConfig to an environment.

    Args:
        db: Open database session.
        env_id: Target environment primary key.
        user_id: ID of the requesting user (ownership check).
        payload: Connector creation payload.

    Returns:
        The created ConnectorConfig.

    Raises:
        EnvironmentNotFoundError: Environment does not exist.
        AccessDeniedError: Environment belongs to another user.
    """
    _require_environment(db, env_id, user_id)
    connector = ConnectorConfig(
        environment_id=env_id,
        connector_type=payload.connector_type,
        name=payload.name,
        enabled=payload.enabled,
        config_json=payload.config_json,
        poll_interval_seconds=payload.poll_interval_seconds,
    )
    db.add(connector)
    db.commit()
    db.refresh(connector)
    return connector


def list_connectors(
    db: Session, env_id: str, user_id: str
) -> List[ConnectorConfig]:
    """List all connector configs for an environment.

    Args:
        db: Open database session.
        env_id: Target environment primary key.
        user_id: ID of the requesting user (ownership check).

    Returns:
        List of ConnectorConfig records.

    Raises:
        EnvironmentNotFoundError: Environment does not exist.
        AccessDeniedError: Environment belongs to another user.
    """
    _require_environment(db, env_id, user_id)
    return db.exec(
        select(ConnectorConfig).where(ConnectorConfig.environment_id == env_id)
    ).all()


def update_connector(
    db: Session,
    env_id: str,
    conn_id: str,
    user_id: str,
    patch: ConnectorConfigUpdate,
) -> ConnectorConfig:
    """Partially update a ConnectorConfig (enable/disable, change config, etc.).

    Args:
        db: Open database session.
        env_id: Environment primary key (used for ownership check).
        conn_id: ConnectorConfig primary key.
        user_id: ID of the requesting user.
        patch: Fields to update; only non-None values are applied.

    Returns:
        The updated ConnectorConfig.

    Raises:
        EnvironmentNotFoundError: Environment does not exist.
        AccessDeniedError: Environment belongs to another user.
        ConnectorNotFoundError: Connector does not exist or belongs to a
            different environment.
    """
    _require_environment(db, env_id, user_id)
    connector = db.get(ConnectorConfig, conn_id)
    if connector is None or connector.environment_id != env_id:
        raise ConnectorNotFoundError(f"Connector '{conn_id}' not found.")

    update_data = patch.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(connector, field, value)

    db.add(connector)
    db.commit()
    db.refresh(connector)
    return connector


# ── Asset ──────────────────────────────────────────────────────────────────────


def upsert_asset(
    db: Session,
    environment_id: str,
    external_id: str,
    name: str,
    asset_type: str,
    connector_type: str,
    *,
    criticality: int = 3,
    vendor: Optional[str] = None,
    connector_meta: Optional[str] = None,
    notes: Optional[str] = None,
) -> Asset:
    """Create or update an asset by its (environment_id, external_id) key.

    Safe to call many times per minute with the same external_id.
    On subsequent calls only last_seen is updated; other fields are not
    overwritten so operator annotations survive repeated ingestion.

    Args:
        db: Open database session.
        environment_id: Parent environment primary key.
        external_id: Vendor/protocol identifier (e.g. "ns=2;i=1001").
        name: Human-readable asset name.
        asset_type: One of the known asset type strings or "custom".
        connector_type: Connector that discovered/is managing this asset.
        criticality: Priority level 1 (low) to 5 (critical).
        vendor: Optional equipment vendor name.
        connector_meta: Optional JSON string with connector-specific metadata.
        notes: Optional operator annotations.

    Returns:
        The created or updated Asset.
    """
    asset = db.exec(
        select(Asset).where(
            Asset.environment_id == environment_id,
            Asset.external_id == external_id,
        )
    ).first()

    now = datetime.utcnow()

    if asset is None:
        asset = Asset(
            environment_id=environment_id,
            external_id=external_id,
            name=name,
            asset_type=asset_type,
            connector_type=connector_type,
            criticality=criticality,
            vendor=vendor,
            connector_meta=connector_meta,
            notes=notes,
            discovered_at=now,
            last_seen=now,
        )
    else:
        asset.last_seen = now

    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def list_assets_with_status(
    db: Session, env_id: str, user_id: str
) -> List[AssetStatusPublic]:
    """Return all active assets in an environment with their latest status.

    Args:
        db: Open database session.
        env_id: Target environment primary key.
        user_id: ID of the requesting user (ownership check).

    Returns:
        List of AssetStatusPublic, one per active asset.

    Raises:
        EnvironmentNotFoundError: Environment does not exist.
        AccessDeniedError: Environment belongs to another user.
    """
    _require_environment(db, env_id, user_id)

    assets = db.exec(
        select(Asset).where(Asset.environment_id == env_id, Asset.is_active == True)
    ).all()

    if not assets:
        return []

    asset_ids = [a.id for a in assets]
    latest_map = _latest_events_for_assets(db, asset_ids)

    return [_build_asset_status(a, latest_map.get(a.id)) for a in assets]


# ── Health events ──────────────────────────────────────────────────────────────


def ingest_event(
    db: Session,
    environment_id: str,
    external_id: str,
    source: str,
    health_score: float,
    severity: str,
    *,
    failure_mode: Optional[str] = None,
    raw_value: Optional[float] = None,
    raw_unit: Optional[str] = None,
    message: Optional[str] = None,
    raw_payload: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    asset_name: Optional[str] = None,
    asset_type: Optional[str] = None,
    vendor: Optional[str] = None,
) -> HealthEvent:
    """Ingest a single health event, auto-discovering the asset if necessary.

    This is the hot path — keep it lean. No LLM calls, no heavy logic.
    The asset is upserted first so we always have a valid FK before writing
    the event.

    Args:
        db: Open database session.
        environment_id: Parent environment primary key.
        external_id: Asset's vendor/protocol identifier.
        source: Connector type that generated this reading.
        health_score: Health score in range [0.0, 100.0].
        severity: One of "ok", "info", "warning", "critical".
        failure_mode: Optional failure classification label.
        raw_value: Original sensor value.
        raw_unit: Unit of raw_value (e.g. "°C", "mm/s").
        message: Human-readable description.
        raw_payload: Original vendor JSON payload, stringified.
        timestamp: Event time (defaults to server UTC now).
        asset_name: Name hint for auto-discovery (defaults to external_id).
        asset_type: Type hint for auto-discovery (defaults to "custom").
        vendor: Vendor hint for auto-discovery.

    Returns:
        The persisted HealthEvent.
    """
    asset = upsert_asset(
        db,
        environment_id=environment_id,
        external_id=external_id,
        name=asset_name or external_id,
        asset_type=asset_type or "custom",
        connector_type=source,
        vendor=vendor,
    )

    event = HealthEvent(
        asset_id=asset.id,
        timestamp=timestamp or datetime.utcnow(),
        source=source,
        health_score=health_score,
        severity=severity,
        failure_mode=failure_mode,
        raw_value=raw_value,
        raw_unit=raw_unit,
        message=message,
        raw_payload=raw_payload,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def ingest_events_batch(
    db: Session,
    requests: List,  # List[IngestRequest] — avoid circular import
) -> List[HealthEvent]:
    """Ingest multiple health events in a single DB transaction.

    All upserts and inserts are flushed together before the final commit
    so we minimise round-trips and guarantee atomicity.

    Args:
        db: Open database session.
        requests: List of IngestRequest objects (up to 500).

    Returns:
        List of persisted HealthEvent objects (same order as input).
    """
    events: List[HealthEvent] = []

    for req in requests:
        asset = upsert_asset(
            db,
            environment_id=req.environment_id,
            external_id=req.external_id,
            name=req.asset_name or req.external_id,
            asset_type=req.asset_type or "custom",
            connector_type=req.source,
            vendor=req.vendor,
        )
        event = HealthEvent(
            asset_id=asset.id,
            timestamp=req.timestamp or datetime.utcnow(),
            source=req.source,
            health_score=req.health_score,
            severity=req.severity,
            failure_mode=req.failure_mode,
            raw_value=req.raw_value,
            raw_unit=req.raw_unit,
            message=req.message,
            raw_payload=req.raw_payload,
        )
        db.add(event)
        events.append(event)

    # Single commit for the whole batch
    db.commit()
    for event in events:
        db.refresh(event)

    return events


def get_asset_status(
    db: Session, asset_id: str, user_id: str
) -> Optional[AssetStatusPublic]:
    """Return the latest status for a single asset.

    Args:
        db: Open database session.
        asset_id: Asset primary key.
        user_id: ID of the requesting user.

    Returns:
        AssetStatusPublic or None if asset not found / not owned.

    Raises:
        AssetNotFoundError: Asset does not exist.
        AccessDeniedError: Asset's environment belongs to another user.
    """
    asset = _require_asset(db, asset_id, user_id)
    latest_map = _latest_events_for_assets(db, [asset.id])
    return _build_asset_status(asset, latest_map.get(asset.id))


def get_asset_history(
    db: Session, asset_id: str, user_id: str, limit: int = 50, since: Optional[datetime] = None
) -> List[HealthEventPublic]:
    """Return the health event history for an asset, newest first.

    Args:
        db: Open database session.
        asset_id: Asset primary key.
        user_id: ID of the requesting user.
        limit: Maximum number of events to return (default 50).
        since: If provided, return only events after this timestamp.

    Returns:
        List of HealthEventPublic ordered by timestamp descending.

    Raises:
        AssetNotFoundError: Asset does not exist.
        AccessDeniedError: Asset's environment belongs to another user.
    """
    _require_asset(db, asset_id, user_id)

    stmt = (
        select(HealthEvent)
        .where(HealthEvent.asset_id == asset_id)
        .order_by(HealthEvent.timestamp.desc())
        .limit(limit)
    )
    if since is not None:
        stmt = stmt.where(HealthEvent.timestamp > since)

    events = db.exec(stmt).all()
    return [HealthEventPublic.model_validate(e) for e in events]


def get_fleet_summary(
    db: Session, environment_id: str, user_id: str
) -> FleetSummary:
    """Return an aggregated health snapshot for all assets in an environment.

    Critical assets: latest severity == "critical".
    Degrading assets: latest severity == "warning".
    by_severity: count of assets per severity bucket (unknown if no events).

    Args:
        db: Open database session.
        environment_id: Target environment primary key.
        user_id: ID of the requesting user.

    Returns:
        FleetSummary with totals, per-severity counts, and asset lists.

    Raises:
        EnvironmentNotFoundError: Environment does not exist.
        AccessDeniedError: Environment belongs to another user.
    """
    _require_environment(db, environment_id, user_id)

    assets = db.exec(
        select(Asset).where(
            Asset.environment_id == environment_id, Asset.is_active == True
        )
    ).all()

    total = len(assets)
    if total == 0:
        return FleetSummary(
            total_assets=0,
            by_severity={},
            critical_assets=[],
            degrading_assets=[],
            last_updated=None,
        )

    asset_ids = [a.id for a in assets]
    latest_map = _latest_events_for_assets(db, asset_ids)

    by_severity: Dict[str, int] = {}
    critical: List[AssetStatusPublic] = []
    degrading: List[AssetStatusPublic] = []
    last_updated: Optional[datetime] = None

    for asset in assets:
        event = latest_map.get(asset.id)
        severity = event.severity if event else "unknown"
        by_severity[severity] = by_severity.get(severity, 0) + 1

        status = _build_asset_status(asset, event)
        if severity == "critical":
            critical.append(status)
        elif severity == "warning":
            degrading.append(status)

        if event and (last_updated is None or event.timestamp > last_updated):
            last_updated = event.timestamp

    return FleetSummary(
        total_assets=total,
        by_severity=by_severity,
        critical_assets=critical,
        degrading_assets=degrading,
        last_updated=last_updated,
    )


def acknowledge_event(
    db: Session, event_id: str, user_id: str
) -> HealthEvent:
    """Mark a health event as acknowledged by the given user.

    Verifies that the event belongs to an asset owned by user_id before
    updating the record.

    Args:
        db: Open database session.
        event_id: HealthEvent primary key.
        user_id: ID of the acknowledging user.

    Returns:
        The updated HealthEvent.

    Raises:
        EventNotFoundError: Event does not exist.
        AccessDeniedError: Event's asset belongs to another user's environment.
    """
    event = db.get(HealthEvent, event_id)
    if event is None:
        raise EventNotFoundError(f"Event '{event_id}' not found.")

    # Ownership check via asset → environment chain
    _require_asset(db, event.asset_id, user_id)

    event.acknowledged = True
    event.acknowledged_by = user_id
    event.acknowledged_at = datetime.utcnow()

    db.add(event)
    db.commit()
    db.refresh(event)
    return event
