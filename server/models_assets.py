"""Asset Registry data models and API schemas.

DB tables: Environment, Asset, HealthEvent, ConnectorConfig.
Public schemas: EnvironmentPublic, AssetStatusPublic, HealthEventPublic,
                FleetSummary, IngestRequest, BatchIngestRequest,
                IngestResponse, ConnectorConfigPublic.
"""

from datetime import datetime
from typing import Dict, List, Optional
import uuid

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


# ── Database models ────────────────────────────────────────────────────────────


class Environment(SQLModel, table=True):
    """A logical grouping of assets (e.g. a factory floor, a data-centre zone).

    One user can own many environments; each environment contains assets
    discovered by one or more connectors.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str  # "Khazna DC-1", "Hotel Atlantis HVAC Zone B"
    env_type: str  # "datacenter" | "factory" | "hotel" | "building" | "custom"
    location: Optional[str] = None  # "Dubai, UAE"
    owner_user_id: str = Field(index=True)  # FK → user.id
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config_json: Optional[str] = None  # serialized dict: thresholds, connector defaults, …


class Asset(SQLModel, table=True):
    """A physical or virtual piece of equipment discovered inside an environment.

    Assets are never created manually; they are upserted by connectors on
    first contact (auto-discovery). The (environment_id, external_id) pair is
    the natural key used for idempotent upserts.
    """

    __table_args__ = (
        UniqueConstraint("environment_id", "external_id", name="uq_asset_env_external"),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    external_id: str = Field(index=True)  # vendor/protocol ID, e.g. "ns=2;i=1001", "P-101"
    environment_id: str = Field(foreign_key="environment.id", index=True)
    name: str  # human-readable, e.g. "Centrifugal Pump A"
    asset_type: str  # "pump" | "motor" | "compressor" | "crac_unit" | "ups" | …
    criticality: int = 3  # 1=low … 5=critical
    vendor: Optional[str] = None  # "Siemens", "Carrier", "APC", …
    connector_type: str  # "opcua" | "modbus" | "http_push" | "simulator" | …
    connector_meta: Optional[str] = None  # JSON: node IDs, register addresses, …
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None  # operator annotations
    is_active: bool = True


class HealthEvent(SQLModel, table=True):
    """A single health reading produced by a connector for one asset.

    Immutable once written. Acknowledged flag is the only mutable field.
    The ingest path must remain <10 ms; no business logic beyond a DB insert.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    asset_id: str = Field(foreign_key="asset.id", index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    source: str  # connector_type that generated this reading
    health_score: float  # 0.0 (dead) → 100.0 (perfect)
    severity: str  # "ok" | "info" | "warning" | "critical"
    failure_mode: Optional[str] = None  # "bearing_fault" | "overheating" | …
    raw_value: Optional[float] = None  # original sensor value
    raw_unit: Optional[str] = None  # "mm/s", "°C", "A", "%", …
    message: Optional[str] = None
    raw_payload: Optional[str] = None  # original vendor JSON, stringified
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None  # user.id
    acknowledged_at: Optional[datetime] = None


class ConnectorConfig(SQLModel, table=True):
    """Persisted configuration for a data-source connector.

    One record per data source per environment. Connectors read their
    parameters from here at startup or on hot-reload.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    environment_id: str = Field(foreign_key="environment.id", index=True)
    connector_type: str  # "opcua" | "modbus" | "http_push" | "simulator" | …
    name: str  # human label: "Plant Floor OPC-UA Server"
    enabled: bool = True
    config_json: str  # type-specific config: host, port, credentials, node filters, …
    poll_interval_seconds: int = 30
    last_connected_at: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Request schemas ────────────────────────────────────────────────────────────


class EnvironmentCreate(SQLModel):
    """Payload for POST /assets/environments."""

    name: str
    env_type: str
    location: Optional[str] = None
    config_json: Optional[str] = None


class ConnectorConfigCreate(SQLModel):
    """Payload for POST /assets/environments/{env_id}/connectors."""

    connector_type: str
    name: str
    config_json: str
    poll_interval_seconds: int = 30
    enabled: bool = True


class ConnectorConfigUpdate(SQLModel):
    """Payload for PATCH /assets/environments/{env_id}/connectors/{conn_id}.

    All fields are optional; only provided fields are updated.
    """

    name: Optional[str] = None
    enabled: Optional[bool] = None
    config_json: Optional[str] = None
    poll_interval_seconds: Optional[int] = None


class IngestRequest(SQLModel):
    """Payload for POST /assets/ingest — one health event from a connector."""

    environment_id: str
    external_id: str  # asset's vendor/protocol identifier
    source: str  # connector_type originating this reading
    health_score: float
    severity: str  # "ok" | "info" | "warning" | "critical"
    failure_mode: Optional[str] = None
    raw_value: Optional[float] = None
    raw_unit: Optional[str] = None
    message: Optional[str] = None
    raw_payload: Optional[str] = None
    timestamp: Optional[datetime] = None  # defaults to server time if omitted
    # optional asset metadata for auto-discovery (used only on first contact)
    asset_name: Optional[str] = None
    asset_type: Optional[str] = None
    vendor: Optional[str] = None


class BatchIngestRequest(SQLModel):
    """Payload for POST /assets/ingest/batch — up to 500 events in one call."""

    events: List[IngestRequest]


class AcknowledgeRequest(SQLModel):
    """Payload for POST /assets/{asset_id}/acknowledge."""

    event_id: str


# ── Response schemas ───────────────────────────────────────────────────────────


class EnvironmentPublic(SQLModel):
    """Safe public representation of an Environment."""

    id: str
    name: str
    env_type: str
    location: Optional[str]
    owner_user_id: str
    created_at: datetime


class ConnectorConfigPublic(SQLModel):
    """Public representation of a ConnectorConfig (config_json excluded)."""

    id: str
    environment_id: str
    connector_type: str
    name: str
    enabled: bool
    poll_interval_seconds: int
    last_connected_at: Optional[datetime]
    last_error: Optional[str]
    created_at: datetime


class AssetStatusPublic(SQLModel):
    """An asset combined with its most-recent health event."""

    asset_id: str
    external_id: str
    environment_id: str
    name: str
    asset_type: str
    criticality: int
    vendor: Optional[str]
    connector_type: str
    last_seen: datetime
    is_active: bool
    # latest health event — None if no events have been ingested yet
    latest_event_id: Optional[str] = None
    latest_health_score: Optional[float] = None
    latest_severity: Optional[str] = None
    latest_failure_mode: Optional[str] = None
    latest_message: Optional[str] = None
    latest_event_timestamp: Optional[datetime] = None


class HealthEventPublic(SQLModel):
    """Public representation of a HealthEvent."""

    id: str
    asset_id: str
    timestamp: datetime
    source: str
    health_score: float
    severity: str
    failure_mode: Optional[str]
    raw_value: Optional[float]
    raw_unit: Optional[str]
    message: Optional[str]
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]


class FleetSummary(SQLModel):
    """Aggregated health snapshot for all assets in an environment.

    Attributes:
        total_assets: Total number of active assets in the environment.
        by_severity: Count of assets by their latest severity level.
        critical_assets: Assets whose latest event has severity "critical".
        degrading_assets: Assets whose latest event has severity "warning".
        last_updated: Timestamp of the most-recent health event in the fleet.
    """

    total_assets: int
    by_severity: Dict[str, int]
    critical_assets: List[AssetStatusPublic]
    degrading_assets: List[AssetStatusPublic]
    last_updated: Optional[datetime]


class IngestResponse(SQLModel):
    """Result of a single or batch ingest operation."""

    ingested: int
    failed: int
    event_ids: List[str]


# ── Connector query schemas ────────────────────────────────────────────────────


class ConnectorQueryRequest(SQLModel):
    """Request body for POST /connectors/query.

    Attributes:
        connector_type: Connector type string (e.g. "opcua", "simulator").
        config: Connector-specific configuration dict (passed inline — no DB
            record required).
        environment_id: Optional environment to scope the query.  Required
            when store_readings is True.
        store_readings: When True, each reading is also written to the asset
            registry via ingest_event.  Requires environment_id.
        timeout_seconds: Max seconds to wait for the connector to respond.
            Defaults to 60 s.  OPC-UA discovery over large node trees may
            need 30–90 s; increase if you hit timeouts on first browse.
    """

    connector_type: str
    config: Dict
    environment_id: Optional[str] = None
    store_readings: bool = False
    timeout_seconds: int = 60


class ConnectorReadingPublic(SQLModel):
    """One cleaned asset reading returned by /connectors/query."""

    external_id: str
    name: str
    asset_type: str
    source: str
    health_score: float
    severity: str
    failure_mode: Optional[str] = None
    raw_value: Optional[float] = None
    raw_unit: Optional[str] = None
    message: Optional[str] = None


class ConnectorQueryResponse(SQLModel):
    """Response from POST /connectors/query.

    Attributes:
        connector_type: The connector type used.
        asset_count: Number of assets in the response.
        readings: Cleaned, normalised readings — one per discovered asset.
        stored: Number of readings written to the DB (0 if store_readings=False).
        queried_at: Server UTC timestamp of the query.
    """

    connector_type: str
    asset_count: int
    readings: List[ConnectorReadingPublic]
    stored: int
    queried_at: datetime
