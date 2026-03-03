"""Abstract base for all FactoryMind data connectors.

Every connector must produce HealthReading objects — a normalised,
protocol-agnostic representation of one asset's current health state.
The connector type string is used as the ``source`` field on HealthEvent rows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class HealthReading:
    """Normalised health reading produced by any connector.

    Attributes:
        external_id: Unique asset identifier within its environment
            (e.g. "ns=2;i=1001", "P-101", "rack-07").
        name: Human-readable asset name discovered from the source.
        asset_type: Standardised asset type string (e.g. "pump", "motor").
        source: Connector type that produced this reading (matches
            ConnectorConfig.connector_type).
        health_score: Current health from 0.0 (dead) to 100.0 (perfect).
        severity: One of "ok", "info", "warning", "critical".
        failure_mode: Optional label for the detected failure class.
        raw_value: Original sensor value before normalisation.
        raw_unit: Engineering unit of raw_value (e.g. "°C", "mm/s").
        message: Human-readable status description.
        raw_payload: Original vendor payload as a dict.
        timestamp: Reading timestamp; None means use server UTC time at ingest.
    """

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
    raw_payload: Optional[dict] = field(default=None)
    timestamp: Optional[datetime] = None


class BaseConnector(ABC):
    """Abstract base class all connectors must implement.

    Lifecycle:
        1. ``connect()``         — establish connection
        2. ``discover_assets()`` — browse and return all assets (once)
        3. ``poll()``            — periodic re-read (called by ConnectorManager)
        4. ``disconnect()``      — clean shutdown
    """

    connector_type: str = "base"

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection cleanly."""

    @abstractmethod
    async def discover_assets(self) -> list[HealthReading]:
        """Browse the data source and return all discoverable assets.

        Called once on connection to populate the Asset registry.
        Returns HealthReading objects with current values for each
        discovered asset.
        """

    @abstractmethod
    async def poll(self) -> list[HealthReading]:
        """Fetch current readings for all known assets.

        Called periodically by ConnectorManager.
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the connection is alive."""
