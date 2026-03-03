"""Simulator connector for development and demo environments.

Generates realistic random-walk health data without any real hardware.
Useful for:
  - Onboarding demos (instant data, no setup)
  - Unit and integration tests
  - Development without physical access to a plant

One asset per simulator instance is designated as the "degrading asset":
its score drifts down 0.1 points per poll, simulating a slow failure.
After 5 consecutive polls below 40 (critical), it self-recovers to 65.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from .base import BaseConnector, HealthReading

# ── Constants ──────────────────────────────────────────────────────────────────

_SEVERITY_THRESHOLDS: list[tuple[float, str]] = [
    (40.0, "critical"),
    (60.0, "warning"),
    (80.0, "info"),
    (101.0, "ok"),
]

_FAILURE_MODES: dict[str, list[str]] = {
    "pump":       ["bearing_fault", "cavitation", "seal_leak", "overheating"],
    "motor":      ["overheating", "bearing_fault", "insulation_breakdown", "overload"],
    "compressor": ["overheating", "valve_failure", "bearing_fault", "vibration_high"],
    "fan":        ["bearing_fault", "blade_imbalance", "overheating"],
    "chiller":    ["refrigerant_leak", "compressor_fault", "low_flow"],
    "conveyor":   ["belt_slip", "bearing_fault", "motor_overload"],
    "equipment":  ["general_fault"],
}

_RECOVERY_THRESHOLD = 40.0
_RECOVERY_CONSECUTIVE = 5
_RECOVERY_SCORE = 65.0
_DEGRADATION_RATE = 0.1
_WALK_STDDEV = 0.8


def _severity(score: float) -> str:
    for threshold, label in _SEVERITY_THRESHOLDS:
        if score < threshold:
            return label
    return "ok"


# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class AssetDefinition:
    """One simulated asset.

    Attributes:
        external_id: Stable identifier used as Asset.external_id.
        name: Human-readable name.
        asset_type: Standardised asset type string.
        base_score: Starting health score (0–100).
    """

    external_id: str
    name: str
    asset_type: str
    base_score: float = 85.0


@dataclass
class SimulatorConfig:
    """Configuration for the SimulatorConnector.

    Attributes:
        assets: List of asset definitions to simulate.
    """

    assets: list[AssetDefinition] = field(default_factory=list)


# ── Connector ──────────────────────────────────────────────────────────────────


class SimulatorConnector(BaseConnector):
    """Generates synthetic health readings with realistic random walk.

    Health scores evolve per poll via N(0, 0.8) Gaussian noise, clamped
    to [0, 100].  One asset slowly degrades; after reaching critical for
    5 consecutive polls it recovers to 65 (simulating maintenance action).
    """

    connector_type = "simulator"

    def __init__(self, config: SimulatorConfig) -> None:
        self._config = config
        self._scores: dict[str, float] = {}
        self._consecutive_low: dict[str, int] = {}
        self._degrading_asset: Optional[str] = None
        self._connected = False

    async def connect(self) -> None:
        """Initialise scores from base values and pick a degrading asset."""
        for asset in self._config.assets:
            self._scores[asset.external_id] = asset.base_score
            self._consecutive_low[asset.external_id] = 0
        if self._config.assets:
            self._degrading_asset = self._config.assets[0].external_id
        self._connected = True

    async def disconnect(self) -> None:
        """Mark as disconnected (no real resource to release)."""
        self._connected = False

    async def discover_assets(self) -> list[HealthReading]:
        """Return initial readings for all configured assets.

        Connects automatically if not yet connected.

        Returns:
            List of HealthReading at base_score values.
        """
        if not self._connected:
            await self.connect()
        return [self._make_reading(a) for a in self._config.assets]

    async def poll(self) -> list[HealthReading]:
        """Advance the random walk and return updated readings.

        Returns:
            List of HealthReading with evolved scores.
        """
        readings: list[HealthReading] = []
        for asset in self._config.assets:
            eid = asset.external_id
            score = self._scores.get(eid, asset.base_score)

            # Random walk
            score += random.gauss(0, _WALK_STDDEV)

            # Slow degradation for the designated asset
            if eid == self._degrading_asset:
                score -= _DEGRADATION_RATE

            score = max(0.0, min(100.0, score))

            # Track consecutive critical polls
            if score < _RECOVERY_THRESHOLD:
                self._consecutive_low[eid] = self._consecutive_low.get(eid, 0) + 1
            else:
                self._consecutive_low[eid] = 0

            # Recovery after enough consecutive critical polls
            if self._consecutive_low.get(eid, 0) >= _RECOVERY_CONSECUTIVE:
                score = _RECOVERY_SCORE
                self._consecutive_low[eid] = 0

            self._scores[eid] = score
            readings.append(self._make_reading(asset, score))

        return readings

    def is_connected(self) -> bool:
        """Return True if connect() has been called."""
        return self._connected

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_reading(
        self, asset: AssetDefinition, score: Optional[float] = None
    ) -> HealthReading:
        """Build a HealthReading for one asset at the given score.

        Args:
            asset: Asset definition.
            score: Health score to use; falls back to the stored score or
                base_score.

        Returns:
            HealthReading populated with severity and failure_mode.
        """
        if score is None:
            score = self._scores.get(asset.external_id, asset.base_score)

        sev = _severity(score)
        failure_mode: Optional[str] = None
        if sev in ("warning", "critical"):
            modes = _FAILURE_MODES.get(asset.asset_type, _FAILURE_MODES["equipment"])
            failure_mode = modes[int(score) % len(modes)]

        return HealthReading(
            external_id=asset.external_id,
            name=asset.name,
            asset_type=asset.asset_type,
            source=self.connector_type,
            health_score=round(score, 2),
            severity=sev,
            failure_mode=failure_mode,
            message=f"Simulated reading: {score:.1f}/100",
        )
