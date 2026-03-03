"""HTTP Push connector — passive receiver for webhook-based vendors.

This connector is intentionally a no-op for connect/poll/discover because
data flows in via the POST /assets/ingest endpoint rather than being polled.
It provides static payload parsers for known vendor formats.

Supported vendor formats:
  - Tractian (industrial vibration/health monitoring)
  - Generic (FactoryMind's own ingest format)
"""

from .base import BaseConnector, HealthReading


def _score_to_severity(score: float) -> str:
    if score >= 80:
        return "ok"
    if score >= 60:
        return "info"
    if score >= 40:
        return "warning"
    return "critical"


class HttpPushConnector(BaseConnector):
    """Passive connector that receives data via HTTP webhook.

    connect/disconnect are no-ops because there is no outbound connection.
    discover_assets/poll return empty lists because assets are discovered
    on the first ingest call, not by active browsing.
    """

    connector_type = "http_push"

    async def connect(self) -> None:
        """No-op — passive receiver needs no outbound connection."""

    async def disconnect(self) -> None:
        """No-op — nothing to close."""

    async def discover_assets(self) -> list[HealthReading]:
        """Return empty list — assets are discovered on first push."""
        return []

    async def poll(self) -> list[HealthReading]:
        """Return empty list — no polling for push-based connectors."""
        return []

    def is_connected(self) -> bool:
        """Always True — receiver is ready as long as the server is running."""
        return True

    @staticmethod
    def parse_tractian_payload(payload: dict) -> HealthReading:
        """Map a Tractian webhook payload to a normalised HealthReading.

        Expected Tractian fields:
            asset_id, asset_name, health_score, asset_type, vibration,
            failure_mode, message.

        Args:
            payload: Raw Tractian webhook body as a dict.

        Returns:
            HealthReading with source="tractian_api".
        """
        score = float(payload.get("health_score", 75.0))
        return HealthReading(
            external_id=str(payload["asset_id"]),
            name=payload.get("asset_name", str(payload["asset_id"])),
            asset_type=payload.get("asset_type", "equipment"),
            source="tractian_api",
            health_score=score,
            severity=_score_to_severity(score),
            failure_mode=payload.get("failure_mode"),
            raw_value=payload.get("vibration"),
            raw_unit="mm/s" if "vibration" in payload else None,
            message=payload.get("message"),
            raw_payload=payload,
        )

    @staticmethod
    def parse_generic_payload(payload: dict) -> HealthReading:
        """Map FactoryMind's own ingest format to a HealthReading.

        Required fields: external_id, health_score.
        All other fields are optional.

        Args:
            payload: Dict matching the IngestRequest schema fields.

        Returns:
            HealthReading with source taken from the payload or "http_push".
        """
        score = float(payload.get("health_score", 75.0))
        severity = payload.get("severity") or _score_to_severity(score)
        return HealthReading(
            external_id=payload["external_id"],
            name=payload.get("asset_name", payload["external_id"]),
            asset_type=payload.get("asset_type", "custom"),
            source=payload.get("source", "http_push"),
            health_score=score,
            severity=severity,
            failure_mode=payload.get("failure_mode"),
            raw_value=payload.get("raw_value"),
            raw_unit=payload.get("raw_unit"),
            message=payload.get("message"),
            raw_payload=payload,
        )
