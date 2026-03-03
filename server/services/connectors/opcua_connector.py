"""OPC-UA connector using the asyncua library.

Connects to any OPC-UA server (Siemens, ABB, Rockwell, Schneider, Ignition …)
and auto-discovers assets by browsing the Objects node tree up to 3 levels deep.
Health scores are computed from discovered variable values using configurable
thresholds.

Typical config_json stored in ConnectorConfig.config_json:
    {
        "endpoint_url": "opc.tcp://192.168.1.10:4840",
        "username": "admin",
        "password": "secret",
        "poll_interval": 30
    }

For Ignition Maker (default port 62541):
    {
        "endpoint_url": "opc.tcp://localhost:62541/discovery"
    }
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import asyncua

from .base import BaseConnector, HealthReading

log = logging.getLogger(__name__)

# ── Asset type inference ───────────────────────────────────────────────────────

_ASSET_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"pump", "pump"),
    (r"motor", "motor"),
    (r"compress", "compressor"),
    (r"fan|blower", "fan"),
    (r"chill", "chiller"),
    (r"conveyor|belt", "conveyor"),
    (r"hvac|air.?handl", "hvac"),
    (r"ups|uninterruptible", "ups"),
    (r"generator|gen(?!eral)", "generator"),
    (r"rack|server", "server_rack"),
]


def _map_asset_type(name: str) -> str:
    """Infer an asset type string from a node's browse name.

    Args:
        name: OPC-UA browse name of the Object node.

    Returns:
        Standardised asset_type string; defaults to "equipment".
    """
    lower = name.lower()
    for pattern, asset_type in _ASSET_TYPE_PATTERNS:
        if re.search(pattern, lower):
            return asset_type
    return "equipment"


# ── Health computation ─────────────────────────────────────────────────────────


def _compute_health(
    node_values: dict, variable_mappings: dict
) -> tuple[float, str, Optional[str]]:
    """Compute (health_score, severity, failure_mode) from OPC-UA variable values.

    Rules applied in order:
    - Variables named like "health / status / condition / score" → used directly
      if the value is in [0, 100].
    - Temperature variables: above critical_max → 20, above healthy_max → 60,
      else 90.  Defaults: healthy_max=80, critical_max=95.
    - Vibration variables (mm/s): >10 → 20, >5 → 60, else 90.
    - Current/amperage: >110% of configured nominal → 40, else 90.
    - No recognisable variables: returns (75.0, "info", None).
    - Multiple variables: minimum score wins.

    Args:
        node_values: Dict mapping variable name → float value.
        variable_mappings: Per-variable config overrides (healthy_max,
            critical_max, nominal, unit).

    Returns:
        Tuple of (health_score, severity, failure_mode).
    """
    scores: list[float] = []
    failure_mode: Optional[str] = None

    for var_name, value in node_values.items():
        if not isinstance(value, (int, float)):
            continue
        lower = var_name.lower()

        # Direct health indicator
        if any(k in lower for k in ("health", "status", "condition", "score")):
            if 0 <= value <= 100:
                scores.append(float(value))
            continue

        # Temperature
        if any(k in lower for k in ("temp", "temperature")):
            mapping = variable_mappings.get(var_name, {})
            healthy_max = mapping.get("healthy_max", 80)
            critical_max = mapping.get("critical_max", 95)
            if value > critical_max:
                scores.append(20.0)
                failure_mode = "overheating"
            elif value > healthy_max:
                scores.append(60.0)
                failure_mode = failure_mode or "high_temperature"
            else:
                scores.append(90.0)
            continue

        # Vibration
        if any(k in lower for k in ("vibr", "vibration", "vib")):
            if value > 10:
                scores.append(20.0)
                failure_mode = failure_mode or "vibration_critical"
            elif value > 5:
                scores.append(60.0)
                failure_mode = failure_mode or "vibration_high"
            else:
                scores.append(90.0)
            continue

        # Current / amperage
        if any(k in lower for k in ("current", "ampere", "amp")):
            mapping = variable_mappings.get(var_name, {})
            nominal = mapping.get("nominal")
            if nominal and value > nominal * 1.1:
                scores.append(40.0)
                failure_mode = failure_mode or "overcurrent"
            else:
                scores.append(90.0)
            continue

    if not scores:
        return 75.0, "info", None

    score = min(scores)
    if score >= 80:
        severity = "ok"
    elif score >= 60:
        severity = "info"
    elif score >= 40:
        severity = "warning"
    else:
        severity = "critical"

    return round(score, 2), severity, failure_mode


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class OpcUaConfig:
    """Configuration for one OPC-UA endpoint.

    Attributes:
        endpoint_url: Full OPC-UA endpoint, e.g. "opc.tcp://192.168.1.10:4840".
        username: Optional server username.
        password: Optional server password.
        namespace_filter: Only browse nodes whose NodeId string contains
            this namespace URI (optional).
        node_id_filter: Only browse nodes whose NodeId string contains
            this substring (optional).
        poll_interval: Seconds between polls.
        variable_mappings: Per-variable threshold overrides keyed by variable
            browse name.
    """

    endpoint_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    namespace_filter: Optional[str] = None
    node_id_filter: Optional[str] = None
    poll_interval: int = 30
    variable_mappings: dict = field(default_factory=dict)
    # OPC-UA session keep-alive timeout in ms.
    # Ignition Maker caps this at 120 000 ms (2 min); setting the same value
    # here suppresses the asyncua "got N ms instead" negotiation warning.
    session_timeout_ms: int = 120_000

    @classmethod
    def from_json(cls, json_str: str) -> "OpcUaConfig":
        """Deserialise from a JSON string (as stored in ConnectorConfig.config_json).

        Args:
            json_str: JSON-encoded OpcUaConfig fields.

        Returns:
            OpcUaConfig instance.
        """
        return cls(**json.loads(json_str))


# ── Connector ──────────────────────────────────────────────────────────────────

# OPC-UA NodeClass integer values (from the OPC-UA specification)
_NODE_CLASS_OBJECT = 1
_NODE_CLASS_VARIABLE = 2


class OpcUaConnector(BaseConnector):
    """OPC-UA connector backed by the asyncua library.

    Automatically discovers assets by browsing the Objects node tree
    and reads variable children for health computation on every poll.
    """

    connector_type = "opcua"

    def __init__(self, config: OpcUaConfig) -> None:
        self._config = config
        self._client: Optional[asyncua.Client] = None
        # Cache of discovered nodes: list of {nodeid, name, asset_type}
        self._discovered_nodes: list[dict] = []

    async def connect(self) -> None:
        """Connect to the OPC-UA server, setting credentials if configured.

        Args: none

        Raises:
            asyncua.ua.UaError: If the connection cannot be established.
        """
        self._client = asyncua.Client(
            url=self._config.endpoint_url,
            session_timeout=self._config.session_timeout_ms,
        )
        if self._config.username and self._config.password:
            self._client.set_user(self._config.username)
            self._client.set_password(self._config.password)
        await self._client.connect()
        log.info("OPC-UA connected: %s", self._config.endpoint_url)

    async def disconnect(self) -> None:
        """Disconnect from the OPC-UA server, suppressing any errors."""
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def discover_assets(self) -> list[HealthReading]:
        """Browse the Objects node recursively and return discovered assets.

        Browses up to 3 levels deep starting from the Objects node (i=85).
        Each Object node becomes one asset; its Variable children are read
        to compute the initial health score.

        Returns:
            List of HealthReading objects, one per discovered Object node.
        """
        if not self._client:
            await self.connect()

        self._discovered_nodes = []
        readings: list[HealthReading] = []
        objects_node = self._client.get_objects_node()
        await self._browse_recursive(objects_node, depth=0, max_depth=3, readings=readings)
        return readings

    async def poll(self) -> list[HealthReading]:
        """Re-read all previously discovered nodes and return fresh readings.

        Falls back to discover_assets() if no nodes have been discovered yet.

        Returns:
            List of HealthReading objects with current values.
        """
        if not self._client or not self._discovered_nodes:
            return await self.discover_assets()

        readings: list[HealthReading] = []
        for node_info in self._discovered_nodes:
            try:
                node = self._client.get_node(node_info["nodeid"])
                node_values = await self._read_child_variables(node)
                score, severity, failure_mode = _compute_health(
                    node_values, self._config.variable_mappings
                )
                readings.append(HealthReading(
                    external_id=str(node_info["nodeid"]),
                    name=node_info["name"],
                    asset_type=node_info["asset_type"],
                    source=self.connector_type,
                    health_score=score,
                    severity=severity,
                    failure_mode=failure_mode,
                    message=f"Polled via OPC-UA: {node_info['name']}",
                    raw_payload={"node_values": node_values},
                ))
            except Exception as exc:
                log.warning("Failed to poll node '%s': %s", node_info.get("name"), exc)

        return readings

    def is_connected(self) -> bool:
        """Return True if the underlying asyncua connection is alive."""
        if self._client is None:
            return False
        try:
            return self._client.uaclient.is_connected()
        except Exception:
            return False

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _browse_recursive(
        self,
        node,
        depth: int,
        max_depth: int,
        readings: list[HealthReading],
    ) -> None:
        """Recursively browse child Object nodes up to max_depth.

        Args:
            node: Current asyncua node to browse.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth allowed.
            readings: Accumulator list for discovered HealthReadings.
        """
        if depth > max_depth:
            return

        try:
            children = await node.get_children()
        except Exception as exc:
            log.debug("Cannot browse node at depth %d: %s", depth, exc)
            return

        for child in children:
            try:
                node_class = await child.read_node_class()
                if node_class.value != _NODE_CLASS_OBJECT:
                    continue

                browse_name = await child.read_browse_name()
                name = browse_name.Name
                node_id_str = str(child.nodeid)

                # Apply optional namespace filter
                if (
                    self._config.namespace_filter
                    and self._config.namespace_filter not in node_id_str
                ):
                    await self._browse_recursive(child, depth + 1, max_depth, readings)
                    continue

                asset_type = _map_asset_type(name)
                node_values = await self._read_child_variables(child)
                score, severity, failure_mode = _compute_health(
                    node_values, self._config.variable_mappings
                )

                # Use the first available value as the representative raw reading
                raw_value: Optional[float] = None
                raw_unit: Optional[str] = None
                if node_values:
                    first_key = next(iter(node_values))
                    raw_value = node_values[first_key]
                    raw_unit = self._config.variable_mappings.get(
                        first_key, {}
                    ).get("unit")

                self._discovered_nodes.append({
                    "nodeid": child.nodeid,
                    "name": name,
                    "asset_type": asset_type,
                })

                readings.append(HealthReading(
                    external_id=node_id_str,
                    name=name,
                    asset_type=asset_type,
                    source=self.connector_type,
                    health_score=score,
                    severity=severity,
                    failure_mode=failure_mode,
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    message=f"Discovered via OPC-UA: {name}",
                    raw_payload={"node_values": node_values},
                ))

                # Recurse into child objects
                await self._browse_recursive(child, depth + 1, max_depth, readings)

            except Exception as exc:
                log.debug("Skipping node at depth %d: %s", depth, exc)
                continue

    async def _read_child_variables(self, node) -> dict:
        """Read all direct Variable children and return {browse_name: float_value}.

        Non-numeric values are silently skipped.

        Args:
            node: asyncua node whose Variable children to read.

        Returns:
            Dict mapping variable browse name → float value.
        """
        values: dict[str, float] = {}
        try:
            children = await node.get_children()
            for child in children:
                try:
                    node_class = await child.read_node_class()
                    if node_class.value != _NODE_CLASS_VARIABLE:
                        continue
                    browse_name = await child.read_browse_name()
                    value = await child.read_value()
                    if isinstance(value, (int, float, bool)):
                        values[browse_name.Name] = float(value)
                except Exception:
                    continue
        except Exception:
            pass
        return values
