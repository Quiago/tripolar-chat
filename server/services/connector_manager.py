"""ConnectorManager — lifecycle manager for all data connectors.

One ConnectorManager per environment_id.  Loads connector configs from the DB,
instantiates the right connector class, runs an asyncio polling loop for each,
and handles graceful shutdown.

Startup integration (called from server/main.py lifespan):
    connector_manager.load_and_start_all(db_factory)

Shutdown integration:
    connector_manager.stop_all()
"""

import asyncio
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Optional

from sqlmodel import Session, select

from server.models_assets import ConnectorConfig
from server.services import asset_service
from server.services.connectors.base import BaseConnector, HealthReading
from server.services.connectors.http_push_connector import HttpPushConnector
from server.services.connectors.opcua_connector import OpcUaConfig, OpcUaConnector
from server.services.connectors.simulator_connector import (
    AssetDefinition, SimulatorConfig, SimulatorConnector,
)

log = logging.getLogger(__name__)

# ── Connector registry ────────────────────────────────────────────────────────

# Maps connector_type string → (factory function that parses config_json and
# returns a concrete BaseConnector instance)
_DbFactory = Callable[[], "contextmanager[Session]"]


def _build_opcua(raw: dict) -> OpcUaConnector:
    return OpcUaConnector(OpcUaConfig(**raw))


def _build_simulator(raw: dict) -> SimulatorConnector:
    assets = [AssetDefinition(**a) for a in raw.get("assets", [])]
    return SimulatorConnector(SimulatorConfig(assets=assets))


def _build_http_push(_raw: dict) -> HttpPushConnector:
    return HttpPushConnector()


_CONNECTOR_FACTORIES: dict[str, Callable[[dict], BaseConnector]] = {
    "opcua":      _build_opcua,
    "simulator":  _build_simulator,
    "http_push":  _build_http_push,
}


def _instantiate_connector(cfg: ConnectorConfig) -> Optional[BaseConnector]:
    """Parse a ConnectorConfig row and return a connector instance.

    Args:
        cfg: ConnectorConfig DB row.

    Returns:
        Concrete BaseConnector, or None if the type is unknown or config is
        malformed.
    """
    factory = _CONNECTOR_FACTORIES.get(cfg.connector_type)
    if factory is None:
        log.warning("Unknown connector_type '%s' — skipping '%s'",
                    cfg.connector_type, cfg.name)
        return None
    try:
        return factory(json.loads(cfg.config_json))
    except Exception as exc:
        log.error("Failed to build connector '%s': %s", cfg.name, exc)
        return None


# ── Manager ───────────────────────────────────────────────────────────────────


class ConnectorManager:
    """Manages all connectors for one environment.

    Attributes:
        environment_id: The environment this manager owns.
    """

    def __init__(self, environment_id: str) -> None:
        self.environment_id = environment_id
        # Maps conn_id → (connector_instance, config_row)
        self._connectors: dict[str, tuple[BaseConnector, ConnectorConfig]] = {}
        # Maps conn_id → asyncio.Task
        self._tasks: dict[str, asyncio.Task] = {}

    def load_from_db(self, db: Session) -> None:
        """Instantiate all enabled connectors from DB rows.

        Args:
            db: Open database session.
        """
        configs = db.exec(
            select(ConnectorConfig).where(
                ConnectorConfig.environment_id == self.environment_id,
                ConnectorConfig.enabled == True,
            )
        ).all()

        for cfg in configs:
            connector = _instantiate_connector(cfg)
            if connector:
                self._connectors[cfg.id] = (connector, cfg)
                log.info("Loaded connector '%s' (%s) for env %s",
                         cfg.name, cfg.connector_type, self.environment_id)

    async def run_discovery(self, db: Session) -> dict:
        """Run asset discovery on all connectors and upsert results.

        Calls discover_assets() for every connector, upserts each discovered
        asset into the registry, and updates last_connected_at / last_error
        on the config row.

        Args:
            db: Open database session.

        Returns:
            Dict with keys:
                discovered (int): total assets discovered.
                updated (int):    total existing assets refreshed.
                errors (list[str]): error messages from failed connectors.
        """
        discovered = 0
        updated = 0
        errors: list[str] = []

        for conn_id, (connector, cfg) in self._connectors.items():
            try:
                if not connector.is_connected():
                    await connector.connect()

                readings = await connector.discover_assets()

                for reading in readings:
                    asset_service.upsert_asset(
                        db,
                        environment_id=self.environment_id,
                        external_id=reading.external_id,
                        name=reading.name,
                        asset_type=reading.asset_type,
                        connector_type=reading.source,
                    )
                    discovered += 1

                cfg.last_connected_at = datetime.utcnow()
                cfg.last_error = None
                db.add(cfg)
                db.commit()

            except Exception as exc:
                msg = f"Connector '{cfg.name}': {exc}"
                log.error("Discovery error — %s", msg)
                errors.append(msg)
                try:
                    cfg.last_error = str(exc)
                    db.add(cfg)
                    db.commit()
                except Exception:
                    db.rollback()

        return {"discovered": discovered, "updated": updated, "errors": errors}

    def start_polling(self, db_factory: _DbFactory) -> None:
        """Create asyncio polling tasks for all loaded connectors.

        Safe to call multiple times — already-running tasks are skipped.

        Args:
            db_factory: Zero-argument context-manager factory that yields a
                Session (e.g. ``lambda: Session(engine)``).
        """
        for conn_id, (connector, cfg) in self._connectors.items():
            if conn_id in self._tasks and not self._tasks[conn_id].done():
                continue
            task = asyncio.create_task(
                self._poll_loop(connector, cfg, db_factory),
                name=f"poll-{cfg.name[:20]}-{conn_id[:8]}",
            )
            self._tasks[conn_id] = task
            log.info("Started polling task for connector '%s' (interval %ds)",
                     cfg.name, cfg.poll_interval_seconds)

    def stop(self) -> None:
        """Cancel all polling tasks for this environment."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        self._tasks.clear()
        log.info("Stopped all polling tasks for env %s", self.environment_id)

    # ── Private ────────────────────────────────────────────────────────────────

    async def _poll_loop(
        self,
        connector: BaseConnector,
        cfg: ConnectorConfig,
        db_factory: _DbFactory,
    ) -> None:
        """Infinite loop: connect → poll → ingest → sleep → repeat.

        On error: logs, records last_error in DB, then sleeps and retries.
        CancelledError propagates immediately for clean shutdown.

        Args:
            connector: The connector instance to poll.
            cfg: The ConnectorConfig row (for interval and error tracking).
            db_factory: Context-manager factory that yields a Session.
        """
        while True:
            try:
                if not connector.is_connected():
                    await connector.connect()

                readings = await connector.poll()

                with db_factory() as db:
                    for reading in readings:
                        asset_service.ingest_event(
                            db,
                            environment_id=self.environment_id,
                            external_id=reading.external_id,
                            source=reading.source,
                            health_score=reading.health_score,
                            severity=reading.severity,
                            failure_mode=reading.failure_mode,
                            raw_value=reading.raw_value,
                            raw_unit=reading.raw_unit,
                            message=reading.message,
                            raw_payload=(
                                json.dumps(reading.raw_payload)
                                if reading.raw_payload else None
                            ),
                            timestamp=reading.timestamp,
                            asset_name=reading.name,
                            asset_type=reading.asset_type,
                        )
                    # Refresh last_connected_at on the config row
                    cfg_row = db.get(ConnectorConfig, cfg.id)
                    if cfg_row:
                        cfg_row.last_connected_at = datetime.utcnow()
                        cfg_row.last_error = None
                        db.add(cfg_row)
                    db.commit()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Poll error for connector '%s': %s", cfg.name, exc)
                try:
                    with db_factory() as db:
                        cfg_row = db.get(ConnectorConfig, cfg.id)
                        if cfg_row:
                            cfg_row.last_error = str(exc)
                            db.add(cfg_row)
                            db.commit()
                except Exception:
                    pass

            await asyncio.sleep(cfg.poll_interval_seconds)


# ── Module-level registry ─────────────────────────────────────────────────────

_managers: dict[str, ConnectorManager] = {}


def get_manager(environment_id: str) -> ConnectorManager:
    """Return (or create) the ConnectorManager for an environment.

    Args:
        environment_id: Environment primary key.

    Returns:
        Singleton ConnectorManager for that environment.
    """
    if environment_id not in _managers:
        _managers[environment_id] = ConnectorManager(environment_id)
    return _managers[environment_id]


def stop_all() -> None:
    """Cancel every polling task across all environments and clear the registry."""
    for manager in _managers.values():
        manager.stop()
    _managers.clear()


def load_and_start_all(db_factory: _DbFactory) -> None:
    """Load and start polling for every environment that has enabled connectors.

    Called once from the FastAPI lifespan after the DB is initialised.

    Args:
        db_factory: Context-manager factory that yields a Session.
    """
    from server.database import engine

    with Session(engine) as db:
        env_ids: list[str] = db.exec(
            select(ConnectorConfig.environment_id)
            .where(ConnectorConfig.enabled == True)
            .distinct()
        ).all()

    for env_id in env_ids:
        manager = get_manager(env_id)
        with Session(engine) as db:
            manager.load_from_db(db)
        manager.start_polling(db_factory)
        log.info("Connector polling started for environment %s", env_id)
