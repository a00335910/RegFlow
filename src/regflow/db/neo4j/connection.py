"""Neo4j driver. Module-level lru_cache + atexit close — same pattern as the Postgres engine."""

from __future__ import annotations

import atexit
from functools import lru_cache

from neo4j import Driver, GraphDatabase

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    s = get_settings().neo4j
    driver = GraphDatabase.driver(s.uri, auth=(s.user, s.password.get_secret_value()))
    atexit.register(driver.close)
    log.info("neo4j.driver_initialized", uri=s.uri)
    return driver


def close_driver() -> None:
    if get_driver.cache_info().currsize:
        get_driver().close()
        get_driver.cache_clear()
