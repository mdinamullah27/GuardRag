"""Fixtures for database-backed tests.

Integration tests run against a dedicated ``guardrag_test`` PostgreSQL database so
the development database is never touched. When PostgreSQL is unreachable the
database tests are skipped, keeping the pure unit tests runnable anywhere.
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app import models  # noqa: F401  (register mappings on Base.metadata)
from app.config import get_settings
from app.database import Base


@pytest.fixture(scope="session")
def pg_engine():
    base = get_settings().database_url
    prefix = base.rsplit("/", 1)[0]
    try:
        admin = create_engine(prefix + "/postgres", isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'guardrag_test'")
            ).scalar()
            if not exists:
                conn.execute(text("CREATE DATABASE guardrag_test"))
        admin.dispose()

        engine = create_engine(prefix + "/guardrag_test")
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)
    except Exception as exc:  # pragma: no cover - depends on local environment
        pytest.skip(f"PostgreSQL unavailable for integration tests: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture
def db(pg_engine):
    session = Session(pg_engine)
    yield session
    try:
        session.rollback()
    finally:
        session.close()
    cleanup = Session(pg_engine)
    try:
        cleanup.execute(text("DELETE FROM documents"))
        cleanup.commit()
    finally:
        cleanup.close()
