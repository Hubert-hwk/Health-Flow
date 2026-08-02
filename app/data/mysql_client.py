"""Database client with SQLite development fallback and production URLs."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.data.models import Base


class MySQLClient:
    """Historical name retained for API compatibility; supports any SQLAlchemy URL."""

    def __init__(self) -> None:
        self.settings = get_settings()
        database_url = self.settings.database_url
        engine_kwargs: dict = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            Path("data").mkdir(parents=True, exist_ok=True)
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs.update({"pool_size": 10, "max_overflow": 20})

        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )
        self._tables_initialized = False

    def create_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)
        self._tables_initialized = True

    def drop_tables(self) -> None:
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()


_mysql_client: MySQLClient | None = None


def get_mysql_client() -> MySQLClient:
    global _mysql_client
    if _mysql_client is None:
        _mysql_client = MySQLClient()
    return _mysql_client


def get_db() -> Generator[Session, None, None]:
    client = get_mysql_client()
    if not client._tables_initialized:
        client.create_tables()
    with client.get_session() as session:
        yield session
