"""MySQL database client."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings
from app.data.models import Base


class MySQLClient:
    """MySQL database client."""

    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.mysql_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all tables."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        """Close database connection."""
        self.engine.dispose()


# Global instance
_mysql_client: MySQLClient | None = None


def get_mysql_client() -> MySQLClient:
    """Get MySQL client singleton."""
    global _mysql_client
    if _mysql_client is None:
        _mysql_client = MySQLClient()
    return _mysql_client


def get_db() -> Generator[Session, None, None]:
    """Get database session for FastAPI dependency injection."""
    client = get_mysql_client()
    with client.get_session() as session:
        yield session
