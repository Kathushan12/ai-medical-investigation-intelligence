import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    # pgvector extension must exist before vector columns/indexes are used.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    index_statements = [
        "CREATE INDEX IF NOT EXISTS ix_investigations_status ON investigations (processing_status)",
        "CREATE INDEX IF NOT EXISTS ix_investigations_severity ON investigations (severity_level)",
        "CREATE INDEX IF NOT EXISTS ix_chunks_investigation_id ON evidence_chunks (investigation_id)",
        "CREATE INDEX IF NOT EXISTS ix_chunks_fts ON evidence_chunks USING gin (to_tsvector('english', content))",
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON evidence_chunks USING hnsw (embedding vector_cosine_ops)",
    ]
    with engine.begin() as connection:
        for statement in index_statements:
            try:
                connection.execute(text(statement))
            except Exception as exc:  # pragma: no cover - index availability depends on pgvector version
                logger.warning("Could not create index: %s | %s", statement, exc)
