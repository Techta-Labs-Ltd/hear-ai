import os
import sqlite3

from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean, create_engine, inspect, text, Index, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class AiJob(Base):
    __tablename__ = "ai_jobs"

    id = Column(String, primary_key=True)
    run_id = Column(String, default=lambda: str(uuid.uuid4()), index=True)
    job_type = Column(String, default="pipeline")
    track_id = Column(String, nullable=True, index=True)
    status = Column(String, default="queued")
    current_stage = Column(String, nullable=True)
    input_url = Column(String)
    callback_url = Column(String)
    result_json = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    attempts = Column(Integer, default=0)
    skip_enhancement = Column(Boolean, default=False)
    skip_transcription = Column(Boolean, default=False)
    existing_transcript = Column(String, nullable=True)
    max_tags = Column(Integer, default=8)
    custom_tags = Column(JSON, nullable=True)
    edited_transcript = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    callback_delivered = Column(Boolean, default=False)


class AiTrackJob(Base):
    __tablename__ = "ai_track_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=False, index=True)
    run_id = Column(String, nullable=False, index=True)
    track_id = Column(String, nullable=False, index=True)
    job_type = Column(String, nullable=False, default="pipeline", index=True)
    status = Column(String, nullable=False, default="queued", index=True)
    current_stage = Column(String, nullable=True, index=True)
    attempts = Column(Integer, default=0)
    transcript = Column(String, nullable=True)
    moderation_json = Column(JSON, nullable=True)
    categorization_json = Column(JSON, nullable=True)
    result_json = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ai_track_jobs_job_run", "job_id", "run_id"),
    )


engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB_PATH}",
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 90,
    },
)
SessionLocal = sessionmaker(bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


MIGRATIONS = [
    ("ai_jobs", "callback_delivered", "BOOLEAN DEFAULT 0"),
    ("ai_jobs", "max_tags", "INTEGER DEFAULT 8"),
    ("ai_jobs", "run_id", "VARCHAR"),
    ("ai_jobs", "track_id", "VARCHAR"),
    ("ai_jobs", "current_stage", "VARCHAR"),
    ("ai_jobs", "edited_transcript", "VARCHAR"),
]


def init_db():
    os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _run_migrations()


def _run_migrations():
    inspector = inspect(engine)
    for table_name, column_name, column_def in MIGRATIONS:
        if not inspector.has_table(table_name):
            continue
        existing = [c["name"] for c in inspector.get_columns(table_name)]
        if column_name not in existing:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"))
            print(f"[MIGRATE] Added column {table_name}.{column_name}")
    if inspector.has_table("ai_jobs"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE ai_jobs SET run_id = lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab', abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6))) WHERE run_id IS NULL OR run_id = ''"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
