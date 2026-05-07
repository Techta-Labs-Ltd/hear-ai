from datetime import datetime
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    JSON,
    Boolean,
    BigInteger,
    create_engine,
    text,
    Index,
)
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


class AiTempFile(Base):
    __tablename__ = "ai_temp_files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=True, index=True)
    run_id = Column(String, nullable=True, index=True)
    track_id = Column(String, nullable=True)
    purpose = Column(String, nullable=False)
    path = Column(String, nullable=False, unique=True)
    size_bytes = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_ai_temp_files_job_run", "job_id", "run_id"),
    )


if not settings.DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required (PostgreSQL). Set it in .env, e.g. "
        "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
    )

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    connect_args={
        "options": (
            f"-c statement_timeout={settings.DB_STATEMENT_TIMEOUT_MS} "
            f"-c application_name=hear-ai"
        ),
        "connect_timeout": 10,
    },
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


MIGRATIONS = [
    ("ai_jobs", "callback_delivered", "BOOLEAN DEFAULT FALSE"),
    ("ai_jobs", "max_tags", "INTEGER DEFAULT 8"),
    ("ai_jobs", "run_id", "VARCHAR"),
    ("ai_jobs", "track_id", "VARCHAR"),
    ("ai_jobs", "current_stage", "VARCHAR"),
    ("ai_jobs", "edited_transcript", "VARCHAR"),
]


def init_db():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    Base.metadata.create_all(bind=engine)
    _run_migrations()


def _run_migrations():
    with engine.begin() as conn:
        for table_name, column_name, column_def in MIGRATIONS:
            conn.execute(
                text(
                    f'ALTER TABLE {table_name} '
                    f'ADD COLUMN IF NOT EXISTS {column_name} {column_def}'
                )
            )
        conn.execute(
            text(
                "UPDATE ai_jobs SET run_id = gen_random_uuid()::text "
                "WHERE run_id IS NULL OR run_id = ''"
            )
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
