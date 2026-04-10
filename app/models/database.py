import os

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean, Float, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class AiJob(Base):
    __tablename__ = "ai_jobs"

    id = Column(String, primary_key=True)
    job_type = Column(String, default="pipeline")
    recording_id = Column(String, nullable=False)
    status = Column(String, default="pending")
    input_url = Column(String)
    callback_url = Column(String)
    result_json = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    attempts = Column(Integer, default=0)
    skip_enhancement = Column(Boolean, default=False)
    skip_transcription = Column(Boolean, default=False)
    existing_transcript = Column(String, nullable=True)
    custom_tags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    callback_delivered = Column(Boolean, default=False)


engine = create_engine(f"sqlite:///{settings.SQLITE_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
