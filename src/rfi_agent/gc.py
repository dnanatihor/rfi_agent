from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from rfi_agent.config import get_settings
from rfi_agent.db import SessionLocal
from rfi_agent.models import AgentRun, Chunk, Session as RFISession, Source


def tombstone_chunks(db: Session, source_id: str) -> int:
    rows = db.query(Chunk).filter(Chunk.source_id == source_id, Chunk.tombstoned.is_(False)).all()
    for chunk in rows:
        chunk.tombstoned = True
    return len(rows)


def delete_session_chunks(db: Session, session_id: str) -> int:
    count = db.query(Chunk).filter(Chunk.session_id == session_id).delete(synchronize_session=False)
    return count


def end_session(db: Session, session: RFISession) -> RFISession:
    if session.ended_at is None:
        session.ended_at = datetime.now(timezone.utc)
    db.commit()
    return session


def gc_expired_sessions() -> int:
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.session_index_retention_days)
    db = SessionLocal()
    closed = 0
    try:
        stale = (
            db.query(RFISession)
            .filter(RFISession.started_at < cutoff)
            .all()
        )
        for session in stale:
            if session.ended_at is None:
                session.ended_at = datetime.now(timezone.utc)
                closed += 1
        db.commit()
    finally:
        db.close()
    return closed


def gc_old_runs() -> int:
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.trace_retention_days)
    db = SessionLocal()
    try:
        count = db.query(AgentRun).filter(AgentRun.created_at < cutoff).delete(synchronize_session=False)
        db.commit()
        return int(count or 0)
    finally:
        db.close()


def project_byte_usage(db: Session, project_id: str) -> int:
    rows = db.query(Source.byte_size).filter(Source.project_id == project_id, Source.detached_at.is_(None)).all()
    return sum(int(r[0] or 0) for r in rows)
