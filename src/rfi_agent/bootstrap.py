from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from rfi_agent.db import Base, engine
from rfi_agent.models import Organization


def init_db() -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    from rfi_agent.gc import gc_expired_sessions, gc_old_runs

    try:
        gc_expired_sessions()
        gc_old_runs()
    except Exception:
        pass


def ensure_local_org(db: Session) -> Organization:
    from rfi_agent.config import get_settings

    settings = get_settings()
    org = db.get(Organization, settings.default_org_id)
    if org is None:
        org = Organization(id=settings.default_org_id, name="", confirmed=False)
        db.add(org)
        db.flush()
    return org
