from __future__ import annotations

import hashlib
import re

from sqlalchemy.orm import Session

from rfi_agent.models import Chunk, Source

_WS = re.compile(r"\s+")


def as_string_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def as_citation_list(value: object | None) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def normalize_text(value: str) -> str:
    return _WS.sub(" ", (value or "").strip()).lower()


def quote_matches_chunk(quote: str, chunk_text: str) -> bool:
    if not quote or not quote.strip():
        return False
    return normalize_text(quote) in normalize_text(chunk_text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(*parts: str) -> str:
    joined = "\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def corpus_version(db: Session, project_id: str) -> str:
    rows = (
        db.query(Source)
        .filter(Source.project_id == project_id, Source.status.in_(("ready", "partial")), Source.detached_at.is_(None))
        .order_by(Source.id.asc())
        .all()
    )
    fingerprint = "|".join(f"{s.id}:{s.content_hash}:{s.chunk_count}" for s in rows)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def citations_resolvable(db: Session, citations: object | None) -> bool:
    citations = as_citation_list(citations)
    if not citations:
        return True
    for cite in citations:
        chunk_id = cite.get("chunk_id")
        if chunk_id:
            chunk = db.get(Chunk, chunk_id)
            if chunk is not None and not chunk.tombstoned:
                return True
        if (cite or {}).get("uri"):
            return True
    return False
