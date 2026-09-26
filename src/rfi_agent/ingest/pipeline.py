from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from rfi_agent.config import get_settings
from rfi_agent.gc import project_byte_usage, tombstone_chunks
from rfi_agent.grounding import sha256_bytes, sha256_text
from rfi_agent.ids import new_id
from rfi_agent.ingest.chunker import chunk_text
from rfi_agent.ingest.drive import download_drive_items, folder_changed, load_credentials, start_change_token
from rfi_agent.redact import redact
from rfi_agent.ingest.files import parse_file, write_upload
from rfi_agent.ingest.sitemap import (
    choose_sitemap,
    expand_page_locs,
    is_sitemap_index,
    looks_like_sitemap,
    origin_of,
    page_fetch_urls,
    parse_sitemap_locs,
    title_from_loc,
)
from rfi_agent.ingest.ssrf import UnsafeURL, assert_public_https
from rfi_agent.models import Chunk, Session as RFISession, Source
from rfi_agent.providers import get_embeddings

USER_AGENT = "RFIAgent/0.4"
log = logging.getLogger(__name__)


@dataclass
class PreparedDoc:
    title: str
    uri: str
    text: str
    tables: list = field(default_factory=list)
    content_type: str = "text/plain"


def classify_source_uri(uri: str) -> str:
    lowered = uri.lower()
    if "drive.google.com" in lowered or "docs.google.com" in lowered:
        return "drive"
    return "url"


def reusable_embeddings(prior_texts: list[str], new_texts: list[str], prior_vectors: list) -> list | None:
    if len(prior_texts) != len(new_texts) or prior_texts != new_texts or len(prior_vectors) != len(new_texts):
        return None
    vectors = []
    for vector in prior_vectors:
        vectors.append(vector.tolist() if hasattr(vector, "tolist") else list(vector))
    return vectors


def ingest_source(db: Session, source: Source, *, reindex: bool = False) -> None:
    if reindex and _drive_unchanged(db, source):
        return
    if reindex:
        tombstone_chunks(db, source.id)
        source.chunk_count = 0
        source.error = ""
        source.detached_at = None
    source.status = "fetching"
    db.commit()
    try:
        docs = _load_docs(db, source)
        source.status = "indexing"
        source.page_count = len(docs)
        source.fetched_at = datetime.now(timezone.utc)
        source.content_hash = sha256_text(*(d.text for d in docs))
        source.embedding_model = get_settings().embedding_model
        if docs:
            source.content_type = docs[0].content_type
        db.commit()
        duplicate = _session_duplicate(db, source)
        if duplicate is not None:
            source.status = "ready"
            source.chunk_count = 0
            source.error = ""
            source.extra = {**(source.extra or {}), "duplicate_of": duplicate.id}
            _mark_session_indexed(db, source)
            db.commit()
            return
        n_chunks = _index_docs(db, source, docs)
        source.chunk_count = n_chunks
        failures = int((source.extra or {}).get("fetch_failures") or 0)
        truncated = bool((source.extra or {}).get("sitemap_truncated"))
        if not n_chunks:
            source.status = "failed"
            source.error = "No indexable text found."
        elif truncated:
            source.status = "partial"
            cap = get_settings().sitemap_max_pages
            source.error = f"Stopped at the {cap} page cap. Raise SITEMAP_MAX_PAGES to index the rest."
        elif failures:
            source.status = "partial"
            source.error = f"{failures} page(s) failed to fetch."
        else:
            source.status = "ready"
            source.error = ""
        _mark_session_indexed(db, source)
        _store_drive_token(db, source)
        db.commit()
    except Exception as exc:  # noqa: BLE001 — persist ingest failure onto the source
        source.status = "failed"
        source.error = redact(str(exc))
        db.commit()


def detach_source(db: Session, source: Source) -> Source:
    tombstone_chunks(db, source.id)
    source.status = "detached"
    source.detached_at = datetime.now(timezone.utc)
    source.chunk_count = 0
    db.commit()
    return source


def _mark_session_indexed(db: Session, source: Source) -> None:
    if source.status not in {"ready", "partial"}:
        return
    session = db.get(RFISession, source.session_id)
    if session:
        session.intake_state = "indexed"


def _drive_unchanged(db: Session, source: Source) -> bool:
    token = str((source.extra or {}).get("drive_page_token") or "")
    if source.type != "drive" or not token:
        return False
    creds = load_credentials(db)
    if creds is None:
        return False
    try:
        changed, new_token = folder_changed(creds, token)
    except Exception:
        return False
    if changed:
        return False
    extra = dict(source.extra or {})
    extra["drive_page_token"] = new_token
    source.extra = extra
    source.status = "ready"
    source.error = ""
    db.commit()
    return True


def _store_drive_token(db: Session, source: Source) -> None:
    if source.type != "drive" or source.status not in {"ready", "partial"}:
        return
    creds = load_credentials(db)
    if creds is None:
        return
    try:
        token = start_change_token(creds)
    except Exception:
        return
    if not token:
        return
    extra = dict(source.extra or {})
    extra["drive_page_token"] = token
    source.extra = extra


def _org_embeddings(db: Session, source: Source, bodies: list[str]) -> list | None:
    if not source.content_hash:
        return None
    settings = get_settings()
    prior = (
        db.query(Source)
        .filter(
            Source.org_id == source.org_id,
            Source.id != source.id,
            Source.content_hash == source.content_hash,
            Source.embedding_model == settings.embedding_model,
            Source.status.in_(("ready", "partial")),
            Source.detached_at.is_(None),
        )
        .order_by(Source.id.desc())
        .first()
    )
    if prior is None:
        return None
    old = (
        db.query(Chunk)
        .filter(Chunk.source_id == prior.id, Chunk.org_id == source.org_id, Chunk.tombstoned.is_(False))
        .order_by(Chunk.chunk_index.asc())
        .all()
    )
    return reusable_embeddings([c.text for c in old], bodies, [c.embedding for c in old])


def _session_duplicate(db: Session, source: Source) -> Source | None:
    if not source.content_hash:
        return None
    return (
        db.query(Source)
        .filter(
            Source.project_id == source.project_id,
            Source.id != source.id,
            Source.content_hash == source.content_hash,
            Source.status.in_(("ready", "partial")),
            Source.detached_at.is_(None),
        )
        .first()
    )


def _load_docs(db: Session, source: Source) -> list[PreparedDoc]:
    settings = get_settings()
    if source.type in {"url", "sitemap"}:
        return _expand_web_url(source)
    if source.type == "upload":
        path = settings.data_dir / source.extra.get("path", "")
        data = path.read_bytes()
        parsed = parse_file(source.title or path.name, data)
        if parsed.image_only:
            raise RuntimeError("parse_failed:image_only_pdf")
        source.byte_size = len(data)
        source.content_type = parsed.content_type
        extra = dict(source.extra or {})
        extra["tables"] = parsed.tables
        source.extra = extra
        return [PreparedDoc(title=source.title or path.name, uri=str(path), text=parsed.text, tables=parsed.tables, content_type=parsed.content_type)]
    if source.type == "drive":
        creds = load_credentials(db)
        if creds is None:
            raise RuntimeError("Google Drive is not connected. Complete OAuth first.")
        items = download_drive_items(creds, source.uri)
        docs = []
        total = 0
        tables: list = []
        for name, data in items:
            total += len(data)
            parsed = parse_file(name, data)
            if parsed.image_only:
                continue
            docs.append(PreparedDoc(title=name, uri=source.uri, text=parsed.text, tables=parsed.tables, content_type=parsed.content_type))
            tables.extend(parsed.tables)
        source.byte_size = total
        extra = dict(source.extra or {})
        extra["tables"] = tables
        extra["file_count"] = len(items)
        source.extra = extra
        if not docs:
            raise RuntimeError("No indexable text found in Drive items.")
        return docs
    raise ValueError(f"Unknown source type: {source.type}")


def _expand_web_url(source: Source) -> list[PreparedDoc]:
    url = source.uri.strip()
    assert_public_https(url)
    sitemap = _canonical_sitemap(url)
    if sitemap:
        docs, failures, meta = _fetch_sitemap_live(sitemap)
        extra = dict(source.extra or {})
        extra["fetch_failures"] = failures
        extra["sitemap"] = sitemap
        extra["sitemaps"] = meta["sitemaps"]
        extra["sitemap_truncated"] = meta["truncated"]
        extra["listed_pages"] = meta["listed_pages"]
        source.extra = extra
        log.info("sitemap %s listed %s pages across %s sitemap files", sitemap, meta["listed_pages"], len(meta["sitemaps"]))
        source.byte_size = sum(len(d.text.encode("utf-8")) for d in docs)
        return docs
    doc = _fetch_url_doc(url)
    source.byte_size = len(doc.text.encode("utf-8"))
    return [doc]


def _sitemap_candidates(url: str) -> list[str]:
    """Same lookup for a site root, a page, or a sitemap file."""
    origin = origin_of(url)
    ordered = [
        _sitemap_from_robots(origin),
        urljoin(origin + "/", "sitemap.xml"),
        urljoin(origin + "/", "sitemap-pages.xml"),
        url if looks_like_sitemap(url) else None,
    ]
    seen: set[str] = set()
    candidates: list[str] = []
    for candidate in ordered:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _canonical_sitemap(url: str) -> str | None:
    origin = origin_of(url)
    fetched: list[tuple[str, bytes | None]] = []
    with httpx.Client(timeout=20.0, follow_redirects=True, max_redirects=3, headers={"User-Agent": USER_AGENT}) as client:
        for candidate in _sitemap_candidates(url):
            body: bytes | None = None
            try:
                assert_public_https(candidate)
                resp = client.get(candidate)
                if resp.status_code < 400 and (parse_sitemap_locs(resp.content, origin) or is_sitemap_index(resp.content)):
                    body = resp.content
            except (UnsafeURL, httpx.HTTPError, ET.ParseError, ValueError):
                body = None
            fetched.append((candidate, body))
    return choose_sitemap(fetched, origin)


def _sitemap_from_robots(origin: str) -> str | None:
    robots = urljoin(origin + "/", "robots.txt")
    try:
        assert_public_https(robots)
        with httpx.Client(timeout=15.0, follow_redirects=True, max_redirects=3, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(robots)
            if resp.status_code >= 400:
                return None
            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidate = line.split(":", 1)[1].strip()
                    if candidate.startswith("https://"):
                        return candidate
    except (UnsafeURL, httpx.HTTPError):
        return None
    return None


def _fetch_url_doc(url: str) -> PreparedDoc:
    with httpx.Client(timeout=30.0, follow_redirects=True, max_redirects=3, headers={"User-Agent": USER_AGENT}) as client:
        for candidate in page_fetch_urls(url):
            try:
                assert_public_https(candidate)
            except UnsafeURL:
                continue
            resp = client.get(candidate)
            if resp.status_code >= 400:
                continue
            final = str(resp.url).split("?")[0]
            if final.startswith("https://"):
                assert_public_https(final)
            ctype = resp.headers.get("content-type", "")
            name = "page.html" if "html" in ctype else "page.md"
            parsed_file = parse_file(name, resp.content)
            return PreparedDoc(
                title=title_from_loc(url),
                uri=str(resp.url),
                text=parsed_file.text,
                content_type=parsed_file.content_type,
            )
    raise RuntimeError(f"Could not fetch {url}")


def _fetch_sitemap_live(sitemap_url: str) -> tuple[list[PreparedDoc], int, dict]:
    assert_public_https(sitemap_url)
    origin = origin_of(sitemap_url)
    settings = get_settings()

    def fetch_xml(url: str) -> bytes:
        assert_public_https(url)
        with httpx.Client(timeout=60.0, follow_redirects=True, max_redirects=3, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content

    locs, meta = expand_page_locs(
        fetch_xml,
        sitemap_url,
        origin,
        max_pages=settings.sitemap_max_pages,
    )
    docs: list[PreparedDoc] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_url_doc, loc): loc for loc in locs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                docs.append(fut.result())
            except Exception:
                failures += 1
            if done % 100 == 0 or done == len(locs):
                log.info("fetched %s/%s pages (%s failed)", done, len(locs), failures)
    if not docs:
        raise RuntimeError(f"Sitemap {sitemap_url} produced no fetchable pages.")
    return docs, failures, meta


def _index_docs(db: Session, source: Source, docs: list[PreparedDoc]) -> int:
    embedder = get_embeddings()
    settings = get_settings()
    prepared: list[tuple[str, str, str, str, int]] = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc.text)):
            prepared.append((doc.title, doc.uri, f"chunk {i+1}", chunk, i))
    if not prepared:
        return 0
    bodies = [p[3] for p in prepared]
    vectors = _org_embeddings(db, source, bodies) or embedder.embed_documents(bodies)
    count = 0
    for (title, uri, locator, body, index), vector in zip(prepared, vectors, strict=True):
        chunk = Chunk(
            id=new_id("chk"),
            org_id=source.org_id,
            project_id=source.project_id,
            session_id=source.session_id or "",
            source_id=source.id,
            source_type=source.type,
            title=title,
            uri=uri,
            locator=locator,
            text=body,
            embedding=vector,
            token_count=max(1, len(body) // 4),
            chunk_index=index,
            content_hash=sha256_text(body),
            embedding_model=settings.embedding_model,
            tombstoned=False,
            extra={},
        )
        db.add(chunk)
        db.flush()
        db.execute(
            text("UPDATE chunks SET tsv = to_tsvector('english', :body) WHERE id = :id"),
            {"body": body, "id": chunk.id},
        )
        count += 1
    return count


def save_upload_source(
    db: Session,
    *,
    org_id: str,
    project_id: str,
    filename: str,
    data: bytes,
    session_id: str | None = None,
) -> Source:
    settings = get_settings()
    if len(data) > settings.max_upload_bytes:
        raise ValueError("File exceeds the 50 MB upload cap.")
    used = project_byte_usage(db, project_id)
    if used + len(data) > settings.max_project_bytes:
        raise ValueError("Project exceeds the 500 MB source cap.")
    path = write_upload(settings.data_dir, project_id, filename, data)
    source = Source(
        id=new_id("src"),
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        type="upload",
        uri=str(path),
        title=filename,
        status="pending",
        byte_size=len(data),
        content_hash=sha256_bytes(data),
        classification="CONFIDENTIAL",
        extra={"path": str(path.relative_to(settings.data_dir))},
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source
