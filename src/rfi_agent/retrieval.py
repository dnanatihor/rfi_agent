from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from rfi_agent.config import get_settings
from rfi_agent.models import AnswerMemory, Chunk
from rfi_agent.providers import get_reranker


DOC_DENSE_SQL = """
            SELECT id, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM chunks
            WHERE org_id = :org_id AND project_id = :project_id {live}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 20
            """

DOC_SPARSE_SQL = """
            SELECT id
            FROM chunks
            WHERE org_id = :org_id AND project_id = :project_id {live}
              AND tsv @@ websearch_to_tsquery('english', :query)
            ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', :query)) DESC
            LIMIT 20
            """

MEMORY_SQL = """
            SELECT id, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM answer_memory
            WHERE org_id = :org_id
              AND (valid_until IS NULL OR valid_until > NOW())
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """


def rrf_merge(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, item_id in enumerate(dense_ids, start=1):
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    for rank, item_id in enumerate(sparse_ids, start=1):
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def search_documents(
    db: Session,
    *,
    org_id: str,
    project_id: str,
    query: str,
    query_vector: list[float],
    k: int = 8,
) -> list[tuple[Chunk, float | None]]:
    settings = get_settings()
    live = "AND (tombstoned = false OR tombstoned IS NULL)"
    dense_rows = db.execute(
        text(DOC_DENSE_SQL.format(live=live)),
        {"org_id": org_id, "project_id": project_id, "embedding": str(query_vector)},
    ).fetchall()
    sparse_rows = db.execute(
        text(DOC_SPARSE_SQL.format(live=live)),
        {"org_id": org_id, "project_id": project_id, "query": query},
    ).fetchall()
    dense_ids = [r[0] for r in dense_rows]
    dense_scores = {r[0]: float(r[1]) for r in dense_rows}
    sparse_ids = [r[0] for r in sparse_rows]
    fused = rrf_merge(dense_ids, sparse_ids)
    candidate_ids = [item_id for item_id, _score in fused[:20]]
    if not candidate_ids:
        return []
    chunks = (
        db.query(Chunk)
        .filter(
            Chunk.id.in_(candidate_ids),
            Chunk.org_id == org_id,
            Chunk.project_id == project_id,
            Chunk.tombstoned.is_(False),
        )
        .all()
    )
    order = {chunk_id: i for i, chunk_id in enumerate(candidate_ids)}
    chunks.sort(key=lambda c: order.get(c.id, 999))
    reranker = get_reranker()
    if reranker and chunks:
        ranked = reranker.rerank(query, [c.text for c in chunks], top_n=min(k * 2, len(chunks)))
        chunks = [chunks[i] for i in ranked if 0 <= i < len(chunks)]
    kept: list[tuple[Chunk, float | None]] = []
    sparse_set = set(sparse_ids)
    for chunk in chunks:
        score = dense_scores.get(chunk.id)
        if score is not None and score < settings.doc_min_score and chunk.id not in sparse_set:
            continue
        kept.append((chunk, score))
        if len(kept) >= k:
            break
    return kept


def search_answer_memory(
    db: Session, *, org_id: str, query_vector: list[float], limit: int = 5
) -> list[tuple[AnswerMemory, float]]:
    rows = db.execute(
        text(MEMORY_SQL),
        {"org_id": org_id, "embedding": str(query_vector), "limit": limit},
    ).fetchall()
    scored = [(r[0], float(r[1])) for r in rows]
    ids = [i for i, _s in scored]
    if not ids:
        return []
    items = db.query(AnswerMemory).filter(AnswerMemory.id.in_(ids)).all()
    by_id = {m.id: m for m in items}
    return [(by_id[i], s) for i, s in scored if i in by_id]
