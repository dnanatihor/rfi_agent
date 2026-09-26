from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from openpyxl import Workbook
from sqlalchemy.orm import Session

from rfi_agent.grounding import as_citation_list, as_string_list
from rfi_agent.models import Answer, Question


def serialize_answer(answer: Answer) -> dict[str, Any]:
    return {
        "id": answer.id,
        "answer_id": answer.id,
        "question_id": answer.question_id,
        "status": answer.status,
        "origin": answer.origin,
        "answer_text": answer.answer_text,
        "short_answer": answer.short_answer,
        "confidence": answer.confidence,
        "citations": as_citation_list(answer.citations),
        "gaps": as_string_list(answer.gaps),
        "conflicts": as_string_list(answer.conflicts),
        "model_id": answer.model_id,
        "corpus_version": answer.corpus_version,
        "reviewer_id": answer.reviewer_id or None,
        "trace_id": answer.run_id,
        "run_id": answer.run_id,
        "question_text": answer.question_text,
        "edited_by_human": answer.edited_by_human,
        "draft": answer.status not in {"approved", "rejected"},
    }


def citation_audit(citations: object | None) -> str:
    parts = []
    for cite in as_citation_list(citations):
        title = cite.get("title") or cite.get("uri") or cite.get("chunk_id") or ""
        locator = cite.get("locator") or ""
        quote = cite.get("quote") or ""
        parts.append(f"{title} {locator}: {quote}".strip())
    return " | ".join(parts)


def export_json_payload(answers: list[Answer], *, include_drafts: bool) -> dict[str, Any]:
    rows = [serialize_answer(a) for a in answers if include_drafts or a.status in {"approved", "rejected"}]
    return {
        "include_drafts": include_drafts,
        "count": len(rows),
        "answers": rows,
    }


def export_xlsx_bytes(db: Session, answers: list[Answer], *, include_drafts: bool) -> bytes:
    wb = Workbook()
    sheet = wb.active
    sheet.title = "RFI answers"
    sheet.append(
        [
            "Question",
            "Category",
            "Constraints",
            "Status",
            "Origin",
            "Short answer",
            "Answer",
            "Confidence",
            "Citations",
            "Gaps",
            "Conflicts",
            "Draft?",
            "Question id",
            "Answer id",
        ]
    )
    ids = [a.question_id for a in answers if a.question_id]
    questions = (
        {q.id: q for q in db.query(Question).filter(Question.id.in_(ids)).all()} if ids else {}
    )
    for answer in answers:
        if not include_drafts and answer.status not in {"approved", "rejected"}:
            continue
        question = questions.get(answer.question_id or "")
        draft = "draft" if answer.status not in {"approved", "rejected"} else ""
        sheet.append(
            [
                answer.question_text,
                question.category if question else "",
                question.constraints if question else "",
                answer.status,
                answer.origin,
                answer.short_answer or "",
                answer.answer_text,
                answer.confidence,
                citation_audit(as_citation_list(answer.citations)),
                "; ".join(as_string_list(answer.gaps)),
                "; ".join(as_string_list(answer.conflicts)),
                draft,
                answer.question_id or "",
                answer.id,
            ]
        )
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def dump_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
