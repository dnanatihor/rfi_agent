from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from rfi_agent.agent import answer_question
from rfi_agent.redact import redact
from rfi_agent.config import get_settings
from rfi_agent.grounding import corpus_version
from rfi_agent.ids import new_id
from rfi_agent.models import AgentRun, Answer, Question, Questionnaire, Session as RFISession
from rfi_agent.questionnaire import parse_questionnaire, write_questionnaire_file


def create_questionnaire(db: Session, session: RFISession, filename: str, data: bytes) -> Questionnaire:
    settings = get_settings()
    parsed = parse_questionnaire(filename, data, max_questions=settings.questionnaire_max_questions)
    path = write_questionnaire_file(settings.data_dir, session.id, filename, data)
    needs_confirm = len(parsed) > settings.questionnaire_plan_threshold
    row = Questionnaire(
        id=new_id("qn"),
        org_id=session.org_id,
        project_id=session.project_id,
        session_id=session.id,
        filename=filename,
        original_path=str(path.relative_to(settings.data_dir)),
        parser_version=settings.parser_version,
        status="needs_confirm" if needs_confirm else "parsed",
        confirmed=not needs_confirm,
        question_count=len(parsed),
        extra={"plan_threshold": settings.questionnaire_plan_threshold},
    )
    db.add(row)
    db.flush()
    for item in parsed:
        db.add(
            Question(
                id=new_id("q"),
                questionnaire_id=row.id,
                session_id=session.id,
                ordinal=item.ordinal,
                text=item.text,
                category=item.category,
                constraints=item.constraints,
            )
        )
    db.commit()
    db.refresh(row)
    return row


def run_questionnaire(
    db: Session,
    session: RFISession,
    questionnaire: Questionnaire,
    *,
    confirm: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    if questionnaire.question_count > settings.questionnaire_plan_threshold and not (questionnaire.confirmed or confirm):
        questionnaire.status = "needs_confirm"
        db.commit()
        return {
            "id": questionnaire.id,
            "status": "needs_confirm",
            "question_count": questionnaire.question_count,
            "message": f"This file has {questionnaire.question_count} questions. Confirm to start drafting.",
        }
    questionnaire.confirmed = True
    questionnaire.status = "running"
    questionnaire.error = ""
    db.commit()
    questions = (
        db.query(Question)
        .filter_by(questionnaire_id=questionnaire.id)
        .order_by(Question.ordinal.asc())
        .all()
    )
    version = corpus_version(db, session.project_id)
    started = time.monotonic()
    tokens_in = 0
    answered = 0
    try:
        for item in questions:
            if time.monotonic() - started > settings.questionnaire_wall_seconds:
                questionnaire.status = "partial"
                questionnaire.error = "Stopped at the batch wall-time budget."
                break
            if tokens_in >= settings.questionnaire_max_input_tokens:
                questionnaire.status = "partial"
                questionnaire.error = "Stopped at the batch token budget."
                break
            existing = (
                db.query(Answer)
                .filter(Answer.question_id == item.id)
                .order_by(Answer.created_at.desc())
                .first()
            )
            if existing and existing.corpus_version == version and not force:
                answered += 1
                continue
            prompt = item.text
            if item.constraints:
                prompt = f"{item.text}\n\nConstraints: {item.constraints}"
            answer = answer_question(db, session, prompt, question_id=item.id, interactive=False)
            answered += 1
            agent_run = db.get(AgentRun, answer.run_id)
            if agent_run and isinstance(agent_run.token_usage, dict):
                tokens_in += int(agent_run.token_usage.get("input") or 0)
        else:
            questionnaire.status = "complete"
        questionnaire.answered_count = answered
        db.commit()
        return {
            "id": questionnaire.id,
            "status": questionnaire.status,
            "question_count": questionnaire.question_count,
            "answered_count": questionnaire.answered_count,
            "error": questionnaire.error,
            "corpus_version": version,
        }
    except Exception as exc:  # noqa: BLE001 — persist batch failure
        questionnaire.status = "partial"
        questionnaire.error = redact(str(exc))
        questionnaire.answered_count = answered
        db.commit()
        return {
            "id": questionnaire.id,
            "status": questionnaire.status,
            "question_count": questionnaire.question_count,
            "answered_count": answered,
            "error": str(exc),
            "corpus_version": version,
        }
