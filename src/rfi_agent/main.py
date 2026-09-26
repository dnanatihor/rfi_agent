from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from rfi_agent.agent import answer_question, promote_to_memory
from rfi_agent.batch import create_questionnaire, run_questionnaire
from rfi_agent.bootstrap import ensure_local_org, init_db
from rfi_agent.config import get_settings
from rfi_agent.db import get_db
from rfi_agent.export import dump_json, export_json_payload, export_xlsx_bytes, serialize_answer
from rfi_agent.gc import end_session
from rfi_agent.ids import new_id
from rfi_agent.ingest.drive import (
    PKCE_COOKIE,
    begin_google_login,
    exchange_code,
    google_configured,
    load_credentials,
    oauth_start_url,
)
from rfi_agent.ingest.pipeline import classify_source_uri, detach_source, ingest_source, save_upload_source
from rfi_agent.models import (
    AgentRun,
    Answer,
    AnswerMemory,
    Organization,
    Project,
    Question,
    Questionnaire,
    Session as RFISession,
    Source,
)
from rfi_agent.redact import redact
from rfi_agent.providers import ProviderNotConfigured

app = FastAPI(title="RFI Agent", version="0.5.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


class SourceIn(BaseModel):
    uri: str
    type: str = "url"
    title: str = ""


class AskIn(BaseModel):
    question: str


class ReviewIn(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")


class AnswerEdit(BaseModel):
    answer_text: str | None = None
    short_answer: str | None = None


class QuestionnaireRunIn(BaseModel):
    confirm: bool = False
    force: bool = False


class OrgPatch(BaseModel):
    name: str | None = None
    preferences: dict[str, str] | None = None


class ProjectIn(BaseModel):
    name: str
    customer: str = ""


class ProjectPatch(BaseModel):
    name: str | None = None
    customer: str | None = None
    status: str | None = None


def _project_out(project: Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "org_id": project.org_id,
        "name": project.name,
        "customer": project.customer,
        "status": project.status,
    }


def _org(db: Session) -> Organization:
    return ensure_local_org(db)


def _org_out(org: Organization) -> dict[str, Any]:
    return {"id": org.id, "name": org.name, "preferences": org.preferences or {}, "confirmed": bool(org.confirmed)}


def _project_or_404(db: Session, project_id: str) -> Project:
    org = _org(db)
    project = db.get(Project, project_id)
    if project is None or project.org_id != org.id:
        raise HTTPException(404, "Project not found")
    return project


def _chat_out(session: RFISession) -> dict[str, Any]:
    return {
        "id": session.id,
        "project_id": session.project_id,
        "title": session.title or "New chat",
        "started_at": session.started_at.isoformat() if session.started_at else None,
    }


def _require_intake(session: RFISession) -> None:
    return None


def _session_or_404(db: Session, session_id: str) -> RFISession:
    session = db.get(RFISession, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


def _project_source_or_404(db: Session, project: Project, source_id: str) -> Source:
    source = db.get(Source, source_id)
    if source is None or source.project_id != project.id:
        raise HTTPException(404, "Source not found")
    return source


def _create_source(db: Session, project: Project, uri: str, source_type: str, title: str) -> Source:
    cleaned = uri.strip()
    kind = classify_source_uri(cleaned) if source_type in {"url", "sitemap", ""} else source_type
    source = Source(
        id=new_id("src"),
        org_id=project.org_id,
        project_id=project.id,
        session_id=None,
        type=kind,
        uri=cleaned,
        title=title or cleaned,
        status="pending",
        classification="CONFIDENTIAL",
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def _source_out(source: Source) -> dict[str, Any]:
    return {
        "id": source.id,
        "type": source.type,
        "uri": source.uri,
        "title": source.title,
        "status": source.status,
        "error": source.error,
        "page_count": source.page_count,
        "chunk_count": source.chunk_count,
        "content_hash": source.content_hash,
        "content_type": source.content_type,
        "byte_size": source.byte_size,
        "classification": source.classification,
    }


def _answer_out(answer: Answer) -> dict[str, Any]:
    return serialize_answer(answer)


def _questionnaire_out(row: Questionnaire) -> dict[str, Any]:
    return {
        "id": row.id,
        "filename": row.filename,
        "status": row.status,
        "confirmed": row.confirmed,
        "question_count": row.question_count,
        "answered_count": row.answered_count,
        "error": row.error,
        "parser_version": row.parser_version,
    }


def _question_out(question: Question, answer: Answer | None = None) -> dict[str, Any]:
    return {
        "id": question.id,
        "ordinal": question.ordinal,
        "text": question.text,
        "category": question.category,
        "constraints": question.constraints,
        "answer": _answer_out(answer) if answer else None,
    }


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    drive_connected = False
    try:
        drive_connected = load_credentials(db) is not None
    except Exception:
        drive_connected = False
    return {
        "ok": True,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "llm_configured": settings.llm_ready(),
        "embeddings_configured": settings.embeddings_ready(),
        "rerank_configured": settings.rerank_ready(),
        "google_configured": google_configured(),
        "drive_connected": drive_connected,
        "oauth_start": oauth_start_url(),
        "ovaledge_corpus": (settings.data_dir / "corpora").exists(),
    }


@app.get("/org")
def get_org(db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    db.commit()
    return _org_out(org)


@app.patch("/org")
def patch_org(payload: OrgPatch, db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    if payload.name is not None and payload.name.strip():
        org.name = payload.name.strip()[:120]
        org.confirmed = True
    if payload.preferences is not None:
        current = dict(org.preferences or {})
        for key in ("tone", "never_commit"):
            if key in payload.preferences:
                current[key] = str(payload.preferences[key] or "")[:500]
        org.preferences = current
    db.commit()
    return _org_out(org)


@app.get("/projects")
def list_projects(db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    rows = db.query(Project).filter_by(org_id=org.id).order_by(Project.created_at.asc()).all()
    db.commit()
    return {"projects": [_project_out(row) for row in rows]}


@app.post("/projects")
def create_project(payload: ProjectIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Project name is required")
    project = Project(
        id=new_id("prj"),
        org_id=org.id,
        name=name[:120],
        customer=payload.customer.strip()[:120],
        status="active",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_out(project)


@app.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    project = db.get(Project, project_id)
    if project is None or project.org_id != org.id:
        raise HTTPException(404, "Project not found")
    return _project_out(project)


@app.patch("/projects/{project_id}")
def patch_project(project_id: str, payload: ProjectPatch, db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    project = db.get(Project, project_id)
    if project is None or project.org_id != org.id:
        raise HTTPException(404, "Project not found")
    if payload.name is not None and payload.name.strip():
        project.name = payload.name.strip()[:120]
    if payload.customer is not None:
        project.customer = payload.customer.strip()[:120]
    if payload.status is not None and payload.status in {"active", "closed"}:
        project.status = payload.status
    db.commit()
    return _project_out(project)


@app.get("/projects/{project_id}/sessions")
def list_project_sessions(project_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    rows = (
        db.query(RFISession)
        .filter_by(project_id=project.id)
        .order_by(RFISession.started_at.desc())
        .all()
    )
    return {"sessions": [_chat_out(row) for row in rows]}


@app.get("/projects/{project_id}/sources")
def list_project_sources(project_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    sources = db.query(Source).filter(Source.project_id == project.id, Source.detached_at.is_(None)).all()
    return {"sources": [_source_out(source) for source in sources]}


@app.post("/projects/{project_id}/sources")
def add_project_source(
    project_id: str,
    payload: SourceIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    source = _create_source(db, project, payload.uri, payload.type, payload.title)
    background.add_task(_run_ingest, source.id, False)
    return _source_out(source)


@app.post("/projects/{project_id}/sources/upload")
def upload_project_source(
    project_id: str,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    data = file.file.read()
    try:
        source = save_upload_source(
            db,
            org_id=project.org_id,
            project_id=project.id,
            filename=file.filename or "upload.bin",
            data=data,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    background.add_task(_run_ingest, source.id, False)
    return _source_out(source)


@app.post("/projects/{project_id}/sources/{source_id}/reindex")
def reindex_project_source(
    project_id: str,
    source_id: str,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    source = _project_source_or_404(db, project, source_id)
    source.status = "pending"
    source.error = ""
    db.commit()
    background.add_task(_run_ingest, source.id, True)
    return _source_out(source)


@app.delete("/projects/{project_id}/sources/{source_id}")
def delete_project_source(project_id: str, source_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    source = _project_source_or_404(db, project, source_id)
    return _source_out(detach_source(db, source))


@app.post("/projects/{project_id}/sessions")
def create_session(project_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = _project_or_404(db, project_id)
    session = RFISession(
        id=new_id("ses"),
        project_id=project.id,
        org_id=project.org_id,
        title="",
        intake_state="open",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {**_chat_out(session), "intake_state": session.intake_state, "sources": [], "questionnaires": []}


@app.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    sources = db.query(Source).filter(Source.project_id == session.project_id, Source.detached_at.is_(None)).all()
    questionnaires = db.query(Questionnaire).filter_by(session_id=session.id).order_by(Questionnaire.created_at.desc()).all()
    return {
        "id": session.id,
        "project_id": session.project_id,
        "title": session.title or "New chat",
        "intake_state": session.intake_state,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "sources": [_source_out(s) for s in sources],
        "questionnaires": [_questionnaire_out(q) for q in questionnaires],
    }


@app.post("/sessions/{session_id}/end")
def close_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    end_session(db, session)
    return {"id": session.id, "ended_at": session.ended_at.isoformat() if session.ended_at else None}


@app.post("/sessions/{session_id}/intake/skip")
def skip_intake(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    if session.intake_state == "pending":
        session.intake_state = "skipped"
        db.commit()
    return {"id": session.id, "intake_state": session.intake_state}


@app.post("/sessions/{session_id}/sources")
def add_source(
    session_id: str,
    payload: SourceIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    uri = payload.uri.strip()
    source_type = classify_source_uri(uri) if payload.type in {"url", "sitemap", ""} else payload.type
    source = Source(
        id=new_id("src"),
        org_id=session.org_id,
        project_id=session.project_id,
        session_id=session.id,
        type=source_type,
        uri=uri,
        title=payload.title or uri,
        status="pending",
        classification="CONFIDENTIAL",
    )
    db.add(source)
    db.commit()
    source_id = source.id
    background.add_task(_run_ingest, source_id, False)
    return _source_out(source)


@app.post("/sessions/{session_id}/sources/upload")
def upload_source(
    session_id: str,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    data = file.file.read()
    try:
        source = save_upload_source(
            db,
            org_id=session.org_id,
            project_id=session.project_id,
            filename=file.filename or "upload.bin",
            data=data,
            session_id=session.id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    background.add_task(_run_ingest, source.id, False)
    return _source_out(source)


@app.get("/sessions/{session_id}/sources")
def list_sources(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    sources = db.query(Source).filter(Source.project_id == session.project_id, Source.detached_at.is_(None)).all()
    return {"sources": [_source_out(s) for s in sources]}


@app.post("/sessions/{session_id}/sources/{source_id}/reindex")
def reindex_source(
    session_id: str,
    source_id: str,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    source = db.get(Source, source_id)
    if source is None or source.project_id != session.project_id:
        raise HTTPException(404, "Source not found")
    source.status = "pending"
    source.error = ""
    db.commit()
    background.add_task(_run_ingest, source.id, True)
    return _source_out(source)


@app.delete("/sessions/{session_id}/sources/{source_id}")
def delete_source(session_id: str, source_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    source = db.get(Source, source_id)
    if source is None or source.project_id != session.project_id:
        raise HTTPException(404, "Source not found")
    return _source_out(detach_source(db, source))


@app.post("/sessions/{session_id}/ask")
def ask(session_id: str, payload: AskIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    _require_intake(session)
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    if not (session.title or "").strip():
        session.title = question[:80]
    try:
        answer = answer_question(db, session, question)
    except ProviderNotConfigured as exc:
        raise HTTPException(503, str(exc)) from exc
    return _answer_out(answer)


@app.get("/sessions/{session_id}/answers")
def list_answers(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    _session_or_404(db, session_id)
    answers = db.query(Answer).filter_by(session_id=session_id).order_by(Answer.created_at.asc()).all()
    return {"answers": [_answer_out(a) for a in answers]}


@app.patch("/sessions/{session_id}/answers/{answer_id}")
def edit_answer(session_id: str, answer_id: str, payload: AnswerEdit, db: Session = Depends(get_db)) -> dict[str, Any]:
    _session_or_404(db, session_id)
    answer = db.get(Answer, answer_id)
    if answer is None or answer.session_id != session_id:
        raise HTTPException(404, "Answer not found")
    if payload.answer_text is not None:
        answer.answer_text = payload.answer_text
        answer.edited_by_human = True
    if payload.short_answer is not None:
        answer.short_answer = payload.short_answer
        answer.edited_by_human = True
    db.commit()
    db.refresh(answer)
    return _answer_out(answer)


@app.post("/sessions/{session_id}/answers/{answer_id}/review")
def review_answer(session_id: str, answer_id: str, payload: ReviewIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    answer = db.get(Answer, answer_id)
    if answer is None or answer.session_id != session_id:
        raise HTTPException(404, "Answer not found")
    if payload.decision == "reject":
        answer.status = "rejected"
        answer.reviewer_id = "local"
        db.commit()
        return _answer_out(answer)
    answer.status = "approved"
    answer.reviewer_id = "local"
    try:
        promote_to_memory(db, session, answer)
    except ProviderNotConfigured as exc:
        raise HTTPException(503, str(exc)) from exc
    db.commit()
    return _answer_out(answer)


@app.post("/sessions/{session_id}/questionnaires")
def upload_questionnaire(
    session_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    _require_intake(session)
    data = file.file.read()
    try:
        row = create_questionnaire(db, session, file.filename or "questionnaire.csv", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _questionnaire_out(row)


@app.post("/sessions/{session_id}/questionnaires/{questionnaire_id}/run")
def start_questionnaire_run(
    session_id: str,
    questionnaire_id: str,
    payload: QuestionnaireRunIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    _require_intake(session)
    row = db.get(Questionnaire, questionnaire_id)
    if row is None or row.session_id != session_id:
        raise HTTPException(404, "Questionnaire not found")
    settings_local = get_settings()
    if row.question_count > settings_local.questionnaire_plan_threshold and not (row.confirmed or payload.confirm):
        row.status = "needs_confirm"
        db.commit()
        return _questionnaire_out(row)
    row.confirmed = True
    row.status = "running"
    db.commit()
    background.add_task(_run_questionnaire_job, session_id, questionnaire_id, payload.force)
    return _questionnaire_out(row)


@app.get("/sessions/{session_id}/questions")
def list_questions(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    _session_or_404(db, session_id)
    questions = db.query(Question).filter_by(session_id=session_id).order_by(Question.ordinal.asc()).all()
    answers = db.query(Answer).filter_by(session_id=session_id).all()
    latest: dict[str, Answer] = {}
    for answer in answers:
        if answer.question_id:
            latest[answer.question_id] = answer
    return {"questions": [_question_out(q, latest.get(q.id)) for q in questions]}


@app.get("/sessions/{session_id}/export.json")
def export_json(
    session_id: str,
    include_drafts: bool = Query(False),
    db: Session = Depends(get_db),
) -> Response:
    _session_or_404(db, session_id)
    answers = db.query(Answer).filter_by(session_id=session_id).order_by(Answer.created_at.asc()).all()
    body = dump_json(export_json_payload(answers, include_drafts=include_drafts))
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.json"'},
    )


@app.get("/sessions/{session_id}/export.xlsx")
def export_xlsx(
    session_id: str,
    include_drafts: bool = Query(False),
    db: Session = Depends(get_db),
) -> Response:
    _session_or_404(db, session_id)
    answers = db.query(Answer).filter_by(session_id=session_id).order_by(Answer.created_at.asc()).all()
    body = export_xlsx_bytes(db, answers, include_drafts=include_drafts)
    return Response(
        content=body,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.xlsx"'},
    )


@app.get("/sessions/{session_id}/runs")
def list_runs(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    rows = (
        db.query(AgentRun)
        .filter_by(session_id=session.id, org_id=session.org_id)
        .order_by(AgentRun.created_at.asc())
        .all()
    )
    return {"runs": [_run_out(row) for row in rows]}


def _run_out(run: AgentRun) -> dict[str, Any]:
    event = run.events[0] if isinstance(run.events, list) and run.events else {}
    if not isinstance(event, dict):
        event = {}
    return {
        "id": run.id,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "token_usage": run.token_usage or {},
        "latency_ms": event.get("latency_ms"),
        "model_id": event.get("model_id") or "",
        "retrieved": event.get("retrieved") or [],
        "memory_id": event.get("memory_id"),
        "memory_score": event.get("memory_score"),
        "validation": event.get("validation") or {},
        "budget_reason": event.get("budget_reason") or "",
    }


@app.get("/memory")
def list_memory(db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    rows = (
        db.query(AnswerMemory)
        .filter_by(org_id=org.id)
        .order_by(AnswerMemory.created_at.desc())
        .limit(100)
        .all()
    )
    return {
        "items": [
            {
                "id": m.id,
                "question_text": m.question_text,
                "answer_text": m.answer_text,
                "citations": m.citations,
                "valid_until": m.valid_until.isoformat() if m.valid_until else None,
                "approved_by": m.approved_by,
            }
            for m in rows
        ]
    }


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    org = _org(db)
    row = db.get(AnswerMemory, memory_id)
    if row is None or row.org_id != org.id:
        raise HTTPException(404, "Memory not found")
    db.delete(row)
    db.commit()
    return {"id": memory_id, "deleted": True}


@app.get("/oauth/google/start")
def google_start(session_id: str = "", db: Session = Depends(get_db)) -> RedirectResponse:
    if not google_configured():
        raise HTTPException(503, "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are not set in .env")
    url, verifier = begin_google_login(db, session_id)
    db.commit()
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        key=PKCE_COOKIE,
        value=verifier,
        httponly=True,
        samesite="lax",
        max_age=600,
        path="/oauth/google",
        secure=False,
    )
    return response


@app.get("/oauth/google/callback")
def google_callback(
    request: Request,
    code: str | None = None,
    state: str = "",
    error: str | None = None,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    from urllib.parse import quote

    def bounce(reason: str) -> RedirectResponse:
        target = f"{settings.frontend_origin}/?drive=error&reason={quote(reason[:180])}"
        if state:
            target += f"&session={state}"
        response = RedirectResponse(target, status_code=302)
        response.delete_cookie(PKCE_COOKIE, path="/oauth/google")
        return response

    if error:
        return bounce(error)
    if not code:
        return bounce("Google did not return an authorization code.")
    try:
        exchange_code(db, code, state, code_verifier=request.cookies.get(PKCE_COOKIE))
        db.commit()
    except Exception as exc:  # noqa: BLE001 — send the user back to the UI instead of a 500 page
        db.rollback()
        return bounce(str(exc))
    target = f"{settings.frontend_origin}/?drive=connected"
    if state:
        target += f"&session={state}"
    response = RedirectResponse(target, status_code=302)
    response.delete_cookie(PKCE_COOKIE, path="/oauth/google")
    return response


def _run_ingest(source_id: str, reindex: bool) -> None:
    from rfi_agent.db import SessionLocal

    db = SessionLocal()
    try:
        source = db.get(Source, source_id)
        if source is None:
            return
        ingest_source(db, source, reindex=reindex)
    except Exception:
        db.rollback()
    finally:
        db.close()


def _run_questionnaire_job(session_id: str, questionnaire_id: str, force: bool) -> None:
    from rfi_agent.db import SessionLocal

    db = SessionLocal()
    try:
        session = db.get(RFISession, session_id)
        row = db.get(Questionnaire, questionnaire_id)
        if session is None or row is None:
            return
        run_questionnaire(db, session, row, confirm=True, force=force)
    except ProviderNotConfigured as exc:
        if row := db.get(Questionnaire, questionnaire_id):
            row.status = "failed"
            row.error = redact(str(exc))
            db.commit()
    finally:
        db.close()
