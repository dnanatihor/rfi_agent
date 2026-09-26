from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from rfi_agent.config import get_settings
from rfi_agent.db import Base

_DIM = get_settings().embedding_dim


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    customer: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, default="")
    intake_state: Mapped[str] = mapped_column(String, default="open")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("sessions.id"), nullable=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    uri: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String, default="")
    content_type: Mapped[str] = mapped_column(String, default="")
    byte_size: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String, default="")
    classification: Mapped[str] = mapped_column(String, default="CONFIDENTIAL")
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    detached_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_live", "org_id", "project_id", postgresql_where=text("tombstoned = false")),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(String, default="")
    session_id: Mapped[str] = mapped_column(String, default="", index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String, default="")
    title: Mapped[str] = mapped_column(String, default="")
    uri: Mapped[str] = mapped_column(Text, default="")
    locator: Mapped[str] = mapped_column(String, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(_DIM))
    tsv = mapped_column(TSVECTOR)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String, default="")
    embedding_model: Mapped[str] = mapped_column(String, default="")
    tombstoned: Mapped[bool] = mapped_column(Boolean, default=False)
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)


class Questionnaire(Base):
    __tablename__ = "questionnaires"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, default="")
    original_path: Mapped[str] = mapped_column(Text, default="")
    parser_version: Mapped[str] = mapped_column(String, default="qparse-1")
    status: Mapped[str] = mapped_column(String, default="parsed")
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    question_count: Mapped[int] = mapped_column(Integer, default=0)
    answered_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    questionnaire_id: Mapped[str] = mapped_column(ForeignKey("questionnaires.id"), nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, default="")
    constraints: Mapped[str] = mapped_column(Text, default="")


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    question_id: Mapped[str | None] = mapped_column(String, nullable=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    origin: Mapped[str] = mapped_column(String, default="documents")
    answer_text: Mapped[str] = mapped_column(Text, default="")
    short_answer: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    gaps: Mapped[list] = mapped_column(JSONB, default=list)
    conflicts: Mapped[list] = mapped_column(JSONB, default=list)
    model_id: Mapped[str] = mapped_column(String, default="")
    run_id: Mapped[str] = mapped_column(String, default="")
    corpus_version: Mapped[str] = mapped_column(String, default="")
    reviewer_id: Mapped[str] = mapped_column(String, default="")
    edited_by_human: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnswerMemory(Base):
    __tablename__ = "answer_memory"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    embedding = mapped_column(Vector(_DIM))
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by: Mapped[str] = mapped_column(String, default="local")
    source_hash: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OAuthToken(Base):
    __tablename__ = "oauth_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    subject: Mapped[str] = mapped_column(String, default="solo")
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str] = mapped_column(Text, default="")
    token_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentRun(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="ok")
    events: Mapped[list] = mapped_column(JSONB, default=list)
    token_usage: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
