"""Documents, their chunks and generated summaries."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import (
    DOC_STATUSES,
    DOC_TYPES,
    SUMMARY_MODES,
    Base,
    CreatedAt,
    UpdatedAt,
    UuidPk,
    in_values,
    utcnow,
)


class Document(Base):
    """An uploaded file and its ingestion state."""

    __tablename__ = "documents"

    id: Mapped[UuidPk]
    title: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str | None] = mapped_column(Text)
    department: Mapped[str | None] = mapped_column(Text)
    semester: Mapped[int | None] = mapped_column(SmallInteger)
    academic_year: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="uploaded")
    progress: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    syllabus_outline: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    uploader_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[UpdatedAt]

    chunks: Mapped[list["ChunkMeta"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        in_values("status", DOC_STATUSES, "status_valid"),
        in_values("doc_type", DOC_TYPES, "doc_type_valid"),
        CheckConstraint("progress BETWEEN 0 AND 100", name="progress_range"),
        Index("uq_documents_sha256", "sha256", unique=True),
        Index("ix_documents_status", "status"),
        Index("ix_documents_filters", "doc_type", "department", "semester"),
        Index("ix_documents_uploader_id", "uploader_id"),
        Index(
            "ix_documents_current",
            text("uploaded_at DESC"),
            postgresql_where=text("superseded_by_id IS NULL"),
        ),
    )


class ChunkMeta(Base):
    """One chunk of a document. Its id is also the point id in the vector store."""

    __tablename__ = "chunks_meta"

    id: Mapped[UuidPk]
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_heading: Mapped[str | None] = mapped_column(Text)
    is_table: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[CreatedAt]

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (Index("uq_chunks_meta_doc_index", "document_id", "chunk_index", unique=True),)


class Summary(Base):
    """A stored summary of a document in one mode."""

    __tablename__ = "summaries"

    id: Mapped[UuidPk]
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    content: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    dropped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[CreatedAt]

    __table_args__ = (
        in_values("mode", SUMMARY_MODES, "mode_valid"),
        in_values("status", ("pending", "ready", "failed"), "status_valid"),
        Index("ix_summaries_user_created", "user_id", text("created_at DESC")),
        Index("ix_summaries_document_mode", "document_id", "mode"),
    )
