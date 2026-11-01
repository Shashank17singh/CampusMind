"""Document upload and status endpoints."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from app.audit import record_audit
from app.auth.deps import CurrentUser, DbSession, require_role
from app.auth.permissions import can_manage_document
from app.config import get_settings
from app.ingestion.errors import FileTooLargeError
from app.ingestion.service import create_document
from app.ingestion.storage import LocalStorage
from app.ingestion.tasks import ingest_document
from app.models import Document, User
from app.rag.service import get_answer_cache
from app.vectorstore import VectorStore, get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

FacultyUser = Annotated[User, Depends(require_role("faculty", "admin"))]

DocType = Literal["syllabus", "circular", "notice", "regulation", "notes", "contact_list"]


class DocumentOut(BaseModel):
    """A document as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    filename: str
    mime_type: str
    size_bytes: int
    doc_type: str | None
    department: str | None
    semester: int | None
    academic_year: str | None
    status: str
    progress: int
    error_message: str | None
    page_count: int | None
    uploader_id: uuid.UUID | None
    superseded_by_id: uuid.UUID | None
    uploaded_at: datetime


class UploadOut(DocumentOut):
    """An upload result. `duplicate_of` is set when the file was already stored."""

    duplicate_of: uuid.UUID | None = None


class DocumentPage(BaseModel):
    """One page of documents."""

    items: list[DocumentOut]
    total: int


class StatusOut(BaseModel):
    """Ingestion progress."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    progress: int
    error_message: str | None


@router.post("", response_model=UploadOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    request: Request,
    db: DbSession,
    user: Annotated[User, Depends(require_role("faculty", "admin"))],
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
    doc_type: Annotated[DocType | None, Form()] = None,
    department: Annotated[str | None, Form()] = None,
    semester: Annotated[int | None, Form(ge=1, le=8)] = None,
) -> UploadOut:
    """Store a file and queue it for ingestion. An exact duplicate is not processed again."""
    settings = get_settings()
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(f"The file is larger than the {settings.max_upload_mb} MB limit.")

    result = create_document(
        db,
        LocalStorage(settings.upload_dir),
        filename=file.filename or "upload",
        data=data,
        uploader_id=user.id,
        title=title,
        doc_type=doc_type,
        department=department,
        semester=semester,
    )
    document = result.document
    record_audit(
        db,
        "document.upload",
        actor_id=user.id,
        entity_type="document",
        entity_id=str(document.id),
        request=request,
        details={"filename": document.filename, "duplicate": result.duplicate},
    )
    db.commit()

    if not result.duplicate:
        ingest_document.delay(str(document.id))
        db.refresh(document)

    payload = UploadOut.model_validate(document)
    payload.duplicate_of = document.id if result.duplicate else None
    return payload


@router.get("", response_model=DocumentPage)
def list_documents(
    db: DbSession,
    user: CurrentUser,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    doc_type: DocType | None = None,
    department: str | None = None,
    semester: Annotated[int | None, Query(ge=1, le=8)] = None,
    mine: bool = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPage:
    """List documents, newest first."""
    filters = []
    if status_filter:
        filters.append(Document.status == status_filter)
    if doc_type:
        filters.append(Document.doc_type == doc_type)
    if department:
        filters.append(Document.department == department)
    if semester:
        filters.append(Document.semester == semester)
    if mine:
        filters.append(Document.uploader_id == user.id)

    total = db.scalar(select(func.count()).select_from(Document).where(*filters)) or 0
    rows = db.scalars(
        select(Document)
        .where(*filters)
        .order_by(Document.uploaded_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return DocumentPage(items=[DocumentOut.model_validate(row) for row in rows], total=total)


def _get_document(db: DbSession, document_id: uuid.UUID) -> Document:
    """Load a document or raise 404."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, db: DbSession, _: CurrentUser) -> Document:
    """Return one document."""
    return _get_document(db, document_id)


@router.get("/{document_id}/status", response_model=StatusOut)
def get_status(document_id: uuid.UUID, db: DbSession, _: CurrentUser) -> Document:
    """Return ingestion status and progress percentage."""
    return _get_document(db, document_id)


class DocumentUpdate(BaseModel):
    """Editable document fields."""

    title: str | None = Field(default=None, min_length=1, max_length=300)
    doc_type: DocType | None = None
    department: str | None = Field(default=None, max_length=50)
    semester: int | None = Field(default=None, ge=1, le=8)


class SupersedeRequest(BaseModel):
    """The document that replaces this one."""

    superseded_by_id: uuid.UUID


def _manageable(db: DbSession, document_id: uuid.UUID, user: User) -> Document:
    """Load a document the user may change, or raise 404 or 403."""
    document = _get_document(db, document_id)
    if not can_manage_document(user, document):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="You cannot manage this document")
    return document


def _queue_ingestion(db: DbSession, document: Document) -> None:
    """Send a document through the pipeline again."""
    document.status = "uploaded"
    document.progress = 0
    document.error_message = None
    db.commit()
    ingest_document.delay(str(document.id))
    db.refresh(document)


@router.patch("/{document_id}", response_model=DocumentOut)
def update_document(
    document_id: uuid.UUID, body: DocumentUpdate, db: DbSession, user: FacultyUser
) -> Document:
    """Edit a document. Filter fields live in the vector store too, so those re-ingest it."""
    document = _manageable(db, document_id, user)
    changes = body.model_dump(exclude_unset=True)
    needs_reingest = any(
        key in changes and getattr(document, key) != changes[key]
        for key in ("doc_type", "department", "semester", "title")
    )
    for key, value in changes.items():
        setattr(document, key, value)
    db.commit()
    if needs_reingest:
        _queue_ingestion(db, document)
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: uuid.UUID,
    request: Request,
    db: DbSession,
    user: FacultyUser,
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> None:
    """Delete a document, its chunks, its vectors and its file."""
    document = _manageable(db, document_id, user)
    store.delete_document(document.id)
    LocalStorage(get_settings().upload_dir).delete(document.storage_path)
    record_audit(
        db,
        "document.delete",
        actor_id=user.id,
        entity_type="document",
        entity_id=str(document.id),
        request=request,
        details={"title": document.title},
    )
    db.delete(document)
    db.commit()
    get_answer_cache().invalidate()


@router.post("/{document_id}/reingest", response_model=DocumentOut, status_code=202)
def reingest_document(document_id: uuid.UUID, db: DbSession, user: FacultyUser) -> Document:
    """Run a document through ingestion again."""
    document = _manageable(db, document_id, user)
    _queue_ingestion(db, document)
    return document


@router.post("/{document_id}/supersede", response_model=DocumentOut)
def supersede_document(
    document_id: uuid.UUID,
    body: SupersedeRequest,
    db: DbSession,
    user: FacultyUser,
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> Document:
    """Mark a document as replaced, which hides it from default searches."""
    document = _manageable(db, document_id, user)
    if body.superseded_by_id == document.id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="A document cannot replace itself"
        )
    _get_document(db, body.superseded_by_id)
    document.superseded_by_id = body.superseded_by_id
    db.commit()
    store.mark_superseded(document.id)
    get_answer_cache().invalidate()
    return document
