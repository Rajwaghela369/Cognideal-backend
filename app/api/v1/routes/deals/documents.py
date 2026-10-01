"""Uploading and listing a deal's documents.

Upload is synchronous by design. There is no `ingest_status` to poll because a
document row exists only when the whole thing succeeded -- extract, chunk,
store. Anything that fails leaves no row, so every document the API returns is
usable. That is affordable because embeddings are not computed yet; when search
lands, the slow step moves to a batch backfill rather than into this request.
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import Deal, Document, DocumentChunk
from app.models.enums import DocumentSourceType
from app.schemas.v1.document import DocumentDetail, DocumentFilters, DocumentListItem
from app.services import activity, ingest, storage

router = APIRouter(prefix="/deals/{deal_id}/documents", tags=["documents"])

_SORT_KEYS = {"occurred_at", "uploaded_at", "title"}

_CHUNK_COUNT = (
    select(func.count())
    .select_from(DocumentChunk)
    .where(DocumentChunk.document_id == Document.id)
    .correlate(Document)
    .scalar_subquery()
    .label("chunk_count")
)


def _list_stmt(deal_id: uuid.UUID, f: DocumentFilters) -> Select:
    stmt = select(
        Document.id,
        Document.title,
        Document.source_type,
        Document.original_filename,
        Document.mime_type,
        Document.byte_size,
        Document.occurred_at,
        Document.uploaded_at,
        _CHUNK_COUNT,
    ).where(Document.deal_id == deal_id)

    if f.source_type:
        stmt = stmt.where(Document.source_type.in_(f.source_type))
    if f.occurred_after is not None:
        stmt = stmt.where(Document.occurred_at >= f.occurred_after)
    if f.occurred_before is not None:
        stmt = stmt.where(Document.occurred_at <= f.occurred_before)
    if f.q:
        pattern = f"%{f.q}%"
        stmt = stmt.where(
            or_(
                Document.title.ilike(pattern),
                Document.original_filename.ilike(pattern),
            )
        )

    descending = f.sort.startswith("-")
    column = getattr(Document, f.sort.lstrip("-"))
    order = column.desc() if descending else column.asc()
    return stmt.order_by(order.nulls_last(), Document.id)


@router.get("", response_model=List[DocumentListItem])
async def list_documents(
    filters: Annotated[DocumentFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if filters.sort.lstrip("-") not in _SORT_KEYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown sort key. Valid: {', '.join(sorted(_SORT_KEYS))}",
        )
    return (await db.execute(_list_stmt(deal.id, filters))).all()


async def _detail(db: AsyncSession, document_id: uuid.UUID) -> Any:
    row = (
        await db.execute(
            select(Document, _CHUNK_COUNT).where(Document.id == document_id)
        )
    ).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    d = row.Document
    return {
        "id": d.id,
        "deal_id": d.deal_id,
        "account_id": d.account_id,
        "title": d.title,
        "source_type": d.source_type,
        "original_filename": d.original_filename,
        "mime_type": d.mime_type,
        "byte_size": d.byte_size,
        "content_hash": d.content_hash,
        "occurred_at": d.occurred_at,
        "uploaded_at": d.uploaded_at,
        "chunk_count": row.chunk_count,
        "created_at": d.created_at,
        "updated_at": d.updated_at,
    }


@router.post("", response_model=DocumentDetail)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    source_type: DocumentSourceType = Form(...),
    occurred_at: Optional[datetime] = Form(default=None),
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Store a document: extract, chunk, persist, upload -- or nothing at all.

    Returns 201 for a new document and **200 for one already stored**. That is
    not an error case: `documents` has UNIQUE(content_hash) precisely so a
    re-upload is idempotent, which is also what keeps the future seeder
    re-runnable. Left unhandled it would surface as an IntegrityError and reach
    the client as a 500.

    The write ordering is deliberate -- see services/ingest.py. The Postgres
    transaction stays open across the object upload so a failure rolls back
    both sides with no compensation code, and a crash leaves an orphaned object
    rather than a document whose bytes were never written.
    """
    data = await file.read()
    ingest.guard_size(data)

    digest = storage.content_hash(data)
    existing = await db.scalar(
        select(Document.id).where(Document.content_hash == digest)
    )
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return await _detail(db, existing)

    text = ingest.extract_text(file.filename or "", file.content_type, data)
    chunks = ingest.split_into_chunks(text)

    document = Document(
        deal_id=deal.id,
        account_id=deal.account_id,
        source_type=source_type,
        title=title or file.filename or "Untitled",
        original_filename=file.filename,
        storage_uri=storage.object_key(digest),
        mime_type=file.content_type,
        byte_size=len(data),
        content_hash=digest,
        # When the conversation happened. Defaults to now only because the
        # caller did not say; a backfilled transcript should always pass it.
        occurred_at=occurred_at or datetime.utcnow(),
    )
    db.add(document)
    await db.flush()

    for index, (content, char_start, char_end) in enumerate(chunks):
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=content,
                token_count=len(content.split()),
                # embedding stays NULL: nothing searches by similarity yet, and
                # computing it here would put an API call inside this request.
                chunk_metadata={"char_start": char_start, "char_end": char_end},
            )
        )
    await db.flush()

    # Dated by occurred_at, not now(): a transcript from June is activity in
    # June. See services/activity.py -- using now() here would let an archive
    # import silently resolve every gone_quiet risk.
    await activity.touch_deal(db, document.deal_id, document.occurred_at)

    # Postgres is still uncommitted. If this raises, the rollback below discards
    # the document and its chunks -- nothing to clean up on either side.
    try:
        storage.put_object(document.storage_uri, data, file.content_type)
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not store the file; nothing was saved. ({exc})",
        )

    document_id = document.id
    await db.commit()

    response.status_code = status.HTTP_201_CREATED
    response.headers["Location"] = f"/documents/{document_id}"
    return await _detail(db, document_id)
