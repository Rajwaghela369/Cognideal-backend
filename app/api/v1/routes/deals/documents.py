"""Uploading and listing a deal's documents.

Upload is synchronous by design. There is no `ingest_status` to poll because a
document row exists only when the whole thing succeeded -- extract, chunk,
store. Anything that fails leaves no row, so every document the API returns is
usable. That is affordable because embeddings are not computed yet; when search
lands, the slow step moves to a batch backfill rather than into this request.

PDF and docx extraction does now happen inside the request, so the claim is
narrower than it was: parsing a text layer is CPU work, bounded only by
`max_upload_bytes`. It stays synchronous because the alternative is a document
row that exists before anyone knows whether it could be read, which is the one
thing the ordering above exists to prevent. If a size limit large enough to
matter is ever set, this is where the 202-and-poll path would have to go --
`documents.status` does not exist today, deliberately.
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
from app.services import analysis as analysis_service

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


async def _meeting_for_transcript(
    db: AsyncSession, deal_id: uuid.UUID, meeting_id: Optional[uuid.UUID]
):
    """Resolve the meeting a transcript is being attached to.

    Both ids are matched, not just the meeting's: fetching by id alone would
    let a transcript be filed against another deal's meeting through this
    deal's URL.

    Refuses a meeting that already has a transcript. Replacing one silently
    would strand every fact extracted from the old document -- they keep a
    `document_id` pointing at a file the meeting no longer claims, and
    `should_extract` would then skip the new transcript because that *meeting*
    already has facts. Deleting the old document first makes that visible.
    """
    if meeting_id is None:
        return None

    from app.models import Meeting

    meeting = await db.scalar(
        select(Meeting).where(Meeting.id == meeting_id, Meeting.deal_id == deal_id)
    )
    if meeting is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found on this deal",
        )
    if meeting.transcript_document_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f'"{meeting.title}" already has a transcript. Delete that '
                f"document first -- replacing it would leave the facts "
                f"extracted from it pointing at a file this meeting no longer "
                f"claims."
            ),
        )
    return meeting


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
    meeting_id: Optional[uuid.UUID] = Form(default=None),
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

    `meeting_id` attaches the document as that meeting's transcript, which is
    what makes a meeting analysable -- extraction reads
    `meeting.transcript_document_id` and nothing else, so a transcript nobody
    points at is invisible to it. Offered here rather than as a PATCH on the
    meeting because this is where the user already is, and because it means a
    transcript is never orphaned in the first place.

    One consequence of UNIQUE(content_hash) being **global** rather than
    per-deal: a file already stored on *another* deal cannot be stored on this
    one. That used to answer 200 with the other deal's document, which told the
    caller "already stored" while this deal got nothing and the `source_type`
    they chose was silently dropped. It is now a 409 that says so.
    """
    data = await file.read()
    ingest.guard_size(data)

    if meeting_id is not None and source_type != DocumentSourceType.MEETING_TRANSCRIPT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Only a meeting_transcript can be a meeting's transcript. "
                "Upload this without a meeting, or change the source type."
            ),
        )
    meeting = await _meeting_for_transcript(db, deal.id, meeting_id)

    digest = storage.content_hash(data)
    existing_row = (
        await db.execute(
            select(Document.id, Document.deal_id, Document.title).where(
                Document.content_hash == digest
            )
        )
    ).first()
    if existing_row is not None:
        # Same deal: the idempotent re-upload this constraint exists for.
        if existing_row.deal_id == deal.id:
            if meeting is not None:
                meeting.transcript_document_id = existing_row.id
                await db.commit()
            response.status_code = status.HTTP_200_OK
            return await _detail(db, existing_row.id)
        # Different deal: there is no honest 200 here. Returning the other
        # deal's row told the caller "already stored" while this deal got
        # nothing.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"These exact bytes are already stored on another deal, as "
                f"\"{existing_row.title}\". Document identity is the file's hash "
                f"and it is global, so the same file cannot be filed twice. "
                f"Upload a distinct file, or work from the deal that has it."
            ),
        )

    text = ingest.extract_text(file.filename or "", file.content_type, data)
    chunks = ingest.chunk_document(text)

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

    for index, chunk in enumerate(chunks):
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk.content,
                token_count=len(chunk.content.split()),
                # embedding stays NULL: nothing searches by similarity yet, and
                # computing it here would put an API call inside this request.
                #
                # metadata carries the offsets and, for a transcript, who is
                # speaking -- see services/ingest.chunk_document.
                chunk_metadata=chunk.metadata,
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

    if meeting is not None:
        # Before record_change, so the detector below sees a meeting that has
        # a transcript rather than one that is still missing it.
        meeting.transcript_document_id = document.id

    if source_type == DocumentSourceType.MEETING_TRANSCRIPT:
        # `record_change`, not a bare `mark_dirty`: the touch_deal above just
        # advanced last_activity_at, and `gone_quiet` reads that column -- so an
        # upload that should clear a gone-quiet risk left it standing until the
        # nightly sweep. Tier 1 is cheap and exhaustive, so run it here like the
        # other sixteen write paths do.
        #
        # No table/row_id/fields: tier 0 re-verifies citations naming a *changed*
        # record field, and a new document changes no existing row.
        await analysis_service.record_change(
            db, deal.id, "meeting transcript uploaded"
        )

    document_id = document.id
    await db.commit()

    response.status_code = status.HTTP_201_CREATED
    response.headers["Location"] = f"/documents/{document_id}"
    return await _detail(db, document_id)
