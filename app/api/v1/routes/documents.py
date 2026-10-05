"""A document by id, its preview, and chunk fetch for citations.

Top-level rather than nested under the deal, unlike upload and listing. These
are reached from places that hold only an id: a preview link, and a Layer C
citation carrying a `chunk_id` with no document in hand.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models import Document, DocumentChunk
from app.schemas.v1.document import ChunkDetail, DocumentDetail
from app.services import claims as claims_service
from app.services import analysis as analysis_service
from app.services import storage

router = APIRouter(tags=["documents"])

_CHUNK_COUNT = (
    select(func.count())
    .select_from(DocumentChunk)
    .where(DocumentChunk.document_id == Document.id)
    .correlate(Document)
    .scalar_subquery()
    .label("chunk_count")
)


async def get_document_or_404(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    return document


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Any:
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


@router.get("/documents/{document_id}/preview", status_code=status.HTTP_302_FOUND)
async def preview_document(
    document: Document = Depends(get_document_or_404),
) -> RedirectResponse:
    """Redirect to a short-lived presigned URL for the original file.

    A redirect rather than a stream: proxying the bytes would make every
    preview an application request and tie up a worker for the length of the
    download. The browser fetches from object storage directly.

    The URL expires, so the UI asks for a new one per view rather than caching
    it -- a presigned link grants read access to anyone holding it.
    """
    url = storage.presigned_url(document.storage_uri, document.original_filename)
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document: Document = Depends(get_document_or_404),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Delete the row, its chunks, and the stored object.

    Postgres first, object storage second -- the same rule as upload, in
    reverse. A crash in between leaves an orphaned object, which is invisible
    and harmless; deleting the object first would leave a document row whose
    preview 404s.

    Chunks cascade, and `evidence.chunk_id` cascades with them, so the
    citation links are cleared first -- otherwise `claim_evidence` rows survive
    pointing at evidence that no longer exists. The claims themselves are kept:
    a risk whose quote was deleted is an *uncited* risk, which Gate 0 will
    flag, not a risk that never happened.

    Still true: `meetings.transcript_document_id` goes NULL, so a meeting
    silently loses its transcript link.
    """
    await claims_service.delete_links_for_document(db, document.id)
    deal_id = document.deal_id
    key = document.storage_uri
    await db.delete(document)
    await analysis_service.record_change(
        db,
        deal_id,
        "document deleted",
        table="documents",
        row_id=document.id,
        fields=("id",),
    )
    await db.commit()
    storage.delete_object(key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chunks/{chunk_id}", response_model=ChunkDetail)
async def get_chunk(chunk_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Any:
    """Fetch one chunk to render a citation.

    This is the citation path, not search. A Layer C claim stores a `chunk_id`
    and char offsets; the UI resolves it here to show the quote, and Gate 0
    reads the same row to re-check that the snippet still occurs verbatim where
    it was recorded.

    Top-level because the caller holds only the chunk id -- the document is
    what it is looking up, not something it already knows.
    """
    row = (
        await db.execute(
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title.label("document_title"),
                Document.source_type,
                Document.occurred_at,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.token_count,
                DocumentChunk.chunk_metadata.label("metadata"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id == chunk_id)
        )
    ).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found"
        )
    return row
