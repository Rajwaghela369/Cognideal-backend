"""The documents contract, both directions.

No `raw_text` anywhere: the original file is served from object storage for
preview, and the chunks ordered by `chunk_index` are the text. No ingest status
either -- a document row exists only when it is fully ingested, so there is no
half-ingested state to report.

Upload is multipart, so the create "model" is a set of Form fields on the
route rather than a body model; only the response and query shapes live here.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.enums import DocumentSourceType
from app.schemas.common import ORM, WRITE

# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class DocumentListItem(BaseModel):
    """One row of the document list on the deal page."""

    model_config = ORM

    id: uuid.UUID
    title: str
    source_type: str
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: Optional[int] = None
    # When the conversation happened, NOT when the file was dragged in. A June
    # transcript uploaded today is still a June transcript, and recency
    # ranking must use this one.
    occurred_at: datetime
    uploaded_at: datetime
    chunk_count: int = 0


class DocumentDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    deal_id: Optional[uuid.UUID] = None
    account_id: Optional[uuid.UUID] = None
    title: str
    source_type: str
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: Optional[int] = None
    # sha256 of the stored bytes. Exposed because it is the idempotency key --
    # a client can tell whether the file it holds is the one already stored.
    content_hash: str
    occurred_at: datetime
    uploaded_at: datetime
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class ChunkDetail(BaseModel):
    """One retrievable span, fetched by id to render a citation.

    This is the *citation* access path, not search: a Layer C claim stores a
    `chunk_id` plus char offsets, and the UI resolves it here to show the
    quote. Gate 0 uses the same read to re-check that the snippet still occurs
    verbatim where it was recorded.
    """

    model_config = ORM

    id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    source_type: str
    occurred_at: datetime
    chunk_index: int
    content: str
    token_count: Optional[int] = None
    # {speaker, page, char_start, char_end} -- what turns a retrieved span into
    # a clickable citation rather than an unattributed blob.
    chunk_metadata: Optional[dict] = Field(default=None, alias="metadata")


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class DocumentFilters(BaseModel):
    """Query parameters of GET /deals/{deal_id}/documents.

    Unpaginated: one deal's documents, which is tens at worst.
    """

    model_config = WRITE

    source_type: List[DocumentSourceType] = Field(default_factory=list)
    q: Optional[str] = Field(default=None, description="Matches title or filename")
    occurred_before: Optional[datetime] = None
    occurred_after: Optional[datetime] = None
    # Newest conversation first, by when it happened rather than when it was
    # uploaded.
    sort: str = "-occurred_at"
