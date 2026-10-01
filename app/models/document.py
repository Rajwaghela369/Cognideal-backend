import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import settings
from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import DocumentSourceType, document_source_type_enum


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Every unstructured input, in one table.

    Transcripts, emails, proposals, contracts and notes are discriminated by
    ``source_type`` rather than split across five tables: a RAG query wants
    "everything relevant to this deal", not a five-way union.

    **A row exists only when the document is fully ingested** -- text
    extracted, chunks written, bytes in object storage. There is no
    ``ingest_status``: anything that fails leaves no row at all, so every
    document the API can see is usable. Dropped in ``0008_document_all_or_
    nothing``.

    ``raw_text`` is gone for the same reason it was always redundant: the
    chunks, ordered by ``chunk_index``, *are* the text. The original file is
    served from object storage for preview.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # Makes re-upload idempotent, and keeps the future seeder re-runnable.
        UniqueConstraint("content_hash", name="content_hash"),
        Index("ix_documents_deal_id_occurred_at", "deal_id", "occurred_at"),
    )

    deal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=True,
    )
    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
    )
    source_type: Mapped[DocumentSourceType] = mapped_column(
        document_source_type_enum, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Object key in MinIO, not a URL: documents/{content_hash}. NOT NULL
    # because a document row only exists once its bytes are stored -- there is
    # no half-ingested state to represent.
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    byte_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # sha256 of the bytes. The idempotency key *and* the storage key, so a
    # re-upload writes identical bytes to the same object rather than
    # duplicating them.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # When the conversation actually happened -- NOT when the file was dragged
    # in. Recency ranking must use this one: a June transcript uploaded today
    # is still a June transcript.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentChunk(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A retrievable span of a document, with its embedding.

    **Chunks are immutable.** If a document is re-ingested, write a new set and
    retain the old one. Re-chunking in place silently rots every citation
    already pointing into that document, and Gate 0 starts failing claims that
    were perfectly good.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="document_chunk_index"),
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Dimension is baked in at migration time -- see settings.embedding_dim.
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.embedding_dim), nullable=True
    )
    # `metadata` is reserved on SQLAlchemy declarative classes, so the
    # attribute is renamed while the column keeps the intended name.
    # Carries {speaker, page, char_start, char_end} -- what turns a retrieved
    # chunk into a clickable citation rather than an unattributed blob.
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
