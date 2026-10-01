"""Turning an uploaded file into a document and its chunks.

The ordering here is the design, not an implementation detail. MinIO and
Postgres cannot share a transaction, so what a crash leaves behind depends
entirely on the sequence:

    extract text -> split chunks -> INSERT (flush, NOT committed)
                                 -> PUT to MinIO
                                 -> COMMIT

The Postgres transaction stays open across the upload. If the upload fails,
rollback discards the rows and there is no compensation code to get wrong. If
the process dies before the commit, Postgres rolls back by itself and the only
residue is an orphaned object -- invisible to the app, harmless, reapable.

Committing first and uploading second inverts that: a crash in the window
leaves a visible document whose bytes were never written, and every preview
404s. The rule, applied to deletes too: **Postgres is the source of truth;
object storage may hold orphans, never the reverse.**
"""

import re
from typing import List, Optional, Tuple

from fastapi import HTTPException, status

from app.core.config import settings

# Text formats we can extract today. PDF (pypdf) and docx (python-docx) are
# each one function more and no schema change -- deliberately left until a real
# file needs them.
TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "",
}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".vtt", ".srt"}

_PARAGRAPH = re.compile(r"\n\s*\n")


def extract_text(filename: str, mime_type: Optional[str], data: bytes) -> str:
    """Decode an uploaded file to text.

    Refuses formats it cannot read rather than storing bytes it will never be
    able to chunk -- a document with no chunks can carry no citation, which
    makes it invisible to every part of the product that matters.
    """
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    readable = (mime_type or "").split(";")[0] in TEXT_MIME_TYPES or suffix in TEXT_EXTENSIONS
    if not readable:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Cannot extract text from {mime_type or suffix or 'this file'}. "
                f"Supported today: {', '.join(sorted(TEXT_EXTENSIONS))}."
            ),
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File is not valid UTF-8 text",
        )
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File contains no text to chunk",
        )
    return text


def split_into_chunks(text: str) -> List[Tuple[str, int, int]]:
    """Split text into overlapping spans, returning (content, start, end).

    Offsets are into the full extracted text and are stored on each chunk,
    which is what lets a citation point at an exact stretch and lets Gate 0
    re-check that the quoted span still says what it said.

    Paragraph boundaries are preferred so a chunk rarely cuts a sentence in
    half; the size is a ceiling, not a target. Overlap means a passage
    straddling a boundary is retrievable from either side.

    These parameters must stay stable once documents exist: chunks are
    immutable and citations point into them by index and offset, so re-cutting
    an existing document would rot every quote already taken from it.
    """
    size = settings.chunk_size_chars
    overlap = settings.chunk_overlap_chars

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        if end < length:
            # Prefer the last paragraph break inside the window; fall back to a
            # sentence end, then to a hard cut.
            window = text[start:end]
            for pattern in ("\n\n", ". ", "\n"):
                cut = window.rfind(pattern)
                # Only honour a break past the halfway mark -- otherwise a
                # document of short paragraphs produces tiny chunks.
                if cut > size // 2:
                    end = start + cut + len(pattern)
                    break

        content = text[start:end].strip()
        if content:
            chunks.append((content, start, end))

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def guard_size(data: bytes) -> None:
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {len(data)} bytes; the limit is "
                f"{settings.max_upload_bytes}. Upload is synchronous -- there "
                f"is no ingest queue to absorb a large file."
            ),
        )
