"""Turning an uploaded file into a document and its chunks.

The ordering here is the design, not an implementation detail. Object storage and
Postgres cannot share a transaction, so what a crash leaves behind depends
entirely on the sequence:

    extract text -> split chunks -> INSERT (flush, NOT committed)
                                 -> PUT to object storage
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

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status

from app.core.config import settings

# Formats we can extract today, split by *how* the text comes out: decoding
# bytes, or asking a library to walk a container. The sets are named for what
# they admit rather than lumped together as one "readable" list, because the
# empty mime type belongs in the first group and must never silently admit the
# second -- a browser that sends no Content-Type for a .pdf should match on the
# suffix, not fall through to a UTF-8 decode of a binary file.
TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "",
}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".vtt", ".srt"}

#: Containers holding their own text layer, read by pypdf / python-docx.
PDF_MIME_TYPES = {"application/pdf"}
DOCX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}

#: Every extension a caller may upload, for the 415 message. Sorted at use.
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS

#: Legacy Word. Refused by name rather than by falling through to the generic
#: 415, because the generic message would list `.docx` one word later and leave
#: the caller to guess what the difference is. `.doc` is a binary OLE compound
#: file; python-docx opens zipped XML and cannot read it, and there is no second
#: library here that can. "Open it in Word and Save As .docx" is the fix, so the
#: error says that.
LEGACY_DOC_EXTENSIONS = {".doc"}
LEGACY_DOC_MIME_TYPES = {"application/msword"}

_PARAGRAPH = re.compile(r"\n\s*\n")

# A transcript turn: a line beginning with a name and a colon.
#
# Deliberately narrow. One to four capitalised words, so "Priya Raman:" and
# "Dana Whitfield:" match while "Policy:" (one lowercase-ish word is still
# allowed, but a sentence like "Sure. We failed" has no colon) and a bare URL
# do not. The cost of a false positive here is a fabricated attendee, which is
# worse than a missed one -- a missing attendee is visible, an invented
# colleague is not.
_SPEAKER_TURN = re.compile(
    r"^(?P<speaker>[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}):[ \t]",
    re.MULTILINE,
)


def _pdf_text(data: bytes) -> str:
    """Extract a PDF's text layer, one page per paragraph break.

    Pages are joined with a blank line so `_PARAGRAPH` treats a page boundary
    as a paragraph boundary. Without it the last sentence of one page and the
    first of the next become one run of text, and a chunk can quote across a
    boundary that does not exist in the document.

    **Reading order is the library's, and it is not always the page's.** pypdf
    emits text in the order the content stream draws it, which for a
    multi-column page or a table can interleave columns. That matters more here
    than it would elsewhere: Gate 0 verifies a citation by searching for the
    quoted span inside the chunk text, so the extracted text *is* the evidence.
    A scrambled extraction does not fail loudly -- it produces citations that
    verify against text no human would recognise as a quotation. Single-column
    documents are safe; anything laid out in columns should be treated as
    unreviewed until it has been looked at.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        # Encrypted, truncated or not actually a PDF. A 415 rather than a 500:
        # the caller's file is the problem and the caller can act on it.
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Could not read this PDF (%s). It may be encrypted or damaged."
            % type(exc).__name__,
        )
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _docx_text(data: bytes) -> str:
    """Extract a .docx's paragraphs and table cells.

    Tables are included because a requirements matrix or a pricing grid is
    exactly the kind of thing a deal document carries, and `document.paragraphs`
    alone silently omits every one of them -- a document that looked extracted
    but had lost its substance.

    Each cell becomes its own line and each row is tab-joined, which is a lossy
    rendering of a table and deliberately so: the chunker and Gate 0 work on
    flat text with offsets, and there is nowhere for structure to live.
    """
    import zipfile

    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = docx.Document(io.BytesIO(data))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        # `BadZipFile` is the one that actually fires for a legacy .doc renamed
        # to .docx, and it is not a `PackageNotFoundError` -- python-docx lets
        # zipfile's own error escape. Caught by test, not by reading the docs.
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Could not read this .docx (%s). It may be damaged, or be a "
            "legacy .doc renamed." % type(exc).__name__,
        )

    blocks = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append("\t".join(cells))
    return "\n\n".join(blocks)


def extract_text(filename: str, mime_type: Optional[str], data: bytes) -> str:
    """Decode an uploaded file to text.

    Refuses formats it cannot read rather than storing bytes it will never be
    able to chunk -- a document with no chunks can carry no citation, which
    makes it invisible to every part of the product that matters.

    Matching is on mime type **or** suffix, not both, in that order of
    preference per format. Browsers disagree about the docx mime type and some
    send nothing at all, so a `.docx` that arrives as
    `application/octet-stream` is still read; conversely a file named
    `notes.txt` that is really a PDF fails at the decode below, which is the
    right place for it to fail.
    """
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    mime = (mime_type or "").split(";")[0].strip().lower()

    # Before the generic refusal: `.doc` is a format we know about and cannot
    # read, which is a different message from one we do not recognise at all.
    if suffix in LEGACY_DOC_EXTENSIONS or mime in LEGACY_DOC_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Legacy .doc files cannot be read. Open it in Word and save as "
                ".docx, then upload that."
            ),
        )

    if mime in PDF_MIME_TYPES or suffix in PDF_EXTENSIONS:
        text = _pdf_text(data)
        if not text.strip():
            # The scanned-page case, and the whole reason this guard is a 415
            # rather than the 422 an empty .txt gets. `application/pdf` is a
            # supported media type; *this* PDF carries no text layer, only
            # pixels. Reading it needs OCR, which this project deliberately
            # does not do (see requirements.txt), so the honest answer is "wrong
            # kind of file" and the fix is on the caller's side. Letting it
            # through would store a document with zero chunks -- present in
            # every list, citable by nothing.
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    "This PDF has no text layer -- it looks scanned. Text "
                    "recognition is not supported; upload a PDF with "
                    "selectable text."
                ),
            )
        return text

    if mime in DOCX_MIME_TYPES or suffix in DOCX_EXTENSIONS:
        text = _docx_text(data)
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File contains no text to chunk",
            )
        return text

    if mime not in TEXT_MIME_TYPES and suffix not in TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Cannot extract text from {mime_type or suffix or 'this file'}. "
                f"Supported today: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
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


def parse_speaker_turns(text: str) -> List[Tuple[str, int, int]]:
    """Find the speaker turns in a transcript: (speaker, start, end).

    ``start`` is the offset of the label itself, so the turn includes
    "Priya Raman: " -- a span quoted from inside a turn is then always
    attributable without re-parsing, and the label is part of the text a
    citation can point at.

    Returns an empty list for anything that is not speaker-labelled, which is
    what makes this safe to call on every upload: an email or a contract falls
    straight through to the paragraph-based chunking that has always run.
    """
    matches = list(_SPEAKER_TURN.finditer(text))
    if len(matches) < 2:
        # One match is far more likely to be a prose line that happens to start
        # with a capitalised word and a colon than a one-speaker transcript.
        return []

    turns: List[Tuple[str, int, int]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        turns.append((match.group("speaker"), match.start(), end))
    return turns


def speakers_in(turns: List[Tuple[str, int, int]], start: int, end: int) -> List[str]:
    """Which speakers a span covers, in order of first appearance."""
    seen: List[str] = []
    for speaker, turn_start, turn_end in turns:
        if turn_start < end and turn_end > start and speaker not in seen:
            seen.append(speaker)
    return seen


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

    # Transcripts get cut on speaker turns where possible. A chunk that ends
    # mid-sentence is merely untidy; one that ends mid-turn splits a statement
    # from the person who made it, and a span quoted across that boundary is
    # attributable to nobody.
    turns = parse_speaker_turns(text)
    boundaries = [turn_start for _, turn_start, _ in turns]

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        if end < length:
            turn_cut = _last_boundary_before(boundaries, start, end, size)
            if turn_cut is not None:
                end = turn_cut
            else:
                # Prefer the last paragraph break inside the window; fall back
                # to a sentence end, then to a hard cut.
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

        # Overlap, then snap back to a turn start. Ending on a boundary is only
        # half the job: a chunk whose *start* falls mid-turn opens with an
        # unattributed fragment, and a span quoted from it has no speaker. For a
        # transcript the overlap therefore means "re-include the last whole
        # turn or two", which is a little more than `chunk_overlap_chars` and
        # the right trade -- every statement in every chunk carries its label.
        next_start = max(end - overlap, start + 1)
        snapped = _first_boundary_at_or_before(boundaries, next_start)
        if snapped is not None and snapped > start:
            next_start = snapped
        start = next_start

    return chunks


def _first_boundary_at_or_before(boundaries: List[int], position: int) -> Optional[int]:
    """The latest turn start at or before ``position``, if any."""
    candidates = [b for b in boundaries if b <= position]
    return max(candidates) if candidates else None


def _last_boundary_before(
    boundaries: List[int], start: int, end: int, size: int
) -> Optional[int]:
    """The latest speaker-turn start inside (start, end], past the halfway mark.

    Halfway for the same reason the paragraph rule uses it: a transcript of
    one-line exchanges would otherwise produce a chunk per turn, and a chunk
    holding a single sentence carries no context for the extractor to read.
    """
    candidates = [b for b in boundaries if start + size // 2 < b <= end]
    return max(candidates) if candidates else None


@dataclass
class Chunk:
    """One chunk, ready to insert.

    Carries its own ``metadata`` so the route does not have to know how speaker
    attribution is derived. ``docs/schema/README.md`` section 3: metadata is
    "what turns a retrieved chunk into a clickable citation rather than an
    unattributed blob".
    """

    content: str
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def chunk_document(text: str) -> List[Chunk]:
    """Split text and attribute each chunk. The one entry point for ingest.

    ``speaker`` is set only when the chunk covers exactly one turn, because for
    anything else a single name would be a lie -- a chunk spanning four turns
    has no speaker. ``speakers`` carries the full list either way, and is what
    the roster in Phase 2.2 reads instead of re-parsing the text.
    """
    turns = parse_speaker_turns(text)
    chunks: List[Chunk] = []
    for content, start, end in split_into_chunks(text):
        metadata: Dict[str, Any] = {"char_start": start, "char_end": end}
        covered = speakers_in(turns, start, end)
        if covered:
            metadata["speakers"] = covered
            metadata["speaker"] = covered[0] if len(covered) == 1 else None
        chunks.append(Chunk(content=content, char_start=start, char_end=end,
                            metadata=metadata))
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
