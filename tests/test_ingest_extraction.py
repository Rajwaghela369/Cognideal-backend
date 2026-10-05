"""PDF and docx text extraction -- tasks C8-C13. No model, no database.

Until these existed the only extraction path was `data.decode("utf-8")`, so
every PDF and every Word document was refused at upload. The whitelist was
doing the refusing, and that is the detail worth keeping in mind while reading
these tests: **the point of the whitelist was never to be a whitelist.** It
exists so that nothing reaches the database that cannot be chunked, because a
document with no chunks can carry no citation, and a document that can carry no
citation is invisible to every part of this product that matters.

So the format tests are the easy half. The ones that earn their place are the
refusals -- a scanned PDF and an image both parse or fail in ways that could
plausibly end with a stored document and zero chunks, and that outcome is
silent. `extract_text` is the only place it can be stopped, because
`documents.py` calls it before the first byte goes to MinIO.

The fixtures are real files, rebuilt by `fixtures/documents/regenerate.py`.
"""

import pathlib

import pytest
from fastapi import HTTPException

from app.services.ingest import chunk_document, extract_text

DOCUMENTS = pathlib.Path(__file__).parent / "fixtures" / "documents"

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def fixture(name):
    return (DOCUMENTS / name).read_bytes()


def refusal(name, mime, data=None):
    """Call `extract_text` expecting an HTTPException, and return it."""
    with pytest.raises(HTTPException) as caught:
        extract_text(name, mime, data if data is not None else fixture(name))
    return caught.value


# --------------------------------------------------------------------------
# What extracts
# --------------------------------------------------------------------------


def test_a_pdf_with_a_text_layer_extracts_its_text():
    text = extract_text("security-review.pdf", PDF_MIME, fixture("security-review.pdf"))

    assert "SOC 2 Type II report" in text
    assert "220000 USD" in text


def test_pdf_text_is_chunkable_and_the_offsets_land_in_it():
    """The property the whole refusal policy protects.

    Extracting is not the goal; chunks with offsets that still index the text
    are, because Gate 0 verifies a citation by finding the quoted span inside
    the chunk. A PDF whose text came out unchunkable would pass the test above
    and still be useless.
    """
    text = extract_text("security-review.pdf", PDF_MIME, fixture("security-review.pdf"))
    chunks = chunk_document(text)

    assert chunks
    for chunk in chunks:
        assert text[chunk.char_start:chunk.char_end] == chunk.content


def test_a_docx_extracts_paragraphs_and_table_cells():
    """`document.paragraphs` omits every table, silently.

    A requirements matrix or a pricing grid is exactly what a deal document
    carries, so an extractor that read paragraphs alone would return something
    that looked complete and had lost the substance.
    """
    text = extract_text("requirements.docx", DOCX_MIME, fixture("requirements.docx"))

    assert "SSO via Okta is a hard requirement" in text   # a paragraph
    assert "Confirmed" in text                             # a table cell
    assert "SOC 2\tPending" in text                        # a whole row


def test_a_docx_is_read_when_the_browser_sends_no_mime_type():
    """Matching is mime *or* suffix. Browsers disagree about the docx type."""
    text = extract_text("requirements.docx", None, fixture("requirements.docx"))
    assert "SSO via Okta" in text


def test_a_plain_text_upload_still_works():
    """The path that existed before any of this. Nothing above may break it."""
    assert extract_text("notes.txt", "text/plain", b"Dana called.") == "Dana called."


# --------------------------------------------------------------------------
# What is refused, and with which status
# --------------------------------------------------------------------------


def test_a_scanned_pdf_is_refused_rather_than_stored_with_no_chunks():
    """The case this whole task exists for.

    A PDF with no text layer parses perfectly -- it is a valid document that
    happens to contain pixels. Nothing raises, `extract_text` would return "",
    and the caller would commit a document with zero chunks: present in every
    list, quotable by nothing, and wrong in a way no error message ever
    mentions. 415 rather than the 422 an empty .txt gets, because the media
    type really is the problem: reading it needs OCR, which is deliberately
    not supported, and the fix is on the caller's side.
    """
    error = refusal("scanned-invoice.pdf", PDF_MIME)

    assert error.status_code == 415
    assert "no text layer" in error.detail
    # The message has to say what to do about it, not just what went wrong.
    assert "selectable text" in error.detail


def test_an_image_is_refused_and_the_message_lists_what_works():
    error = refusal("screenshot.png", "image/png")

    assert error.status_code == 415
    assert ".pdf" in error.detail and ".docx" in error.detail


def test_legacy_doc_is_refused_by_name_with_the_fix_in_the_message():
    """Not a generic 415.

    The generic message lists `.docx` one word after refusing `.doc`, which
    leaves the caller to guess what the difference is. `.doc` is a binary OLE
    container; python-docx opens zipped XML and no library here reads it.
    """
    error = refusal("proposal.doc", "application/msword", data=b"\xd0\xcf\x11\xe0junk")

    assert error.status_code == 415
    assert "save as" in error.detail.lower()
    assert ".docx" in error.detail


def test_a_damaged_pdf_is_a_415_not_a_500():
    """Garbage with the right extension is the caller's problem, not a crash."""
    error = refusal("truncated.pdf", PDF_MIME, data=b"%PDF-1.4\nnot really a pdf")

    assert error.status_code == 415
    assert "pdf" in error.detail.lower()


def test_a_renamed_doc_pretending_to_be_docx_is_a_415_not_a_500():
    error = refusal("renamed.docx", DOCX_MIME, data=b"\xd0\xcf\x11\xe0junk")

    assert error.status_code == 415
    assert "legacy .doc" in error.detail.lower()


def test_an_empty_text_file_is_still_a_422():
    """Unchanged, and the contrast that makes the 415 above meaningful.

    The media type is fine and the file is readable; there is simply nothing in
    it. That is a different conversation with the caller.
    """
    assert refusal("empty.txt", "text/plain", data=b"   \n\n  ").status_code == 422
