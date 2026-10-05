"""Rebuild the binary document fixtures. Run by hand, committed output.

    ../../../../deal-pilot-env/bin/python regenerate.py

Same reasoning as `tests/eval/recorded/regenerate.py`: the files next to this
script are real binaries that the extraction tests read, and a committed binary
nobody can regenerate is a fixture nobody can safely change. Four files, each
standing for one branch of `extract_text`:

    security-review.pdf   a text layer, two paragraphs      -> extracts
    scanned-invoice.pdf    a page with no text at all        -> 415
    requirements.docx      paragraphs *and* a table          -> extracts both
    screenshot.png         not a document at all             -> 415

The text PDF is assembled by hand rather than with a PDF authoring library.
reportlab is the usual way and is not a dependency here; adding one to the
project so the tests can build their own input would be the wrong trade. The
file below is a minimal but genuine PDF -- pypdf parses it the same way it
parses anything else, which is the only property the tests need.
"""

import pathlib
import zlib

HERE = pathlib.Path(__file__).parent

PAGE_TEXT = [
    "Northwind security review -- 14 August 2026",
    "",
    "Dana Whitfield confirmed that the completed SOC 2 Type II report",
    "will be circulated to the security reviewers before the end of the",
    "month. No exceptions were raised against the access control policy.",
    "",
    "The annual contract value remains 220000 USD pending CFO approval.",
]


def _content_stream(lines):
    """A Tj per line, 14pt leading, starting near the top of the page."""
    out = ["BT", "/F1 11 Tf", "14 TL", "1 0 0 1 72 720 Tm"]
    for line in lines:
        out.append("(%s) Tj" % line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)"))
        out.append("T*")
    out.append("ET")
    return "\n".join(out).encode("ascii")


def build_pdf(lines):
    """A one-page PDF with a real text layer and a correct xref table."""
    stream = _content_stream(lines)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    # Offsets are byte positions into the finished file, so they have to be
    # recorded while it is being built -- this is the part a hand-written PDF
    # gets wrong, and pypdf rejects the file outright when it does.
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(out)


def build_textless_pdf():
    """A valid PDF whose single page draws nothing.

    The scanned-document case without shipping a scan: what the extractor sees
    is identical -- a page it can parse and no text to take from it. A real
    scan would differ only by carrying an image the extractor also ignores.
    """
    return build_pdf([])


def build_docx():
    import docx

    document = docx.Document()
    document.add_paragraph("Northwind requirements -- discovery")
    document.add_paragraph(
        "SSO via Okta is a hard requirement for the security review."
    )
    # The reason _docx_text reads tables: `document.paragraphs` omits every
    # one of these, so a fixture without a table cannot catch that bug.
    table = document.add_table(rows=3, cols=2)
    for row, (name, value) in enumerate(
        [("Requirement", "Status"), ("SSO (Okta)", "Confirmed"), ("SOC 2", "Pending")]
    ):
        table.rows[row].cells[0].text = name
        table.rows[row].cells[1].text = value
    target = HERE / "requirements.docx"
    document.save(str(target))
    return target


def build_png():
    """A 1x1 PNG, built from chunks rather than pasted as a base64 blob."""

    def chunk(kind, payload):
        body = kind + payload
        return (
            len(payload).to_bytes(4, "big")
            + body
            + zlib.crc32(body).to_bytes(4, "big")
        )

    header = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    written = []
    for name, data in [
        ("security-review.pdf", build_pdf(PAGE_TEXT)),
        ("scanned-invoice.pdf", build_textless_pdf()),
        ("screenshot.png", build_png()),
    ]:
        (HERE / name).write_bytes(data)
        written.append(name)
    written.append(build_docx().name)
    for name in sorted(written):
        print("%-24s %6d bytes" % (name, (HERE / name).stat().st_size))
