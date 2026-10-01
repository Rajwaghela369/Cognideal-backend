"""Fill in label offsets from the chunks the real upload path produced.

Task 1.2's acceptance is "the labels are machine-readable and offsets resolve
verbatim against the ingested chunks", and that ordering is the point: offsets
cannot be authored by hand. ``split_into_chunks`` decides the boundaries, so a
hand-written offset is a guess about an implementation detail -- and Gate 0
checks ``evidence.snippet == document_chunks.content[char_start:char_end]``,
which is an exact comparison against a chunk, not against the file.

So the snippets are authored and the offsets are derived. This script resolves
each snippet against the loaded chunks and writes the result back into the
label file.

Three outcomes per fact, and the third is a genuine finding rather than an
error in the labels:

*   **resolved** -- found verbatim inside one chunk. Offsets recorded.
*   **not in the document** -- the authored snippet is wrong. Fix the label.
*   **straddles a chunk boundary** -- present in the file but in no single
    chunk. No extractor can cite it within one chunk, so either the snippet
    must be shortened or the fact is uncitable as written. Reported loudly
    because it is a property of the chunking, not of the prompt.

Run after ``load.py``:

    ../deal-pilot-env/bin/python tests/fixtures/resolve_labels.py
"""

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, ".")

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Account, Deal, Document, DocumentChunk

HERE = pathlib.Path(__file__).parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


async def chunks_for(db, original_filename):
    document = await db.scalar(
        select(Document)
        .join(Deal, Document.deal_id == Deal.id)
        .join(Account, Deal.account_id == Account.id)
        .where(
            Account.name == MANIFEST["account"]["name"],
            Document.original_filename == original_filename,
        )
    )
    if document is None:
        return None, []
    rows = (
        await db.execute(
            select(DocumentChunk.chunk_index, DocumentChunk.content)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
    ).all()
    return document, rows


def locate(snippet, rows):
    """First chunk containing the snippet verbatim, plus how many contain it.

    First rather than any: chunks overlap by ``chunk_overlap_chars``, so a
    snippet near a boundary legitimately appears twice. Either citation passes
    Gate 0; recording the lowest index just makes the label deterministic.
    """
    hits = []
    for index, content in rows:
        at = content.find(snippet)
        if at != -1:
            hits.append((index, at, at + len(snippet), content))
    if not hits:
        return None
    index, start, end, content = hits[0]
    assert content[start:end] == snippet, "offset arithmetic is wrong"
    return {"chunk_index": index, "char_start": start, "char_end": end,
            "appears_in_chunks": len(hits)}


async def main() -> int:
    resolved_total = straddling = missing = 0

    async with SessionLocal() as db:
        for meeting in MANIFEST["meetings"]:
            name = pathlib.Path(meeting["transcript"]).name
            label_path = HERE / "labels" / (pathlib.Path(name).stem + ".json")
            if not label_path.exists():
                print("  no label file for %s" % name)
                continue

            document, rows = await chunks_for(db, name)
            if document is None:
                print("FAIL %s is not loaded -- run load.py first" % name)
                return 1

            source = (HERE / meeting["transcript"]).read_text()
            labels = json.loads(label_path.read_text())
            print("\n%s  (%d chunks)" % (name, len(rows)))

            for fact in labels["facts"]:
                found = locate(fact["snippet"], rows)
                if found:
                    fact["chunk_index"] = found["chunk_index"]
                    fact["char_start"] = found["char_start"]
                    fact["char_end"] = found["char_end"]
                    if found["appears_in_chunks"] > 1:
                        fact["appears_in_chunks"] = found["appears_in_chunks"]
                    else:
                        fact.pop("appears_in_chunks", None)
                    resolved_total += 1
                    print("  ok   %-4s chunk %d [%4d:%4d] %s%s"
                          % (fact["id"], found["chunk_index"], found["char_start"],
                             found["char_end"], fact["fact_type"],
                             "  (also in %d chunks)" % found["appears_in_chunks"]
                             if found["appears_in_chunks"] > 1 else ""))
                elif fact["snippet"] in source:
                    straddling += 1
                    fact["unciteable"] = "straddles a chunk boundary"
                    print("  SPAN %-4s straddles a chunk boundary: %r"
                          % (fact["id"], fact["snippet"][:60]))
                else:
                    missing += 1
                    fact["unciteable"] = "snippet not present in the transcript"
                    print("  BAD  %-4s snippet not in the transcript: %r"
                          % (fact["id"], fact["snippet"][:60]))

            label_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")

    print("\n%d resolved, %d straddling a boundary, %d not found"
          % (resolved_total, straddling, missing))
    return 0 if (straddling == 0 and missing == 0) else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
