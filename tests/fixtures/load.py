"""Load the fixture corpus. Idempotent -- re-running replaces it.

Structural rows are inserted directly because they carry backdated timestamps
the API will not accept: a deal created in July, meetings in July, August and
September. The *documents* go through the real HTTP upload path, because that
path is what extracts text, cuts chunks and records offsets -- and every label
in ``labels/`` points into those chunks. Seeding documents directly would mean
labelling offsets that the running system never produces.

``meeting_attendees`` is deliberately left empty. Deriving the roster from
transcript speaker labels is Phase 2.2, and seeding it would hide whether that
works. ``manifest.json`` records the expected rosters as labels instead.

Needs postgres and minio up, and the API running:

    docker compose up -d postgres minio
    ../deal-pilot-env/bin/uvicorn app.main:app --port 8099 --log-level warning &
    ../deal-pilot-env/bin/python tests/fixtures/load.py
"""

import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

import httpx
from sqlalchemy import delete, select

from app.db.session import SessionLocal
from app.models import (
    Account,
    Contact,
    Deal,
    DealContact,
    DealStageHistory,
    Document,
    Meeting,
)

HERE = pathlib.Path(__file__).parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
API = "http://127.0.0.1:8099/api/v1"


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


async def wipe(db) -> None:
    """Remove a previous load, including documents.

    Documents carry UNIQUE(content_hash) globally, so a re-run would otherwise
    get 200 "already stored" and attach the chunks to the previous deal. They
    cascade from the account, so deleting the account is enough -- but it has to
    happen before the upload, not after.
    """
    existing = await db.scalar(select(Account).where(Account.name == MANIFEST["account"]["name"]))
    if existing is not None:
        await db.execute(delete(Account).where(Account.id == existing.id))
        await db.commit()


async def main() -> int:
    async with SessionLocal() as db:
        await wipe(db)

        account = Account(**MANIFEST["account"])
        db.add(account)
        await db.flush()

        spec = dict(MANIFEST["deal"])
        deal = Deal(
            account_id=account.id,
            name=spec["name"],
            stage=spec["stage"],
            value=spec["value"],
            currency=spec["currency"],
            expected_close_date=_ts(spec["expected_close_date"] + "T00:00:00Z").date(),
            created_at=_ts(spec["created_at"]),
        )
        db.add(deal)
        await db.flush()

        # The stage history row the stalled_stage rule reads. Without it
        # STAGE_CHANGED_AT falls back to deals.created_at, which happens to be
        # close here -- but the rule should be exercised against a real row.
        db.add(DealStageHistory(deal_id=deal.id, to_stage=spec["stage"],
                                changed_at=_ts("2026-07-28T14:00:00Z")))

        for c in MANIFEST["contacts"]:
            contact = Contact(
                account_id=account.id,
                first_name=c["first_name"],
                last_name=c["last_name"],
                title=c["title"],
            )
            db.add(contact)
            await db.flush()
            db.add(DealContact(
                deal_id=deal.id, contact_id=contact.id,
                buying_role=c["buying_role"], influence=c["influence"],
                is_primary=c["is_primary"],
            ))

        meetings = {}
        for m in MANIFEST["meetings"]:
            occurred = _ts(m["occurred_at"])
            meeting = Meeting(
                deal_id=deal.id, title=m["title"], meeting_type=m["meeting_type"],
                status=m["status"], scheduled_at=occurred, started_at=occurred,
                ended_at=occurred,
            )
            db.add(meeting)
            await db.flush()
            meetings[m["key"]] = meeting.id

        deal_id = deal.id
        await db.commit()

    # --- documents through the real upload path ---
    uploaded = {}
    with httpx.Client(base_url=API, timeout=60) as http:
        for m in MANIFEST["meetings"]:
            path = HERE / m["transcript"]
            response = http.post(
                "/deals/%s/documents" % deal_id,
                files={"file": (path.name, path.read_bytes(), "text/plain")},
                data={
                    "source_type": "meeting_transcript",
                    "title": m["title"] + " transcript",
                    # Dated by when the conversation happened, which is also
                    # what last_activity_at and recency ranking read.
                    "occurred_at": m["occurred_at"],
                },
            )
            if response.status_code not in (200, 201):
                print("FAIL upload %s -> %s %s" % (path.name, response.status_code, response.text[:200]))
                return 1
            uploaded[m["key"]] = response.json()["id"]

    # Link each transcript to its meeting -- transcript_document_id is what the
    # pipeline reads to find the text.
    async with SessionLocal() as db:
        for key, document_id in uploaded.items():
            meeting = await db.get(Meeting, meetings[key])
            meeting.transcript_document_id = document_id
        await db.commit()

        chunk_counts = {}
        for m in MANIFEST["meetings"]:
            document = await db.get(Document, uploaded[m["key"]])
            from app.models import DocumentChunk

            n = await db.scalar(
                select(DocumentChunk.id).where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index.desc()).limit(1)
            )
            total = await db.scalar(
                select(DocumentChunk.chunk_index).where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index.desc()).limit(1)
            )
            chunk_counts[m["key"]] = (total or 0) + 1

    print("loaded deal %s" % deal_id)
    for key, document_id in uploaded.items():
        print("  %-16s document=%s chunks=%d" % (key, document_id, chunk_counts[key]))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
