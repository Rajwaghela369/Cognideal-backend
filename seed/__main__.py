"""Seed the demo portfolio. Safe to re-run: anything already there is left alone.

Run from ``backend/`` with the same environment the API uses (``backend/.env``
locally, or a Render shell in production):

    python -m seed                      # accounts, deals, meetings, tasks, transcripts
    python -m seed --skip-transcripts   # without touching object storage
    python -m seed --analyze            # also queue AI analysis for the worker

Each row is looked up by its natural key before it is inserted (see
``seed/data.py``), so a second run reports everything as existing and writes
nothing. A row edited in the app after seeding keeps the edit -- the seeder
only ever adds what is missing, it never updates or deletes.

What is deliberately *not* seeded: facts, commitments, risks and
recommendations. Those are claims with evidence pointing into transcript
chunks, and the pipeline is what produces them. Fabricating them here would
put model-shaped assertions in the database that no model made. Deterministic
risks are computed for every deal at the end (Tier 1, no AI); ``--analyze``
additionally queues the AI pass, which runs in the worker when AI is enabled.

Transcripts are stored the way the upload endpoint stores them -- text
extracted, chunked, bytes put to object storage, all in one transaction -- so
the pipeline reads them exactly as it would an uploaded file.
"""

import argparse
import asyncio
import pathlib
import sys
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import SessionLocal, engine
from app.models import (
    Account,
    Contact,
    Deal,
    DealContact,
    DealStageHistory,
    Document,
    DocumentChunk,
    Meeting,
    MeetingAttendee,
    Task,
)
from app.models.enums import DocumentSourceType
from app.services import activity, analysis, ingest, storage
from seed.data import ACCOUNTS, AE

TRANSCRIPTS = pathlib.Path(__file__).parent / "transcripts"

created: Counter = Counter()
existing: Counter = Counter()


def _anchor() -> datetime:
    """Today at 14:00 UTC: meetings land at a plausible hour, not run time."""
    today = datetime.now(timezone.utc).date()
    return datetime.combine(today, time(14, 0), tzinfo=timezone.utc)


async def _get_or_add(db: AsyncSession, kind: str, model, lookup: dict, values: dict):
    """Return the row matching ``lookup``, inserting ``lookup | values`` if absent."""
    row = await db.scalar(select(model).filter_by(**lookup).limit(1))
    if row is not None:
        existing[kind] += 1
        return row, False
    row = model(**lookup, **values)
    db.add(row)
    await db.flush()
    created[kind] += 1
    return row, True


async def _seed_account(db: AsyncSession, spec: dict) -> Dict[str, Contact]:
    account, _ = await _get_or_add(
        db, "accounts", Account, {"name": spec["name"]},
        {k: spec[k] for k in ("industry", "website", "employee_band", "hq_region")},
    )
    contacts = {}
    for c in spec["contacts"]:
        contact, _ = await _get_or_add(
            db, "contacts", Contact, {"account_id": account.id, "email": c["email"]},
            {k: c[k] for k in ("first_name", "last_name", "title", "phone")},
        )
        contacts[c["key"]] = contact
    contacts["__account__"] = account
    return contacts


async def _seed_deal(db: AsyncSession, account: Account, contacts: Dict[str, Contact],
                     spec: dict, anchor: datetime, with_transcripts: bool) -> Deal:
    def at(days: int) -> datetime:
        return anchor + timedelta(days=days)

    closed_days_ago = spec.get("closed_days_ago")
    deal, is_new = await _get_or_add(
        db, "deals", Deal, {"account_id": account.id, "name": spec["name"]},
        {
            "stage": spec["stage"],
            "value": spec["value"],
            "currency": spec["currency"],
            "win_probability": spec["win_probability"],
            "expected_close_date": at(spec["expected_close_in_days"]).date(),
            "closed_at": at(-closed_days_ago) if closed_days_ago is not None else None,
            "created_at": at(-spec["created_days_ago"]),
        },
    )

    # History belongs to the deal's creation: written once, never appended to
    # on a re-run, so a stage changed in the app is not contradicted here.
    has_history = await db.scalar(
        select(DealStageHistory.id).where(DealStageHistory.deal_id == deal.id).limit(1)
    )
    if has_history is None:
        previous = None
        for stage, days, note in spec["stage_history"]:
            db.add(DealStageHistory(deal_id=deal.id, from_stage=previous, to_stage=stage,
                                    changed_at=at(days), note=note))
            previous = stage
            created["stage history"] += 1
    else:
        existing["stage history"] += len(spec["stage_history"])

    for s in spec["stakeholders"]:
        await _get_or_add(
            db, "stakeholders", DealContact,
            {"deal_id": deal.id, "contact_id": contacts[s["contact"]].id},
            {
                "buying_role": s["buying_role"],
                "influence": s["influence"],
                "sentiment": s["sentiment"],
                "is_primary": s.get("is_primary", False),
                "notes": s.get("notes"),
            },
        )

    by_name = {f"{c.first_name} {c.last_name}": c for k, c in contacts.items()
               if k != "__account__"}
    for m in spec["meetings"]:
        start = at(m["days"])
        completed = m["status"] == "completed"
        meeting, _ = await _get_or_add(
            db, "meetings", Meeting, {"deal_id": deal.id, "title": m["title"]},
            {
                "meeting_type": m["meeting_type"],
                "status": m["status"],
                "scheduled_at": start,
                "started_at": start if completed else None,
                "ended_at": start + timedelta(minutes=m["minutes"]) if completed else None,
            },
        )
        for name in m.get("attendees", []):
            contact = by_name.get(name)
            await _get_or_add(
                db, "attendees", MeetingAttendee,
                {"meeting_id": meeting.id, "raw_name": name},
                {
                    "contact_id": contact.id if contact else None,
                    "is_internal": name == AE,
                    "attended": True,
                },
            )
        if completed:
            await activity.touch_deal(db, deal.id, meeting.ended_at)

    for t in spec["tasks"]:
        due = at(t["due_in_days"])
        done = t["status"] == "done"
        await _get_or_add(
            db, "tasks", Task, {"deal_id": deal.id, "title": t["title"]},
            {
                "description": t.get("description"),
                "due_date": due.date(),
                "status": t["status"],
                "priority": t["priority"],
                "completed_at": due if done else None,
            },
        )

    await db.commit()

    if with_transcripts:
        for m in spec["meetings"]:
            if "transcript" in m:
                await _seed_transcript(db, deal, m)
    return deal


async def _seed_transcript(db: AsyncSession, deal: Deal, spec: dict) -> None:
    """Mirror of ``POST /deals/{id}/documents`` with ``meeting_id`` set."""
    meeting = await db.scalar(
        select(Meeting).where(Meeting.deal_id == deal.id, Meeting.title == spec["title"])
    )
    if meeting.transcript_document_id is not None:
        existing["transcripts"] += 1
        return

    path = TRANSCRIPTS / spec["transcript"]
    data = path.read_bytes()
    digest = storage.content_hash(data)

    stored = await db.scalar(select(Document).where(Document.content_hash == digest))
    if stored is not None:
        if stored.deal_id != deal.id:
            print(f"  ! {path.name} is already stored on another deal; not attached")
            return
        meeting.transcript_document_id = stored.id
        await db.commit()
        existing["transcripts"] += 1
        return

    text = ingest.extract_text(path.name, "text/plain", data)
    document = Document(
        deal_id=deal.id,
        account_id=deal.account_id,
        source_type=DocumentSourceType.MEETING_TRANSCRIPT,
        title=f"{spec['title']} transcript",
        original_filename=path.name,
        storage_uri=storage.object_key(digest),
        mime_type="text/plain",
        byte_size=len(data),
        content_hash=digest,
        occurred_at=meeting.started_at,
    )
    db.add(document)
    await db.flush()
    for index, chunk in enumerate(ingest.chunk_document(text)):
        db.add(DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk.content,
            token_count=len(chunk.content.split()),
            chunk_metadata=chunk.metadata,
        ))
    await db.flush()
    await activity.touch_deal(db, deal.id, document.occurred_at)
    meeting.transcript_document_id = document.id

    # Same ordering as the endpoint: rows flushed but uncommitted while the
    # bytes go up, so a storage failure leaves nothing behind in Postgres.
    try:
        storage.put_object(document.storage_uri, data, "text/plain")
    except Exception as exc:
        await db.rollback()
        print(f"  ! could not store {path.name}; nothing saved ({exc})")
        return
    await db.commit()
    created["transcripts"] += 1


async def main(analyze: bool, with_transcripts: bool) -> int:
    anchor = _anchor()
    async with SessionLocal() as db:
        deals = []
        for account_spec in ACCOUNTS:
            contacts = await _seed_account(db, account_spec)
            account = contacts["__account__"]
            await db.commit()
            for deal_spec in account_spec["deals"]:
                deal = await _seed_deal(db, account, contacts, deal_spec, anchor,
                                        with_transcripts)
                deals.append(deal)
                print(f"  {account.name} / {deal.name}")

        # Deterministic risks (Tier 1) for every deal, as any write in the app
        # would trigger. Tier 2 -- the AI pass -- only on request.
        for deal in deals:
            await analysis.record_change(db, deal.id, "seed", tier2=analyze)
        await db.commit()

    await engine.dispose()

    print()
    print("%-16s %8s %8s" % ("", "created", "existing"))
    for kind in ("accounts", "contacts", "deals", "stage history", "stakeholders",
                 "meetings", "attendees", "tasks", "transcripts"):
        print("%-16s %8d %8d" % (kind, created[kind], existing[kind]))
    return 0


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m seed", description=__doc__.split("\n")[0])
    parser.add_argument("--analyze", action="store_true",
                        help="queue the AI analysis pass for each deal (needs AI_ENABLED)")
    parser.add_argument("--skip-transcripts", action="store_true",
                        help="do not ingest transcripts or touch object storage")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()
    sys.exit(asyncio.run(main(args.analyze, not args.skip_transcripts)))
