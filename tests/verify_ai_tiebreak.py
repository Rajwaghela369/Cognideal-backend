"""Task 2.4's live acceptance check: the attendee-name tiebreak.

Needs a key, so not in the pytest suite. The fixture corpus deliberately
contains no ambiguous name -- resolution settles every speaker without a model
-- so this script constructs the two cases that matter and cleans up after
itself.

The second case is the one worth watching. Two contacts share a first name, so
there is no right answer; the prompt says a shared first name alone is not
enough, and **null is the correct output**. A model that picks the
higher-similarity candidate anyway is guessing, and a wrong contact_id silently
corrupts every attendance-based risk.
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

from sqlalchemy import select

from app.ai import tiebreak
from app.db.session import SessionLocal
from app.models import Account, Contact
from app.services import roster

R = []


def check(label, ok, detail=""):
    R.append(bool(ok))
    print("  %s %s%s" % ("PASS" if ok else "FAIL", label, (" -- " + detail) if detail else ""))


async def main() -> int:
    async with SessionLocal() as db:
        account = Account(name="TIEBREAK %s" % uuid.uuid4().hex[:6])
        db.add(account)
        await db.flush()
        for first, last in (("Priya", "Raman"), ("Marcus", "Webb")):
            db.add(Contact(account_id=account.id, first_name=first, last_name=last))
        await db.flush()
        account_id = account.id
        await db.commit()

    try:
        async with SessionLocal() as db:
            print("\ncase 1: a transcription slip -- 'Priya Ramen'")
            one = await roster.resolve(db, account_id, "Priya Ramen")
            print("  similarity %.2f -> %s" % (one.similarity or 0, one.decision))
            check("deterministic pass defers it", one.needs_tiebreak,
                  "decision=%s" % one.decision)
            if one.needs_tiebreak:
                await tiebreak.resolve_ambiguous(one)
                linked = await db.scalar(
                    select(Contact.first_name + " " + Contact.last_name)
                    .where(Contact.id == one.contact_id)
                ) if one.contact_id else None
                check("model links it to Priya Raman", linked == "Priya Raman",
                      "got %r" % linked)

            print("\ncase 2: two people share the first name -- 'Priya'")
            async with SessionLocal() as extra:
                extra.add(Contact(account_id=account_id, first_name="Priya", last_name="Shah"))
                await extra.commit()

            two = await roster.resolve(db, account_id, "Priya")
            print("  candidates: %s" % ", ".join(
                "%s %.2f" % (c.full_name, c.similarity) for c in two.candidates[:3]))
            check("deterministic pass defers it", two.needs_tiebreak,
                  "decision=%s" % two.decision)
            if two.needs_tiebreak:
                await tiebreak.resolve_ambiguous(two)
                check("model declines to guess (contact_id stays NULL)",
                      two.contact_id is None,
                      "linked to %s" % two.contact_id)

            print("\ncase 3: nobody we know -- 'Tom Alvarez'")
            three = await roster.resolve(db, account_id, "Tom Alvarez")
            check("never reaches the model at all", three.decision == roster.UNKNOWN,
                  "decision=%s similarity=%s" % (three.decision, three.similarity))
    finally:
        async with SessionLocal() as cleanup:
            obsolete = await cleanup.get(Account, account_id)
            if obsolete is not None:
                await cleanup.delete(obsolete)
                await cleanup.commit()

    print("\n%d/%d checks passed" % (sum(R), len(R)))
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
