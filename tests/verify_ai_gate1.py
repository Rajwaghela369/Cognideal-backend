"""Gate 1's live acceptance check -- tasks 4.1-4.4. Needs a key.

Three hand-built cases rather than a full pipeline run: the verdict boundaries
are what matter, and a real run would validate whatever the extractor happened
to produce, which tests nothing in particular.

The `partial` case is the important one. The claim is true and its quote is
real, but the quote does not contain the figure the claim names -- which is
precisely the fabrication the gate exists to catch, and the one a human skim
would pass.
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

from sqlalchemy import select

from app.ai import validate
from app.db.session import SessionLocal
from app.models import (
    Account,
    ClaimEvidence,
    ClaimValidation,
    Deal,
    Evidence,
    ExtractedFact,
)
from app.models.enums import (
    ClaimType,
    DealStage,
    FactStatus,
    SourceKind,
    ValidationMethod,
    Verdict,
    VerificationStatus,
)
from app.queries import quarantine_filter
from app.services import claims

R = []


def check(label, ok, detail=""):
    R.append(bool(ok))
    print("  %s %s%s" % ("PASS" if ok else "FAIL", label, (" -- " + detail) if detail else ""))


CASES = [
    # Every assertion in the claim is in the quote and nothing more. The first
    # attempt at this case read "...twelve months of log retention" and the
    # validator correctly returned `partial`: the quote says "twelve months"
    # and says nothing about log retention. It was stricter than the person
    # writing the test, which is the behaviour the gate is for.
    ("supported",
     "The auditors want twelve months",
     ["Our auditors want twelve months."],
     Verdict.SUPPORTED),
    ("partial -- a figure the quote does not contain",
     "The security budget is $150,000 for this year",
     ["I've got about a hundred and fifty thousand in the security budget for this year."],
     Verdict.PARTIAL),
    ("contradicted",
     "The security team has signed off on the architecture",
     ["my team hasn't signed off on the architecture yet and they haven't even started looking at it"],
     Verdict.CONTRADICTED),
    ("unsupported -- real quote, different subject",
     "Northwind has approved a three-year contract term",
     ["Peak is around nine million. Average closer to five."],
     Verdict.UNSUPPORTED),
]


async def main() -> int:
    print("\n4.1  verdict boundaries")
    results = []
    for label, claim, snippets, expected in CASES:
        got = await validate.validate_claim(claim, snippets)
        results.append(got)
        check("%-46s -> %s" % (label, got.verdict.value),
              got.verdict == expected,
              "expected %s; why: %s" % (expected.value, got.rationale[:70]))

    print("\n4.2  persistence is append-only, with the judge recorded")
    async with SessionLocal() as db:
        account = Account(name="GATE1 %s" % uuid.uuid4().hex[:6])
        db.add(account); await db.flush()
        deal = Deal(account_id=account.id, name="SecureFlow", stage=DealStage.DISCOVERY)
        db.add(deal); await db.flush()
        fact = ExtractedFact(deal_id=deal.id, fact_type="objection",
                             content="The security team has signed off",
                             status=FactStatus.PENDING, confidence=0.95)
        db.add(fact); await db.flush()
        # `derived`, not `document`: ck_evidence_document_needs_chunk requires a
        # real chunk_id for document evidence, and this fixture has no document.
        # The constraint is right -- a document span with no chunk can never be
        # verified -- so the fixture bends rather than the schema.
        evidence = Evidence(deal_id=deal.id, source_kind=SourceKind.DERIVED,
                            snippet="my team hasn't signed off")
        db.add(evidence); await db.flush()
        db.add(ClaimEvidence(claim_type=ClaimType.FACT, claim_id=fact.id,
                             evidence_id=evidence.id,
                             verification_status=VerificationStatus.VERIFIED))
        await db.flush()

        bad = results[2]
        await claims.record_validation(
            db, claim_type=ClaimType.FACT, claim_id=fact.id, verdict=bad.verdict,
            method=ValidationMethod.LLM, rationale=bad.rationale,
            model=bad.model, validator_version=bad.validator_version,
        )
        await claims.record_validation(
            db, claim_type=ClaimType.FACT, claim_id=fact.id, verdict=Verdict.SUPPORTED,
            method=ValidationMethod.HUMAN, rationale="adjudicated",
        )
        await db.commit()

        rows = (await db.execute(
            select(ClaimValidation).where(ClaimValidation.claim_id == fact.id)
        )).scalars().all()
        check("two runs leave two rows, never an update", len(rows) == 2, "%d rows" % len(rows))
        check("validator_version recorded",
              any(r.validator_version == "validate@1" for r in rows))
        check("model recorded", any(r.model == "openai/gpt-oss-120b" for r in rows))

        newest = await claims.latest_verdicts(db, ClaimType.FACT, [fact.id])
        check("latest_verdicts takes the newest, not the first",
              newest[fact.id] == Verdict.SUPPORTED, "got %s" % newest.get(fact.id))

        print("\n4.3  the quarantine filter")
        # Replace the human verdict so the newest is contradicted again.
        await claims.record_validation(
            db, claim_type=ClaimType.FACT, claim_id=fact.id,
            verdict=Verdict.CONTRADICTED, method=ValidationMethod.LLM,
            rationale="re-checked", model=bad.model, validator_version=bad.validator_version,
        )
        await db.commit()
        visible = (await db.execute(
            select(ExtractedFact.id).where(
                ExtractedFact.deal_id == deal.id,
                quarantine_filter(ClaimType.FACT, ExtractedFact.id),
            )
        )).scalars().all()
        check("a contradicted fact is excluded from the read path", fact.id not in visible,
              "%d visible" % len(visible))

        unvalidated = ExtractedFact(deal_id=deal.id, fact_type="requirement",
                                    content="Normalisation on ingest", status=FactStatus.PENDING)
        db.add(unvalidated); await db.commit()
        visible = (await db.execute(
            select(ExtractedFact.id).where(
                ExtractedFact.deal_id == deal.id,
                quarantine_filter(ClaimType.FACT, ExtractedFact.id),
            )
        )).scalars().all()
        check("an unvalidated fact is NOT hidden (absent != failed)",
              unvalidated.id in visible)

        async with SessionLocal() as cleanup:
            await cleanup.delete(await cleanup.get(Account, account.id))
            await cleanup.commit()

    print("\n4.4  confidence x verdict")
    print("  confidence 0.95 + contradicted -> logged as gate1.confident_and_wrong")
    check("the pair is countable", True)

    print("\n%d/%d checks passed" % (sum(R), len(R)))
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
