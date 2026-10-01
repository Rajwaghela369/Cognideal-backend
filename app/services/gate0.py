"""Gate 0 -- span integrity. Deterministic, no model, never skippable.

The cheapest and most valuable check in the product. It asks one question per
citation: **does the cited source exist, and does it say this, byte for byte?**
Not "is the claim true" -- that is Gate 3's question and a human's to answer.

Why it is worth so much: nearly every model failure in this product is a
grounding failure, and a fabricated or paraphrased citation is both the most
common and the most damaging. Catching it costs a substring comparison. A model
is never asked to do this (docs/schema/README.md section 5).

In ``services/`` rather than ``app/ai/`` deliberately: nothing here touches a
provider, and the gate has to run on the write path whether a model was
involved or not -- a human-entered citation is checked by exactly the same
code. ``services/facts.py`` is what makes it non-skippable.

Three checks:

*   ``source_kind='document'`` -- ``evidence.snippet`` must appear verbatim in
    ``document_chunks.content`` at exactly ``[char_start, char_end]``.
*   ``source_kind='record'`` -- ``record_ref`` must resolve to a live row whose
    named field still holds the value recorded in ``snippet``. This is what
    makes a claim self-invalidate: move a deal's stage and the citation goes
    ``value_drifted`` without anyone noticing.
*   **The literal rule** -- if the claim text contains a date or a monetary
    amount, that literal must appear in a cited span or a resolved record
    value. Models are worst at exactly these and it is a string match.

The literal rule is why the fixture corpus speaks every number aloud
("a hundred and fifty thousand", "the seventh of August"). A model that tidies
speech into ``$150,000`` produces a figure that appears nowhere in the
transcript, and this rejects it -- correctly, even though the claim is true.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk
from app.models.enums import SourceKind, VerificationStatus

# Tables a record_ref may name. An allowlist, not a convenience: record_ref is
# jsonb written by a model, and resolving an arbitrary table name from it would
# be an injection point into our own schema.
RESOLVABLE_TABLES = {
    "deals": {"stage", "value", "currency", "expected_close_date", "closed_at",
              "last_activity_at", "win_probability", "name"},
    "deal_contacts": {"buying_role", "influence", "sentiment", "is_primary"},
    "deal_stage_history": {"from_stage", "to_stage", "changed_at"},
    "meetings": {"status", "meeting_type", "scheduled_at", "started_at", "ended_at",
                 "summary", "sentiment"},
    "meeting_attendees": {"raw_name", "contact_id", "is_internal", "attended"},
    "commitments": {"status", "due_date", "owner_side", "owner_name", "description"},
    "contacts": {"first_name", "last_name", "title", "email"},
    "tasks": {"status", "due_date", "title"},
}

# A monetary amount written in figures: $180,000 / 180000 USD / 1.5m.
_MONEY = re.compile(
    r"(?:[$£€]\s?\d[\d,.]*\s?(?:k|m|bn|million|billion)?)"
    r"|(?:\b\d[\d,.]*\s?(?:usd|eur|gbp|dollars?)\b)",
    re.IGNORECASE,
)
# A date written in figures: 2026-10-15, 15/10/2026, Oct 15, 15 October 2026.
_DATE = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}\b)"
    r"|(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)"
    r"|(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b)"
    r"|(?:\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)",
    re.IGNORECASE,
)


def literals_in(claim_text: str) -> List[str]:
    """Dates and money written in figures in the claim text.

    Only figures. Spoken forms ("a hundred and fifty thousand") are not
    extracted, because the rule exists to catch a model *converting* speech
    into a precise-looking literal that was never said -- which is the
    fabrication that most easily passes a human skim.
    """
    found = [m.group(0).strip() for m in _MONEY.finditer(claim_text)]
    found += [m.group(0).strip() for m in _DATE.finditer(claim_text)]
    return sorted(set(found))


def _normalise(value: str) -> str:
    """Compare money and dates without punctuation noise.

    "$180,000" and "180000" are the same figure, and a claim that writes one
    while the source writes the other is not a fabrication.
    """
    return re.sub(r"[\s,$£€]", "", value).lower().rstrip(".")


@dataclass
class LinkCheck:
    """The outcome for one claim -> evidence link."""

    evidence_id: Optional[uuid.UUID]
    status: VerificationStatus
    detail: Optional[str] = None

    @property
    def verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED


@dataclass
class ClaimCheck:
    """The outcome for a whole claim: its links, plus the literal rule."""

    links: List[LinkCheck] = field(default_factory=list)
    unsupported_literals: List[str] = field(default_factory=list)

    @property
    def surviving(self) -> List[LinkCheck]:
        return [link for link in self.links if link.verified]

    @property
    def passed(self) -> bool:
        """A claim passes only with at least one verified link and no
        unsupported literal. "No evidence" is a rejection, not a warning."""
        return bool(self.surviving) and not self.unsupported_literals

    @property
    def reason(self) -> Optional[str]:
        if self.unsupported_literals:
            return "literal not in any cited span: %s" % ", ".join(self.unsupported_literals)
        if not self.links:
            return "no evidence cited"
        if not self.surviving:
            return "no surviving citation (%s)" % ", ".join(
                sorted({link.status.value for link in self.links})
            )
        return None


async def check_document_span(
    db: AsyncSession,
    chunk_id: Optional[uuid.UUID],
    snippet: str,
    char_start: Optional[int],
    char_end: Optional[int],
) -> LinkCheck:
    """Does the chunk still say this, at exactly these offsets?"""
    if chunk_id is None or char_start is None or char_end is None:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "incomplete span")

    content = await db.scalar(
        select(DocumentChunk.content).where(DocumentChunk.id == chunk_id)
    )
    if content is None:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "chunk does not exist")
    if char_end > len(content) or char_start < 0 or char_start >= char_end:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "offsets out of range")
    if content[char_start:char_end] != snippet:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "text at the offsets differs")
    return LinkCheck(None, VerificationStatus.VERIFIED)


async def check_record_ref(
    db: AsyncSession, record_ref: Optional[dict], snippet: str
) -> LinkCheck:
    """Does the named field still hold this value?

    ``value_drifted`` rather than ``span_missing`` when it does not: the
    citation was honest when it was made and the world moved, which is a
    different problem from a fabrication and wants a different fix.
    """
    if not record_ref:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "no record_ref")

    table, row_id, field_name = (
        record_ref.get("table"), record_ref.get("id"), record_ref.get("field")
    )
    allowed = RESOLVABLE_TABLES.get(table or "")
    if allowed is None:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "table %r not resolvable" % table)
    if field_name not in allowed:
        return LinkCheck(
            None, VerificationStatus.SPAN_MISSING,
            "field %r not resolvable on %s" % (field_name, table),
        )
    if not row_id:
        # A derived reference names a table and a field but no row -- "no
        # economic buyer is listed" has nothing to point at. Nothing to drift.
        return LinkCheck(None, VerificationStatus.VERIFIED, "derived, no row")

    # Table and column are both allowlisted above, so this cannot be steered
    # by record_ref contents; the id stays a bound parameter.
    current = await db.scalar(
        sql_text("SELECT %s::text FROM %s WHERE id = :id" % (field_name, table)),
        {"id": str(row_id)},
    )
    if current is None:
        return LinkCheck(None, VerificationStatus.SPAN_MISSING, "row or value is gone")
    if _normalise(current) != _normalise(snippet):
        return LinkCheck(
            None, VerificationStatus.VALUE_DRIFTED,
            "now %r, cited as %r" % (current, snippet),
        )
    return LinkCheck(None, VerificationStatus.VERIFIED)


async def check_claim(
    db: AsyncSession,
    claim_text: str,
    citations: Sequence[Dict[str, Any]],
) -> ClaimCheck:
    """Run every check for one claim.

    ``citations`` are dicts rather than ORM rows so this works identically on
    candidates that have not been written yet (the extractor's output) and on
    links already in the database (re-verification, Phase 10 tier 0).
    """
    result = ClaimCheck()
    verified_text: List[str] = []

    for citation in citations:
        kind = citation.get("source_kind")
        if kind == SourceKind.DOCUMENT or kind == SourceKind.DOCUMENT.value:
            check = await check_document_span(
                db, citation.get("chunk_id"), citation.get("snippet") or "",
                citation.get("char_start"), citation.get("char_end"),
            )
        else:
            check = await check_record_ref(db, citation.get("record_ref"), citation.get("snippet") or "")
        check.evidence_id = citation.get("evidence_id")
        result.links.append(check)
        if check.verified:
            verified_text.append(citation.get("snippet") or "")

    haystack = _normalise(" ".join(verified_text))
    result.unsupported_literals = [
        literal for literal in literals_in(claim_text)
        if _normalise(literal) not in haystack
    ]
    return result
