"""The deal, rendered as cited lines a model can only reference. Task 7.2.

Pure Python. No model call, no provider import, and that is the point: this is
the mechanism by which the risk detector never queries the database.

Every entry is ``handle -> (source_kind, record_ref, text)``. The model is shown
the handles and the text, and answers with handles. It therefore **cannot cite
a source it was not given** -- an evidence reference outside
``dossier.keys()`` is rejected before any write (task 7.3), where a model asked
to invent a ``chunk_id`` or a table name would sometimes produce a plausible one
belonging to nothing.

It also means the model never computes anything. "58 days in discovery" is a
line we calculated and handed over; the model's job is to notice that it
matters, not to do date arithmetic. A claim citing ``r2`` inherits the
correctness of our own SQL.

What goes in is chosen by what the ten risk types actually need, plus the two
things a detector cannot work without: **the currently-open risks**, so
resolution can be asked rather than inferred from silence (task 7.9), and
**prior dismissals**, so the model is not re-proposing what a human already
declined (task 7.10).
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Commitment,
    Contact,
    Deal,
    DealContact,
    DealStageHistory,
    ExtractedFact,
    Meeting,
    MeetingAttendee,
    Recommendation,
    Risk,
)
from app.models.enums import (
    CommitmentStatus,
    FactStatus,
    RecommendationStatus,
    RiskStatus,
    SourceKind,
)

logger = logging.getLogger("dealpilot.ai.dossier")


@dataclass
class Entry:
    """One cited line."""

    handle: str
    source_kind: SourceKind
    text: str
    record_ref: Optional[dict] = None
    #: Set for a `fact` entry: the fact's own id, so a risk citing it can be
    #: traced to the span the fact rests on rather than to a paraphrase of it.
    fact_id: Optional[uuid.UUID] = None


@dataclass
class Dossier:
    deal_id: uuid.UUID
    entries: Dict[str, Entry] = field(default_factory=dict)
    #: Open risks, by id, for the per-risk verdicts in task 7.9.
    open_risks: Dict[str, Any] = field(default_factory=dict)
    #: `(risk_type, risk_key) -> [(reason, action_type, decided_at)]`
    dismissals: Dict[Tuple[str, str], List[Tuple[str, str, Any]]] = field(default_factory=dict)

    def keys(self):
        return self.entries.keys()

    def render(self) -> str:
        """The numbered block the prompt interpolates."""
        return "\n".join(
            "%s. %s" % (entry.handle, entry.text) for entry in self.entries.values()
        )

    def render_open_risks(self) -> str:
        if not self.open_risks:
            return "(none)"
        return "\n".join(
            "%s. [%s] %s" % (rid, risk.risk_type, risk.title)
            for rid, risk in self.open_risks.items()
        )

    def render_dismissals(self) -> str:
        if not self.dismissals:
            return "(none)"
        lines = []
        for (risk_type, risk_key), records in sorted(self.dismissals.items()):
            for reason, action, when in records:
                lines.append(
                    "- %s%s: a %s suggestion was dismissed as '%s'%s"
                    % (risk_type, "/" + risk_key if risk_key else "", action, reason,
                       " on %s" % str(when)[:10] if when else "")
                )
        return "\n".join(lines)


def _ref(table: str, row_id: Optional[Any], field_name: str) -> dict:
    return {"table": table, "id": str(row_id) if row_id else None, "field": field_name}


async def build(db: AsyncSession, deal_id: uuid.UUID) -> Dossier:
    """Assemble the dossier. One deal, one pass, no model."""
    dossier = Dossier(deal_id=deal_id)
    n = [0]

    def add(source_kind, text, record_ref=None, fact_id=None, prefix="r"):
        n[0] += 1
        handle = "%s%d" % (prefix, n[0])
        dossier.entries[handle] = Entry(
            handle=handle, source_kind=source_kind, text=text,
            record_ref=record_ref, fact_id=fact_id,
        )
        return handle

    deal = await db.get(Deal, deal_id)
    if deal is None:
        return dossier

    # --- the deal itself. Every value is also a resolvable record_ref, so a
    # risk about the stage cites the stage and self-invalidates when it moves.
    add(SourceKind.RECORD, "stage = %s" % deal.stage.value,
        _ref("deals", deal.id, "stage"))
    if deal.value is not None:
        add(SourceKind.RECORD, "deal value = %s %s" % (deal.value, deal.currency),
            _ref("deals", deal.id, "value"))
    if deal.expected_close_date:
        days = (deal.expected_close_date - date.today()).days
        add(SourceKind.RECORD,
            "expected_close_date = %s (%d days from today)" % (deal.expected_close_date, days),
            _ref("deals", deal.id, "expected_close_date"))
    if deal.last_activity_at:
        quiet = (datetime.now(deal.last_activity_at.tzinfo) - deal.last_activity_at).days
        add(SourceKind.RECORD,
            "last activity %s (%d days ago)" % (str(deal.last_activity_at)[:10], quiet),
            _ref("deals", deal.id, "last_activity_at"))

    # --- how long in this stage. Computed here, never by the model.
    changed_at = await db.scalar(
        select(func.max(DealStageHistory.changed_at)).where(
            DealStageHistory.deal_id == deal_id
        )
    )
    anchor = changed_at or deal.created_at
    if anchor:
        dwell = (datetime.now(anchor.tzinfo) - anchor).days
        add(SourceKind.RECORD,
            "in stage '%s' since %s (%d days)" % (deal.stage.value, str(anchor)[:10], dwell),
            _ref("deal_stage_history", None, "changed_at"))

    # --- who is on the deal, and whether they have actually turned up.
    # Attendance is the distinction `no_economic_buyer` turns on: listed is not
    # the same as present.
    links = (
        await db.execute(
            select(DealContact, Contact)
            .join(Contact, Contact.id == DealContact.contact_id)
            .where(DealContact.deal_id == deal_id)
        )
    ).all()
    for link, contact in links:
        attended = await db.scalar(
            select(func.count())
            .select_from(MeetingAttendee)
            .join(Meeting, Meeting.id == MeetingAttendee.meeting_id)
            .where(
                Meeting.deal_id == deal_id,
                MeetingAttendee.contact_id == contact.id,
                MeetingAttendee.attended.is_(True),
            )
        )
        add(SourceKind.RECORD,
            "%s %s -- %s, influence %s, attended %d meeting(s)"
            % (contact.first_name, contact.last_name, link.buying_role.value,
               link.influence.value, attended or 0),
            _ref("deal_contacts", link.id, "buying_role"))

    unresolved = (
        await db.execute(
            select(MeetingAttendee.raw_name, func.count())
            .join(Meeting, Meeting.id == MeetingAttendee.meeting_id)
            .where(
                Meeting.deal_id == deal_id,
                MeetingAttendee.contact_id.is_(None),
                MeetingAttendee.is_internal.is_(False),
            )
            .group_by(MeetingAttendee.raw_name)
        )
    ).all()
    for raw_name, count in unresolved:
        add(SourceKind.DERIVED,
            "%s attended %d meeting(s) and is not a tracked contact" % (raw_name, count))

    meetings = (
        await db.execute(
            select(Meeting.meeting_type, Meeting.scheduled_at, Meeting.status)
            .where(Meeting.deal_id == deal_id)
            .order_by(Meeting.scheduled_at)
        )
    ).all()
    if meetings:
        add(SourceKind.DERIVED,
            "%d meeting(s): %s" % (
                len(meetings),
                "; ".join("%s on %s" % (m[0], str(m[1])[:10]) for m in meetings)))

    for commitment in (
        await db.execute(
            select(Commitment).where(
                Commitment.deal_id == deal_id,
                Commitment.status == CommitmentStatus.PENDING,
            )
        )
    ).scalars():
        overdue = (
            " -- OVERDUE" if commitment.due_date and commitment.due_date < date.today() else ""
        )
        add(SourceKind.RECORD,
            "open commitment (%s): %s, due %s%s"
            % (commitment.owner_side.value, commitment.description,
               commitment.due_date, overdue),
            _ref("commitments", commitment.id, "status"))

    # --- the accepted facts. `pending` is excluded: an unreviewed proposal is
    # not something to build a second assertion on top of.
    for fact in (
        await db.execute(
            select(ExtractedFact)
            .where(
                ExtractedFact.deal_id == deal_id,
                ExtractedFact.status == FactStatus.ACCEPTED,
            )
            .order_by(ExtractedFact.extracted_at.desc())
            .limit(40)
        )
    ).scalars():
        add(SourceKind.DERIVED, "%s: %s" % (fact.fact_type, fact.content),
            fact_id=fact.id, prefix="f")

    # --- what is already open, so resolution can be asked for (task 7.9)
    for risk in (
        await db.execute(
            select(Risk).where(Risk.deal_id == deal_id, Risk.status == RiskStatus.OPEN)
        )
    ).scalars():
        dossier.open_risks[str(risk.id)] = risk

    # --- what a human already declined (task 7.10)
    dismissed = (
        await db.execute(
            select(
                Recommendation.dismissal_reason,
                Recommendation.action_type,
                Recommendation.decided_at,
                Risk.risk_type,
                Risk.risk_key,
            )
            .outerjoin(Risk, Risk.id == Recommendation.source_risk_id)
            .where(
                Recommendation.deal_id == deal_id,
                Recommendation.status == RecommendationStatus.DISMISSED,
            )
        )
    ).all()
    for reason, action, when, risk_type, risk_key in dismissed:
        key = (risk_type or "", risk_key or "")
        dossier.dismissals.setdefault(key, []).append((reason or "other", action, when))

    logger.info(
        "dossier.built deal=%s entries=%d open_risks=%d dismissals=%d",
        deal_id, len(dossier.entries), len(dossier.open_risks), len(dossier.dismissals),
    )
    return dossier
