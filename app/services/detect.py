"""Deterministic risk detection.

Four of the ten `risk_type` values need no model at all -- they are joins over
tables that already exist. That matters more than it sounds: it means a cited,
working risk panel ships before any agent does, and it means these four can
never hallucinate.

Each detected risk gets `evidence` rows with `source_kind='record'` pointing at
the field that triggered it, which is the case README section 4 makes for why
evidence cannot only be document quotes:

    Most risk detection reasons over structured state, not over quotes --
    "close date is 21 days out", "stage has not moved in 58 days", "no
    economic buyer has attended a meeting". If evidence could only point at
    document chunks, every one of those risks would render uncited and look
    like a hunch.

Detection is an **upsert**, never an insert. `uq_risks_deal_id_risk_type_open`
exists precisely so a re-run bumps `last_seen_at` rather than adding a fifth
"single-threaded"; the risk panel otherwise fills with duplicates inside a week.

Each risk also produces one recommendation, from the RiskType -> ActionType
mapping below. The two enums line up one-for-one, which is not a coincidence --
they were designed as a pair. Writing both in one pass means the risk card can
always be assembled, and that two separate triggers cannot disagree.
"""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Deal,
    DealContact,
    DealStageHistory,
    Meeting,
    MeetingAttendee,
    Recommendation,
    Risk,
)
from app.models.enums import (
    ActionType,
    ClaimType,
    Priority,
    RecommendationStatus,
    RiskStatus,
    RiskType,
    Severity,
    SourceKind,
)
from app.queries import CLOSE_DATE_WARNING_DAYS, CLOSED_STAGES
from app.services import claims as claims_service

# What to do about each kind of risk. The two vocabularies were built to line
# up; this is that pairing made explicit.
ACTION_FOR_RISK: Dict[str, str] = {
    RiskType.NO_ECONOMIC_BUYER.value: ActionType.ENGAGE_STAKEHOLDER.value,
    RiskType.SINGLE_THREADED.value: ActionType.ENGAGE_STAKEHOLDER.value,
    RiskType.STALLED_STAGE.value: ActionType.SCHEDULE_MEETING.value,
    RiskType.CLOSE_DATE_AT_RISK.value: ActionType.UPDATE_CLOSE_DATE.value,
    RiskType.UNRESOLVED_OBJECTION.value: ActionType.ADDRESS_OBJECTION.value,
    RiskType.SECURITY_REVIEW_PENDING.value: ActionType.SEND_DOCUMENT.value,
    RiskType.BUDGET_UNCONFIRMED.value: ActionType.ENGAGE_STAKEHOLDER.value,
    RiskType.COMPETITOR_PRESSURE.value: ActionType.INTERNAL_ESCALATION.value,
    RiskType.MISSED_COMMITMENT.value: ActionType.FOLLOW_UP_EMAIL.value,
    RiskType.GONE_QUIET.value: ActionType.FOLLOW_UP_EMAIL.value,
}


@dataclass
class Finding:
    """One detected risk, with the citations that justify it and the action it
    implies."""

    risk_type: str
    title: str
    description: str
    severity: Severity
    action_title: str
    rationale: str
    priority: Priority = Priority.MEDIUM
    # (source_kind, snippet, record_ref)
    citations: List[tuple] = field(default_factory=list)


def _ref(table: str, row_id, column: str) -> dict:
    """A record_ref Gate 0 can re-resolve: which table, which row, which field."""
    return {"table": table, "id": str(row_id) if row_id else None, "field": column}


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


async def _no_economic_buyer(db: AsyncSession, deal: Deal) -> Optional[Finding]:
    """Nobody who can say yes has been in the room.

    Deliberately not "is there an economic_buyer stakeholder" -- a name on a
    list is not engagement. The question is whether one has *attended*, which
    is what makes this table's shape the product thesis rather than a CRM field.
    """
    listed = (
        await db.execute(
            select(DealContact.contact_id, DealContact.id)
            .where(
                DealContact.deal_id == deal.id,
                DealContact.buying_role == "economic_buyer",
            )
        )
    ).all()

    attended_ids = set(
        (
            await db.scalars(
                select(MeetingAttendee.contact_id)
                .join(Meeting, Meeting.id == MeetingAttendee.meeting_id)
                .where(
                    Meeting.deal_id == deal.id,
                    MeetingAttendee.attended.is_(True),
                    MeetingAttendee.contact_id.is_not(None),
                )
            )
        ).all()
    )

    if any(contact_id in attended_ids for contact_id, _ in listed):
        return None

    if listed:
        title = "Economic buyer has never attended a meeting"
        description = (
            f"{len(listed)} stakeholder(s) are marked economic_buyer on this "
            f"deal, and none has attended any meeting."
        )
        # Cite the listed-but-absent stakeholder row.
        citations = [
            (SourceKind.RECORD, "economic_buyer", _ref("deal_contacts", listed[0][1], "buying_role"))
        ]
    else:
        title = "No economic buyer identified"
        description = "No stakeholder on this deal is marked as the economic buyer."
        # Nothing exists to point at. This is what SourceKind.DERIVED is for --
        # an assertion about the *absence* of rows cannot cite one.
        citations = [
            (SourceKind.DERIVED, "0 stakeholders with buying_role='economic_buyer'",
             _ref("deal_contacts", None, "buying_role"))
        ]

    return Finding(
        risk_type=RiskType.NO_ECONOMIC_BUYER.value,
        title=title,
        description=description,
        severity=Severity.HIGH,
        action_title="Get the economic buyer into a meeting",
        rationale=(
            "A deal without a decision-maker in the room slips at the "
            "signature stage, and by then there is no time to build the "
            "relationship."
        ),
        priority=Priority.HIGH,
        citations=citations,
    )


async def _single_threaded(db: AsyncSession, deal: Deal) -> Optional[Finding]:
    """Everything depends on one person."""
    people = list(
        (
            await db.scalars(
                select(func.distinct(MeetingAttendee.contact_id))
                .join(Meeting, Meeting.id == MeetingAttendee.meeting_id)
                .where(
                    Meeting.deal_id == deal.id,
                    MeetingAttendee.attended.is_(True),
                    MeetingAttendee.is_internal.is_(False),
                    MeetingAttendee.contact_id.is_not(None),
                )
            )
        ).all()
    )
    meetings = await db.scalar(
        select(func.count()).select_from(Meeting).where(Meeting.deal_id == deal.id)
    )
    # No meetings at all is a different problem, and flagging it as
    # single-threaded would be wrong -- there is no thread yet.
    if not meetings or len(people) != 1:
        return None

    return Finding(
        risk_type=RiskType.SINGLE_THREADED.value,
        title="Single-threaded on one contact",
        description=(
            f"Across {meetings} meeting(s), exactly one external person has "
            f"attended. If they leave or go quiet, the deal has no other route in."
        ),
        severity=Severity.MEDIUM,
        action_title="Broaden the relationship beyond one contact",
        rationale=(
            "One contact is one point of failure. A champion who changes role "
            "takes the deal's whole history with them."
        ),
        citations=[
            (SourceKind.DERIVED, f"{len(people)} distinct external attendee",
             _ref("meeting_attendees", None, "contact_id"))
        ],
    )


async def _stalled_stage(db: AsyncSession, deal: Deal) -> Optional[Finding]:
    """Talking, but not moving.

    Uses the same per-stage thresholds as the `stalled` filter on GET /deals --
    queries.STALL_THRESHOLD_DAYS -- so the panel and the pipeline list can
    never disagree about what counts as stuck.
    """
    from app.queries import STALL_THRESHOLD_DAYS

    if deal.stage.value in CLOSED_STAGES:
        return None
    threshold = STALL_THRESHOLD_DAYS.get(deal.stage)
    if threshold is None:
        return None

    row = (
        await db.execute(
            select(DealStageHistory.id, DealStageHistory.changed_at)
            .where(DealStageHistory.deal_id == deal.id)
            .order_by(DealStageHistory.changed_at.desc())
            .limit(1)
        )
    ).first()

    # Falls back to created_at: a deal with no history row has still been
    # sitting in its stage since it existed.
    changed_at = row.changed_at if row else deal.created_at
    since = (datetime.now(timezone.utc) - changed_at).days
    if since < threshold:
        return None

    return Finding(
        risk_type=RiskType.STALLED_STAGE.value,
        title=f"No stage movement in {since} days",
        description=(
            f"This deal has been in {deal.stage.value} for {since} days; the "
            f"threshold for that stage is {threshold}. Activity is not the same "
            f"as progress."
        ),
        severity=Severity.HIGH if since >= threshold * 2 else Severity.MEDIUM,
        action_title="Agree a concrete next step with a date",
        rationale=(
            "A deal that keeps having meetings without changing stage is "
            "usually missing a decision, not a meeting."
        ),
        citations=[
            (
                SourceKind.RECORD,
                changed_at.isoformat(),
                _ref("deal_stage_history", row.id if row else None, "changed_at"),
            )
        ],
    )


async def _close_date_at_risk(db: AsyncSession, deal: Deal) -> Optional[Finding]:
    """The forecast says imminent; the stage says otherwise."""
    if deal.expected_close_date is None or deal.stage.value in CLOSED_STAGES:
        return None
    if deal.stage.value == "negotiation":
        return None

    days_out = (deal.expected_close_date - date.today()).days
    if days_out > CLOSE_DATE_WARNING_DAYS:
        return None

    overdue = days_out < 0
    return Finding(
        risk_type=RiskType.CLOSE_DATE_AT_RISK.value,
        title=(
            f"Close date {'passed' if overdue else f'{days_out} days out'} "
            f"while still in {deal.stage.value}"
        ),
        description=(
            f"Expected close is {deal.expected_close_date.isoformat()} but the "
            f"deal has not reached negotiation. The forecast and the stage "
            f"disagree, and one of them is wrong."
        ),
        severity=Severity.CRITICAL if overdue else Severity.HIGH,
        action_title="Re-forecast the close date or escalate the stage",
        rationale=(
            "A close date nobody believes corrupts the whole pipeline number. "
            "Either the date moves or the deal does."
        ),
        priority=Priority.URGENT if overdue else Priority.HIGH,
        citations=[
            (
                SourceKind.RECORD,
                deal.expected_close_date.isoformat(),
                _ref("deals", deal.id, "expected_close_date"),
            )
        ],
    )


RULES = (_no_economic_buyer, _single_threaded, _stalled_stage, _close_date_at_risk)

# What this detector is authoritative about. A risk of any other type was put
# there by something else, and its absence from this pass says nothing about
# whether it still holds -- so auto-resolution is scoped to these four.
DETERMINISTIC_TYPES = (
    RiskType.NO_ECONOMIC_BUYER.value,
    RiskType.SINGLE_THREADED.value,
    RiskType.STALLED_STAGE.value,
    RiskType.CLOSE_DATE_AT_RISK.value,
)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


async def run(db: AsyncSession, deal: Deal) -> Dict[str, int]:
    """Detect, upsert risks and their recommendations, resolve what is fixed."""
    findings = [f for f in [await rule(db, deal) for rule in RULES] if f is not None]
    detected = {f.risk_type for f in findings}

    risks_written = 0
    recs_written = 0

    for finding in findings:
        # ON CONFLICT on the partial unique index: a re-run must bump
        # last_seen_at, not insert a duplicate.
        stmt = (
            insert(Risk)
            .values(
                deal_id=deal.id,
                risk_type=finding.risk_type,
                title=finding.title,
                description=finding.description,
                severity=finding.severity.value,
                status=RiskStatus.OPEN.value,
                confidence=1.0,  # a SQL rule is not a guess
            )
            .on_conflict_do_update(
                index_elements=[Risk.deal_id, Risk.risk_type],
                # text(), not Risk.status == "open". The index predicate is
                # `WHERE (status = 'open'::risk_status)` -- a *typed* enum
                # literal -- while the ORM comparison compiles to a bound
                # parameter. Postgres cannot prove a bound param implies the
                # typed predicate, so index inference fails with "no unique or
                # exclusion constraint matching the ON CONFLICT specification".
                # The predicate has to match what CREATE INDEX recorded.
                index_where=text("status = 'open'::risk_status"),
                set_={
                    "title": finding.title,
                    "description": finding.description,
                    "severity": finding.severity.value,
                    "last_seen_at": func.now(),
                },
            )
            .returning(Risk.id)
        )
        risk_id = await db.scalar(stmt)
        risks_written += 1

        # Citations are rewritten each run rather than accumulated: the
        # evidence is a snapshot of the state that triggered the risk, and a
        # stale snippet is what Gate 0 would flag as value_drifted.
        await claims_service.delete_claim_links(db, ClaimType.RISK, [risk_id])
        for source_kind, snippet, record_ref in finding.citations:
            await claims_service.attach_evidence(
                db,
                claim_type=ClaimType.RISK,
                claim_id=risk_id,
                deal_id=deal.id,
                source_kind=source_kind,
                snippet=snippet,
                record_ref=record_ref,
                relevance=1.0,
            )

        if await _upsert_recommendation(db, deal, risk_id, finding):
            recs_written += 1

    # A risk that no longer detects is fixed. Resolve it rather than deleting:
    # "this was true in September and was dealt with" is worth keeping.
    stale_filter = [
        Risk.deal_id == deal.id,
        Risk.status == RiskStatus.OPEN.value,
    ]
    if detected:
        stale_filter.append(Risk.risk_type.notin_(detected))
    # Only the four deterministic types: an LLM-detected risk that this pass
    # cannot see is not thereby resolved.
    stale_filter.append(Risk.risk_type.in_([f for f in DETERMINISTIC_TYPES]))
    stale = (await db.execute(select(Risk).where(*stale_filter))).scalars().all()
    for risk in stale:
        risk.status = RiskStatus.RESOLVED
        risk.resolved_at = func.now()

    return {
        "risks_detected": risks_written,
        "recommendations_written": recs_written,
        "risks_auto_resolved": len(stale),
    }


async def _upsert_recommendation(
    db: AsyncSession, deal: Deal, risk_id: uuid.UUID, finding: Finding
) -> bool:
    """One live suggestion per risk, unless a human already ruled on it.

    Keyed on the risk rather than the action type. Two risks share an
    action_type often enough -- no_economic_buyer and single_threaded both map
    to `engage_stakeholder` -- and keying on the action would silently drop the
    second one's advice.

    Returns False when nothing new was written, which includes the case that
    matters most: a recommendation the user dismissed must not reappear on the
    next run.
    """
    action_type = ACTION_FOR_RISK[finding.risk_type]

    existing = (
        await db.execute(
            select(Recommendation)
            .where(
                Recommendation.deal_id == deal.id,
                Recommendation.source_risk_id == risk_id,
                Recommendation.status.in_(
                    (
                        RecommendationStatus.SUGGESTED.value,
                        RecommendationStatus.DISMISSED.value,
                        RecommendationStatus.ACCEPTED.value,
                    )
                ),
            )
            .limit(1)
        )
    ).scalar_one_or_none()

    if existing is not None:
        if existing.status == RecommendationStatus.SUGGESTED:
            # Refresh the wording -- the risk's description may have changed
            # (the day count in "no stage movement in 60 days" moves daily).
            existing.title = finding.action_title
            existing.description = finding.description
            existing.rationale = finding.rationale
            existing.priority = finding.priority
            return False
        # Dismissed or accepted: the human has spoken, and re-suggesting would
        # undo their decision.
        return False

    recommendation = Recommendation(
        deal_id=deal.id,
        source_risk_id=risk_id,
        title=finding.action_title,
        description=finding.description,
        rationale=finding.rationale,
        action_type=action_type,
        priority=finding.priority,
        confidence=1.0,
        status=RecommendationStatus.SUGGESTED,
    )
    db.add(recommendation)
    await db.flush()

    # The recommendation inherits the risk's citations: the reason to act is
    # the same evidence as the reason to worry.
    for source_kind, snippet, record_ref in finding.citations:
        await claims_service.attach_evidence(
            db,
            claim_type=ClaimType.RECOMMENDATION,
            claim_id=recommendation.id,
            deal_id=deal.id,
            source_kind=source_kind,
            snippet=snippet,
            record_ref=record_ref,
            relevance=1.0,
        )
    return True
