"""Phase 8: generate and persist one pre-meeting brief per meeting."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client
from app.ai.prompts import get as get_prompt
from app.ai.schemas import BriefOut
from app.models import Commitment, Contact, Deal, DealContact, Meeting, MeetingBrief, Risk
from app.models.enums import CommitmentStatus, RiskStatus


def _lines(values) -> str:
    rendered = [str(value) for value in values if value]
    return "\n".join("- %s" % value for value in rendered) or "(none)"


async def generate(
    db: AsyncSession,
    meeting: Meeting,
    *,
    force: bool = False,
    budget: Optional[client.RunBudget] = None,
) -> MeetingBrief:
    """Return the stored brief, calling the model only when absent or forced."""
    existing = await db.scalar(
        select(MeetingBrief).where(MeetingBrief.meeting_id == meeting.id)
    )
    if existing is not None and not force:
        return existing

    deal = await db.get(Deal, meeting.deal_id)
    risks = list((await db.scalars(
        select(Risk)
        .where(Risk.deal_id == meeting.deal_id, Risk.status == RiskStatus.OPEN)
        .order_by(Risk.severity.desc(), Risk.last_seen_at.desc())
    )).all())
    commitments = list((await db.scalars(
        select(Commitment)
        .where(
            Commitment.deal_id == meeting.deal_id,
            Commitment.status == CommitmentStatus.PENDING,
        )
        .order_by(Commitment.due_date.asc().nulls_last())
    )).all())
    summaries = list((await db.execute(
        select(Meeting.title, Meeting.scheduled_at, Meeting.summary)
        .where(
            Meeting.deal_id == meeting.deal_id,
            Meeting.id != meeting.id,
            Meeting.summary.is_not(None),
        )
        .order_by(Meeting.scheduled_at.desc().nulls_last())
        .limit(5)
    )).all())
    stakeholders = list((await db.execute(
        select(Contact.first_name, Contact.last_name, Contact.title,
               DealContact.buying_role, DealContact.influence)
        .join(DealContact, DealContact.contact_id == Contact.id)
        .where(DealContact.deal_id == meeting.deal_id)
        .order_by(DealContact.is_primary.desc(), Contact.last_name)
    )).all())

    prompt = get_prompt("brief")
    answer, run = await client.structured(
        BriefOut,
        prompt.messages(
            meeting="%s; type=%s; scheduled=%s" % (
                meeting.title, meeting.meeting_type, meeting.scheduled_at,
            ),
            deal="%s; stage=%s; value=%s %s; expected close=%s" % (
                deal.name, deal.stage, deal.value, deal.currency, deal.expected_close_date,
            ),
            risks=_lines("[%s] %s: %s" % (r.severity, r.title, r.description or "") for r in risks),
            commitments=_lines("%s (owner=%s, due=%s)" % (
                c.description, c.owner_name or c.owner_side, c.due_date,
            ) for c in commitments),
            summaries=_lines("%s (%s): %s" % row for row in summaries),
            stakeholders=_lines("%s %s — %s; role=%s; influence=%s" % row for row in stakeholders),
        ),
        task="brief",
        prompt_version=prompt.version,
        reasoning_effort="medium",
        budget=budget,
    )

    if existing is not None:
        await db.delete(existing)
        await db.flush()

    brief = MeetingBrief(
        meeting_id=meeting.id,
        objectives=answer.objectives,
        key_risks=answer.key_risks,
        recommended_questions=answer.recommended_questions,
        context_summary=answer.context_summary.strip(),
        model=run.model,
    )
    db.add(brief)
    await db.flush()
    return brief
