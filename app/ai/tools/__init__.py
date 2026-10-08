"""Evidence-carrying tools for the Phase 9 chat agent.

Eight read tools and exactly one write: `propose_task`, which creates a
`recommendation` with `status='suggested'` and never a `task`. `README.md` §6
allowed that one "at most, after the read path is trusted", and the read path
now is -- every result carries an evidence handle, scope comes from
`chat_sessions.deal_id` in Python, and answers are Gate 0 checked.

The distinction it preserves is the product: `risk -> recommendation -> [human
accepts] -> task`. An agent that could write a `task` would be doing the
deciding; one that can only suggest is proposing. There is no update tool and no
delete tool, and `propose_task` cannot touch any row that already exists.
"""

import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client
from app.models import (
    Commitment,
    Contact,
    Deal,
    DealContact,
    DealStageHistory,
    Document,
    DocumentChunk,
    Meeting,
    Recommendation,
    Risk,
    Task,
)
from app.models.enums import (
    ActionType,
    ClaimType,
    CommitmentStatus,
    DealStage,
    Origin,
    RecommendationStatus,
    RiskStatus,
)
from app.services import claims as claims_service

#: Distinguishes a chat-proposed recommendation from a detector-proposed one in
#: `recommendations.detector_version`. Bump it when the tool's contract changes,
#: for the same reason every prompt carries a version.
PROPOSE_VERSION = "chat-propose-1"

logger = logging.getLogger("cognideal.ai.tools")

#: Deals one `search_deals` call returns. Every field of every deal is its own
#: evidence entry, and the whole result stays in the thread for every later
#: model step that turn -- so this bounds tokens, not just rows.
SEARCH_DEALS_LIMIT = 20

# The argument schemas are sent to OpenAI with `strict: true` (see
# `chat.stream_turn`), which constrains decoding to the schema. Strict mode has
# no optional fields: every field is required and absence is an explicit null,
# so nothing here has a default. Length rules are checked in Python rather than
# declared as `minLength`/`maxLength`, which strict mode does not accept.


class DealArgs(BaseModel):
    deal_id: Optional[str] = Field(
        description="Deal UUID. null in a deal-scoped chat, required in a global one."
    )


class SearchDealsArgs(BaseModel):
    name: Optional[str] = Field(description="Part of the deal name, or null")
    stage: Optional[DealStage] = Field(description="Exact pipeline stage, or null")
    stale_days: Optional[int] = Field(
        description="Only deals with no activity in this many days, or null"
    )
    value_min: Optional[float] = Field(description="Minimum deal value, or null")


class SearchDocumentsArgs(DealArgs):
    query: str = Field(description="Exact text to find, matched case-insensitively")


class ProposeTaskArgs(DealArgs):
    """No `status`, no `priority`, no ids of existing rows. The model supplies
    prose and nothing structural."""

    title: str = Field(description="The action to take, imperative, 3-200 characters")
    rationale: str = Field(description="Why, in one or two sentences")


def _logged(name: str, coroutine):
    """Log every tool call with its arguments, size of result and duration.

    The one place a chat turn's database work is visible: without it a slow or
    failing turn shows only the model's side in the log.
    """

    async def run(**kwargs):
        args = {k: v for k, v in kwargs.items() if v is not None}
        shown = repr(args)
        if len(shown) > 200:
            shown = shown[:200] + "..."
        started = time.monotonic()
        try:
            result = await coroutine(**kwargs)
        except Exception as exc:
            logger.warning(
                "chat.tool_failed tool=%s args=%s seconds=%.2f error=%r",
                name, shown, time.monotonic() - started, exc,
            )
            raise
        logger.info(
            "chat.tool tool=%s args=%s seconds=%.2f result_chars=%d",
            name, shown, time.monotonic() - started, len(result),
        )
        return result

    run.__name__ = name
    return run


def _plain(value: Any) -> Any:
    """``DealStage.NEGOTIATION`` -> ``negotiation``, as the snippet stores it."""
    return value.value if isinstance(value, Enum) else value


class ToolRegistry:
    """Build tools bound to one DB session and one immutable chat scope."""

    def __init__(
        self,
        db: AsyncSession,
        scoped_deal_id: Optional[uuid.UUID],
        handle_offset: int = 0,
    ) -> None:
        self.db = db
        self.scoped_deal_id = scoped_deal_id
        self.evidence: Dict[str, Dict[str, Any]] = {}
        # Where this turn's handles start. A registry is per-turn, so without an
        # offset every turn reissues `e1` -- fine while the agent had no memory
        # of earlier turns, and wrong now that the checkpointer replays earlier
        # tool results into the same context. Two live `e1`s meaning different
        # citations is how a claim gets attached to the wrong evidence, which is
        # the one failure the evidence spine exists to prevent.
        #
        # Defaults to 0, so a first turn still starts at `e1` and the handles a
        # single-turn caller sees are unchanged.
        self._next = handle_offset

    def _handle(self, citation: Dict[str, Any]) -> str:
        self._next += 1
        handle = "e%d" % self._next
        self.evidence[handle] = citation
        return handle

    def _record(
        self, table: str, row_id, field: str, snippet: Any, deal_id: uuid.UUID,
    ) -> Dict[str, Any]:
        return {
            "source_kind": "record",
            "deal_id": str(deal_id),
            "record_ref": {"table": table, "id": str(row_id), "field": field},
            "snippet": str(_plain(snippet)),
        }

    def _entry(self, text: str, citation: Dict[str, Any]) -> Dict[str, Any]:
        return {"handle": self._handle(citation), "content": text}

    def _deal(self, requested: Optional[str]) -> uuid.UUID:
        """Apply deal scope in Python, never by trusting a model argument."""
        if self.scoped_deal_id is not None:
            if requested is not None:
                try:
                    requested_id = uuid.UUID(requested)
                except (TypeError, ValueError) as exc:
                    raise ToolException("deal_id must be a UUID") from exc
                if requested_id != self.scoped_deal_id:
                    raise ToolException("deal-scoped session cannot access another deal")
            return self.scoped_deal_id
        if requested is None:
            raise ToolException("deal_id is required in a global session")
        try:
            return uuid.UUID(requested)
        except (TypeError, ValueError) as exc:
            raise ToolException("deal_id must be a UUID") from exc

    @staticmethod
    def _json(entries, **extra) -> str:
        return json.dumps({"entries": entries, **extra}, default=str)

    async def search_deals(self, name=None, stage=None, stale_days=None, value_min=None) -> str:
        stmt = select(Deal).order_by(Deal.updated_at.desc()).limit(SEARCH_DEALS_LIMIT)
        if self.scoped_deal_id is not None:
            stmt = stmt.where(Deal.id == self.scoped_deal_id)
        if name:
            pattern = "%%%s%%" % name.replace("%", "\\%").replace("_", "\\_")
            stmt = stmt.where(Deal.name.ilike(pattern, escape="\\"))
        if stage:
            stmt = stmt.where(Deal.stage == stage)
        if stale_days is not None:
            cutoff = func.now() - func.make_interval(0, 0, 0, stale_days)
            stmt = stmt.where(or_(Deal.last_activity_at.is_(None), Deal.last_activity_at < cutoff))
        if value_min is not None:
            stmt = stmt.where(Deal.value >= value_min)
        entries = []
        for deal in (await self.db.scalars(stmt)).all():
            for field in ("name", "stage", "value", "last_activity_at"):
                value = getattr(deal, field)
                if value is not None:
                    entries.append(self._entry(
                        "deal %s: %s = %s" % (deal.id, field, _plain(value)),
                        self._record("deals", deal.id, field, value, deal.id),
                    ))
        return self._json(entries)

    async def get_deal_snapshot(self, deal_id=None) -> str:
        did = self._deal(deal_id)
        deal = await self.db.get(Deal, did)
        if deal is None:
            return self._json([])
        fields = ("name", "stage", "value", "expected_close_date", "last_activity_at")
        return self._json([
            self._entry("%s = %s" % (field, _plain(getattr(deal, field))),
                        self._record("deals", deal.id, field, getattr(deal, field), did))
            for field in fields if getattr(deal, field) is not None
        ])

    async def list_risks(self, deal_id=None) -> str:
        did = self._deal(deal_id)
        rows = (await self.db.scalars(
            select(Risk).where(Risk.deal_id == did, Risk.status == RiskStatus.OPEN)
            .order_by(Risk.last_seen_at.desc())
        )).all()
        entries = []
        uncited = []
        for risk in rows:
            evidence_rows = await claims_service.evidence_for(
                self.db, ClaimType.RISK, risk.id
            )
            if not evidence_rows:
                # Still reported -- an open risk the user can see in the app
                # should not be invisible to the agent -- but without a handle:
                # a risk is itself a conclusion, so it cannot be its own
                # evidence.
                uncited.append({
                    "severity": _plain(risk.severity),
                    "title": risk.title,
                    "note": "No evidence on file. Not citable; say it is unsupported.",
                })
                continue
            for evidence in evidence_rows:
                entries.append(self._entry(
                    "[%s] %s: %s — source: %s" % (
                        _plain(risk.severity), risk.title, risk.description or "", evidence.snippet,
                    ),
                    {
                        "source_kind": evidence.source_kind,
                        "deal_id": str(did),
                        "document_id": str(evidence.document_id) if evidence.document_id else None,
                        "chunk_id": str(evidence.chunk_id) if evidence.chunk_id else None,
                        "record_ref": evidence.record_ref,
                        "snippet": evidence.snippet,
                        "char_start": evidence.char_start,
                        "char_end": evidence.char_end,
                        "speaker": evidence.speaker,
                        "occurred_at": evidence.occurred_at,
                    },
                ))
        return self._json(entries, uncited_risks=uncited) if uncited else self._json(entries)

    async def list_commitments(self, deal_id=None) -> str:
        did = self._deal(deal_id)
        rows = (await self.db.scalars(
            select(Commitment).where(
                Commitment.deal_id == did,
                Commitment.status == CommitmentStatus.PENDING,
            ).order_by(Commitment.due_date.asc().nulls_last())
        )).all()
        entries = []
        for commitment in rows:
            for field in ("description", "owner_side", "owner_name", "due_date"):
                value = getattr(commitment, field)
                if value is not None:
                    entries.append(self._entry(
                        "commitment %s: %s = %s" % (commitment.id, field, _plain(value)),
                        self._record("commitments", commitment.id, field, value, did),
                    ))
        return self._json(entries)

    async def list_tasks(self, deal_id=None) -> str:
        did = self._deal(deal_id)
        rows = (await self.db.scalars(
            select(Task).where(Task.deal_id == did).order_by(Task.due_date.asc().nulls_last())
        )).all()
        entries = []
        for task in rows:
            for field in ("title", "status", "due_date", "priority"):
                value = getattr(task, field)
                if value is not None:
                    entries.append(self._entry(
                        "task %s: %s = %s" % (task.id, field, _plain(value)),
                        self._record("tasks", task.id, field, value, did),
                    ))
        return self._json(entries)

    async def get_timeline(self, deal_id=None) -> str:
        did = self._deal(deal_id)
        entries = []
        stages = (await self.db.scalars(
            select(DealStageHistory).where(DealStageHistory.deal_id == did)
            .order_by(DealStageHistory.changed_at.desc()).limit(20)
        )).all()
        for row in stages:
            for field in ("from_stage", "to_stage", "changed_at"):
                value = getattr(row, field)
                if value is not None:
                    entries.append(self._entry(
                        "stage event %s: %s = %s" % (row.id, field, _plain(value)),
                        self._record("deal_stage_history", row.id, field, value, did),
                    ))
        meetings = (await self.db.scalars(
            select(Meeting).where(Meeting.deal_id == did)
            .order_by(Meeting.scheduled_at.desc().nulls_last()).limit(20)
        )).all()
        for row in meetings:
            for field in ("title", "status", "scheduled_at"):
                value = getattr(row, field)
                if value is not None:
                    entries.append(self._entry(
                        "meeting %s: %s = %s" % (row.id, field, _plain(value)),
                        self._record("meetings", row.id, field, value, did),
                    ))
        return self._json(entries)

    async def get_stakeholder_map(self, deal_id=None) -> str:
        did = self._deal(deal_id)
        rows = (await self.db.execute(
            select(DealContact, Contact).join(Contact, Contact.id == DealContact.contact_id)
            .where(DealContact.deal_id == did)
        )).all()
        entries = []
        for link, contact in rows:
            for field in ("first_name", "last_name", "title"):
                value = getattr(contact, field)
                if value is not None:
                    entries.append(self._entry(
                        "contact %s: %s = %s" % (contact.id, field, _plain(value)),
                        self._record("contacts", contact.id, field, value, did),
                    ))
            for field in ("buying_role", "influence"):
                value = getattr(link, field)
                # An unset role is absence, not a fact; citing it would put a
                # handle on the string "None".
                if value is None:
                    continue
                entries.append(self._entry(
                    "deal contact %s: %s = %s" % (link.id, field, _plain(value)),
                    self._record("deal_contacts", link.id, field, value, did),
                ))
        return self._json(entries)

    async def search_documents(self, query, deal_id=None) -> str:
        did = self._deal(deal_id)
        query = (query or "").strip()
        if not query:
            raise ToolException("query must not be empty")
        pattern = "%%%s%%" % query.replace("%", "\\%").replace("_", "\\_")
        rows = (await self.db.execute(
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.deal_id == did, DocumentChunk.content.ilike(pattern, escape="\\"))
            .order_by(Document.occurred_at.desc(), DocumentChunk.chunk_index)
            .limit(10)
        )).all()
        return self._json([
            self._entry(
                "%s (%s): %s" % (doc.title, doc.occurred_at, chunk.content),
                {
                    "source_kind": "document",
                    "deal_id": str(did),
                    "document_id": str(doc.id),
                    "chunk_id": str(chunk.id),
                    "snippet": chunk.content,
                    "char_start": 0,
                    "char_end": len(chunk.content),
                    "speaker": (chunk.chunk_metadata or {}).get("speaker"),
                    "occurred_at": doc.occurred_at,
                },
            ) for chunk, doc in rows
        ])

    async def propose_task(self, title, rationale, deal_id=None) -> str:
        """The one write: a suggestion for a human to accept or dismiss.

        `action_type='follow_up_email'` is wrong for most of what an agent will
        propose, so the value is deliberately the generic one -- `schedule_meeting`
        and friends each assert something specific about the action that a free-text
        suggestion cannot support. `internal_escalation` was the other candidate and
        asserts more.

        `source_risk_id` stays null: this is proactive advice, which is exactly the
        case the one-live-suggestion index excludes, so a chatty session could file
        several. That is accepted -- they are visible, dismissible, and attributable
        by `origin='ai'` plus `detector_version='chat-propose-1'`, which is how a
        reviewer tells agent suggestions from detector ones.
        """
        deal = self._deal(deal_id)
        title, rationale = (title or "").strip(), (rationale or "").strip()
        if not 3 <= len(title) <= 200:
            raise ToolException("title must be 3-200 characters")
        if len(rationale) < 3:
            raise ToolException("rationale must be at least 3 characters")
        if await self.db.get(Deal, deal) is None:
            raise ToolException("deal does not exist")

        recommendation = Recommendation(
            deal_id=deal,
            title=title,
            description=title,
            rationale=rationale,
            action_type=ActionType.INTERNAL_ESCALATION.value,
            status=RecommendationStatus.SUGGESTED,
            origin=Origin.AI,
            model=client.model_for(client.ROLE_PRIMARY),
            detector_version=PROPOSE_VERSION,
        )
        self.db.add(recommendation)
        await self.db.flush()
        recommendation_id = recommendation.id
        # Committed now, not with the turn. The tool result below goes into
        # the checkpointed thread straight away; if the turn later failed and
        # rolled back, the agent would go on remembering a suggestion that
        # does not exist.
        await self.db.commit()
        return json.dumps({
            "created": "recommendation",
            "id": str(recommendation_id),
            "status": "suggested",
            "note": "Suggested only. A person must accept it before it becomes a task.",
        })

    def langchain_tools(self):
        specs = (
            ("search_deals", self.search_deals, SearchDealsArgs,
             "Find deals by name, stage, staleness or value. Returns at most %d "
             "deals, most recently updated first." % SEARCH_DEALS_LIMIT),
            ("get_deal_snapshot", self.get_deal_snapshot, DealArgs, "Get the current fields for one deal."),
            ("list_risks", self.list_risks, DealArgs, "List open risks for a deal."),
            ("list_commitments", self.list_commitments, DealArgs, "List pending commitments for a deal."),
            ("list_tasks", self.list_tasks, DealArgs, "List tasks for a deal."),
            ("get_timeline", self.get_timeline, DealArgs, "List recent stage and meeting events."),
            ("get_stakeholder_map", self.get_stakeholder_map, DealArgs, "List stakeholders and buying roles."),
            ("search_documents", self.search_documents, SearchDocumentsArgs, "Search exact text in this deal's documents."),
            ("propose_task", self.propose_task, ProposeTaskArgs,
             "Suggest an action for the user to accept or dismiss. Creates a "
             "suggestion only -- it does NOT create a task or change any record."),
        )
        return [
            StructuredTool.from_function(
                coroutine=_logged(name, coroutine), name=name, description=description,
                args_schema=args_schema,
            )
            for name, coroutine, args_schema, description in specs
        ]
