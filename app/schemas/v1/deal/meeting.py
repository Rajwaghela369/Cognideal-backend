"""The /deals/{deal_id}/meetings contract, both directions.

Nested under the deal throughout: meetings are read from the deal detail page
and nowhere else, so there is no cross-deal collection. `meetings.deal_id` is
NOT NULL and there is no `account_id` column -- a meeting belongs to exactly
one deal, and "meetings for this account" is a join through `deals`.

Columns absent from every request model below, and why:

    id                      server-generated
    deal_id                 in the path
    analysis_status         the analyzer owns it; POST .../analysis moves it
    analyzed_at             written when analysis completes
    transcript_document_id  set by the document upload flow, not by a client
                            PATCHing a foreign key
    created_at/updated_at   server
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.enums import AnalysisStatus, MeetingStatus, MeetingType, Sentiment
from app.schemas.common import ORM, WRITE

# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class MeetingListItem(BaseModel):
    """One row of the meeting track on the deal page."""

    model_config = ORM

    id: uuid.UUID
    title: str
    meeting_type: str
    status: str
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    sentiment: Optional[str] = None
    analysis_status: str
    analyzed_at: Optional[datetime] = None
    # Derived, not stored: whether a transcript document is attached, and how
    # many people were on the call.
    has_transcript: bool = False
    attendee_count: int = 0


class MeetingDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    deal_id: uuid.UUID
    deal_name: str
    account_id: uuid.UUID
    account_name: str
    title: str
    meeting_type: str
    status: str
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    analysis_status: str
    analyzed_at: Optional[datetime] = None
    transcript_document_id: Optional[uuid.UUID] = None
    has_transcript: bool = False
    attendee_count: int = 0
    created_at: datetime
    updated_at: datetime


class MeetingAnalysis(BaseModel):
    """What the Meeting Analyzer screen polls.

    A narrow projection rather than the whole meeting, so the polling loop is
    cheap and the screen's contract does not shift when unrelated meeting
    fields change.
    """

    model_config = ORM

    meeting_id: uuid.UUID
    analysis_status: str
    analyzed_at: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    has_transcript: bool = False
    #: Which degradable stages failed, and why. Set on a `complete` run as well
    #: as a failed one -- a meeting whose summary stage died still has its
    #: facts, and the screen should be able to say so rather than implying the
    #: analysis was clean. Written by stages.finalize.
    analysis_error: Optional[str] = None
    #: What the run produced, for the completion notice. Only filled in once
    #: the status is `complete`, so polling a queued run stays one row read.
    #: `facts_count` is this meeting's extracted facts, `evidence_count` the
    #: citations attached to them, and `open_risks_count` the deal's open
    #: risks after the run (detection reads the whole deal, not one meeting).
    facts_count: Optional[int] = None
    evidence_count: Optional[int] = None
    open_risks_count: Optional[int] = None


class MeetingBriefResponse(BaseModel):
    model_config = ORM

    id: uuid.UUID
    meeting_id: uuid.UUID
    objectives: Optional[list] = None
    context_summary: Optional[str] = None
    key_risks: Optional[list] = None
    recommended_questions: Optional[list] = None
    model: Optional[str] = None
    generated_at: datetime


class BriefRequest(BaseModel):
    model_config = WRITE

    force: bool = False


class MeetingAttendee(BaseModel):
    """One person on one call.

    `raw_name` is the name as it appeared -- in the invite, or as a transcript
    speaker label. `contact_id` is the optional *resolution* of that name to a
    known person, which is why `resolved` is worth returning explicitly: an
    unresolved attendee is the raw signal for "missing stakeholder".
    """

    model_config = ORM

    id: uuid.UUID
    raw_name: str
    contact_id: Optional[uuid.UUID] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_title: Optional[str] = None
    is_internal: bool
    # Only meaningful once the meeting is `completed`. On a scheduled meeting
    # this is really "invited", which the column cannot distinguish.
    attended: bool
    resolved: bool = False


class DealParticipant(BaseModel):
    """One person, rolled up across every meeting on the deal.

    Not the same list as `/stakeholders`, and the difference is the point:

    *   attended but `is_stakeholder=False` -- someone is influencing this deal
        and nobody is tracking them
    *   `is_stakeholder=True` with `meetings_attended=0` -- a stakeholder who
        has never turned up, which for an economic buyer is the
        NO_ECONOMIC_BUYER risk

    Neither is visible from the stakeholder list or from one meeting's
    attendees; only the roll-up shows the gap between the two.
    """

    model_config = ORM

    contact_id: Optional[uuid.UUID] = None
    name: str
    email: Optional[str] = None
    title: Optional[str] = None
    is_internal: bool = False
    resolved: bool = False
    is_stakeholder: bool = False
    buying_role: Optional[str] = None
    influence: Optional[str] = None
    meetings_attended: int = 0
    last_seen_at: Optional[datetime] = None


# --------------------------------------------------------------------------
# Requests -- bodies and query parameters
# --------------------------------------------------------------------------


class MeetingCreate(BaseModel):
    model_config = WRITE

    title: str = Field(min_length=1, max_length=255)
    meeting_type: MeetingType = MeetingType.OTHER
    status: MeetingStatus = MeetingStatus.SCHEDULED
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[Sentiment] = None

    @model_validator(mode="after")
    def _ended_after_started(self) -> "MeetingCreate":
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("ended_at cannot be earlier than started_at")
        return self


class MeetingUpdate(BaseModel):
    """Partial update. Read with ``model_dump(exclude_unset=True)``.

    `summary` and `sentiment` are writable by a human even though the analyzer
    also produces them -- a salesperson typing their own notes is legitimate.
    The cost is that `meetings` has no `origin` column, so once both can write
    there is no way to answer "did a person or the model write this summary?".
    Noted rather than fixed: adding `origin`/`confidence` here is a schema
    change that should wait until the analyzer actually exists.
    """

    model_config = WRITE

    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    meeting_type: Optional[MeetingType] = None
    status: Optional[MeetingStatus] = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[Sentiment] = None


class MeetingFilters(BaseModel):
    """Query parameters of GET /deals/{deal_id}/meetings.

    No pagination: this is one deal's meeting track, which is tens of rows at
    worst. No `deal_id` either -- it is in the path.
    """

    model_config = WRITE

    status: List[MeetingStatus] = Field(default_factory=list)
    meeting_type: List[MeetingType] = Field(default_factory=list)
    analysis_status: List[AnalysisStatus] = Field(default_factory=list)
    upcoming: Optional[bool] = Field(
        default=None,
        description="Scheduled and still in the future -- the prep queue. "
        "false gives everything already held or cancelled.",
    )
    has_transcript: Optional[bool] = None
    q: Optional[str] = Field(default=None, description="Matches title or summary")
    # Newest first: the track shows recent meetings at the top. Pass
    # `scheduled_at` with upcoming=true for the prep queue's natural order.
    sort: str = "-scheduled_at"


class AnalysisRequest(BaseModel):
    """Inputs for the Meeting Analyzer.

    NOT YET CONSUMED. POST .../analysis sets analysis_status to `queued` and
    returns; no worker picks it up, so a meeting stays queued until the
    analyzer is built. The contract is here so the screen can be built against
    its final shape.
    """

    model_config = WRITE

    transcript_document_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Defaults to the meeting's own transcript_document_id. "
        "Documents routes do not exist yet, so today this is the only way to "
        "point the analyzer at one.",
    )
    focus: List[str] = Field(
        default_factory=list,
        description="Optional hints -- 'pricing', 'security', a competitor name",
    )
    force: bool = Field(
        default=False,
        description="Re-run even if analysis_status is already `complete`",
    )


class AttendeeCreate(BaseModel):
    """`raw_name` is required and `contact_id` optional, never the reverse.

    A transcript speaker is frequently not in `contacts` yet, and those are
    precisely the people worth surfacing. Snapshotting the name also means the
    row degrades to "unresolved attendee" rather than becoming anonymous when
    a contact is deleted.
    """

    model_config = WRITE

    raw_name: str = Field(min_length=1, max_length=200)
    contact_id: Optional[uuid.UUID] = None
    is_internal: bool = False
    attended: bool = True


class AttendeeUpdate(BaseModel):
    model_config = WRITE

    raw_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    contact_id: Optional[uuid.UUID] = None
    is_internal: Optional[bool] = None
    attended: Optional[bool] = None


class ContactIdentity(BaseModel):
    """A new person, created from an attendee.

    No `account_id`: it is derived from attendee -> meeting -> deal -> account.
    That is not just convenience -- it makes a contact on the wrong account
    structurally impossible rather than something the service has to check.
    """

    model_config = WRITE

    first_name: str = Field(min_length=1, max_length=120)
    last_name: str = Field(min_length=1, max_length=120)
    email: Optional[str] = Field(default=None, max_length=320)
    title: Optional[str] = Field(default=None, max_length=200)
    phone: Optional[str] = Field(default=None, max_length=50)


class StakeholderIdentity(BaseModel):
    """Selling metadata, if the resolved person should also become a
    stakeholder on the deal."""

    model_config = WRITE

    buying_role: Optional[str] = None
    influence: Optional[str] = None
    sentiment: Optional[str] = None
    is_primary: bool = False
    notes: Optional[str] = None


class AttendeeResolve(BaseModel):
    """Turn "Dana (procurement)" into a tracked person, in one transaction.

    Exactly one of `contact_id` (link someone who already exists) or `contact`
    (create them first). `stakeholder` optionally adds the deal_contacts row in
    the same call -- three writes for one human action, so it cannot half-fail
    with a contact created and no stakeholder link.

    This is the same shape as extracted_facts promotion: a raw signal, a human
    approving it, and a Layer A row created as a result. The attendee row *is*
    the staging record here, so no fact row is involved.
    """

    model_config = WRITE

    contact_id: Optional[uuid.UUID] = None
    contact: Optional[ContactIdentity] = None
    stakeholder: Optional[StakeholderIdentity] = None
    force: bool = Field(
        default=False, description="Re-point an attendee that is already resolved"
    )

    @model_validator(mode="after")
    def _exactly_one_identity(self) -> "AttendeeResolve":
        if (self.contact_id is None) == (self.contact is None):
            raise ValueError(
                "provide exactly one of contact_id (link an existing person) "
                "or contact (create a new one)"
            )
        return self


class AttendeeFilters(BaseModel):
    model_config = WRITE

    resolved: Optional[bool] = Field(
        default=None,
        description="false lists attendees whose name maps to no known contact "
        "-- the missing-stakeholder signal",
    )
    is_internal: Optional[bool] = None
    attended: Optional[bool] = None


class ParticipantFilters(BaseModel):
    model_config = WRITE

    is_stakeholder: Optional[bool] = Field(
        default=None,
        description="false lists people who turned up but are not tracked on "
        "the deal",
    )
    resolved: Optional[bool] = None
    is_internal: Optional[bool] = Field(
        default=False,
        description="Defaults to false: your own people are not deal "
        "participants in the sense this list means",
    )
