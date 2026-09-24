"""Controlled vocabularies for the schema.

Two storage strategies, chosen per set (see docs/schema/README.md section 1):

*   **Native Postgres enum** -- for sets that will not churn. Type safety at the
    database level, and an invalid value is rejected by the server.
*   **text + CHECK** -- for sets that *will* churn as prompts are tuned
    (risk_type, fact_type, action_type, ...). Adding a value is a one-line
    CHECK swap instead of ALTER TYPE, which Postgres cannot run inside a
    transaction block alongside other DDL.

Every member's *value* is what reaches the database; `pg_enum` builds each
native type with `values_callable` so the lowercase values are stored rather
than the uppercase Python member names.
"""

from enum import Enum
from typing import Type

from sqlalchemy import CheckConstraint
from sqlalchemy import Enum as SAEnum


def pg_enum(enum_cls: Type[Enum], name: str) -> SAEnum:
    """A native Postgres enum type storing member values, not member names."""
    return SAEnum(
        enum_cls,
        name=name,
        values_callable=lambda cls: [member.value for member in cls],
    )


def check_in(column: str, enum_cls: Type[Enum], name: str) -> CheckConstraint:
    """A CHECK constraint restricting a text column to an enum's values."""
    allowed = ", ".join(f"'{member.value}'" for member in enum_cls)
    return CheckConstraint(f"{column} IN ({allowed})", name=name)


# --------------------------------------------------------------------------
# Native enums -- stable sets
# --------------------------------------------------------------------------


class DealStage(str, Enum):
    QUALIFICATION = "qualification"
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"
    SECURITY_AND_LEGAL = "security_and_legal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Origin(str, Enum):
    """Who authored a row. Without this, 'how much of this did the model
    write?' is unanswerable -- which is the first question asked the moment
    someone stops trusting the output."""

    USER = "user"
    AI = "ai"


class BuyingRole(str, Enum):
    CHAMPION = "champion"
    ECONOMIC_BUYER = "economic_buyer"
    TECHNICAL = "technical"
    BLOCKER = "blocker"
    INFLUENCER = "influencer"
    UNKNOWN = "unknown"


class InfluenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class MeetingStatus(str, Enum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AnalysisStatus(str, Enum):
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class TaskStatus(str, Enum):
    OPEN = "open"
    DONE = "done"
    CANCELLED = "cancelled"


class DocumentSourceType(str, Enum):
    MEETING_TRANSCRIPT = "meeting_transcript"
    EMAIL = "email"
    PROPOSAL = "proposal"
    CONTRACT = "contract"
    NOTE = "note"


class IngestStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class SourceKind(str, Enum):
    """What a piece of evidence points at.

    RECORD is not an afterthought: most risk detection reasons over structured
    state ("stage has not moved in 58 days"), not over quotes. Evidence limited
    to document spans would leave those risks uncited.
    """

    DOCUMENT = "document"
    RECORD = "record"
    DERIVED = "derived"


class ClaimType(str, Enum):
    """The five tables that hold model assertions. Used as the discriminator
    half of claim_evidence's polymorphic (claim_type, claim_id) pair."""

    FACT = "fact"
    COMMITMENT = "commitment"
    RISK = "risk"
    RECOMMENDATION = "recommendation"
    CHAT_MESSAGE = "chat_message"


class VerificationStatus(str, Enum):
    """Gate 0 outcome for a single claim -> evidence link."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    SPAN_MISSING = "span_missing"
    VALUE_DRIFTED = "value_drifted"
    STALE = "stale"


class Verdict(str, Enum):
    """Gate 1 outcome for a whole claim."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class ValidationMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HUMAN = "human"


class FactStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class OwnerSide(str, Enum):
    US = "us"
    CUSTOMER = "customer"


class CommitmentStatus(str, Enum):
    PENDING = "pending"
    MET = "met"
    MISSED = "missed"
    WAIVED = "waived"


class RiskStatus(str, Enum):
    OPEN = "open"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class RecommendationStatus(str, Enum):
    SUGGESTED = "suggested"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    COMPLETED = "completed"


class ChatScope(str, Enum):
    GLOBAL = "global"
    DEAL = "deal"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageStatus(str, Enum):
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


# --------------------------------------------------------------------------
# text + CHECK -- churny sets, expected to grow as agents are tuned
# --------------------------------------------------------------------------


class MeetingType(str, Enum):
    DISCOVERY = "discovery"
    DEMO = "demo"
    TECHNICAL_REVIEW = "technical_review"
    SECURITY_REVIEW = "security_review"
    NEGOTIATION = "negotiation"
    CHECK_IN = "check_in"
    OTHER = "other"


class ActivityType(str, Enum):
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    NOTE = "note"
    STAGE_CHANGE = "stage_change"
    DOCUMENT_UPLOAD = "document_upload"
    TASK_COMPLETED = "task_completed"


class FactType(str, Enum):
    REQUIREMENT = "requirement"
    OBJECTION = "objection"
    STAKEHOLDER = "stakeholder"
    COMMITMENT = "commitment"
    DEADLINE = "deadline"
    BUDGET = "budget"
    COMPETITOR = "competitor"
    DECISION_CRITERIA = "decision_criteria"


class RiskType(str, Enum):
    NO_ECONOMIC_BUYER = "no_economic_buyer"
    SINGLE_THREADED = "single_threaded"
    STALLED_STAGE = "stalled_stage"
    CLOSE_DATE_AT_RISK = "close_date_at_risk"
    UNRESOLVED_OBJECTION = "unresolved_objection"
    SECURITY_REVIEW_PENDING = "security_review_pending"
    BUDGET_UNCONFIRMED = "budget_unconfirmed"
    COMPETITOR_PRESSURE = "competitor_pressure"
    MISSED_COMMITMENT = "missed_commitment"
    GONE_QUIET = "gone_quiet"


class ActionType(str, Enum):
    SCHEDULE_MEETING = "schedule_meeting"
    SEND_DOCUMENT = "send_document"
    FOLLOW_UP_EMAIL = "follow_up_email"
    ENGAGE_STAKEHOLDER = "engage_stakeholder"
    UPDATE_CLOSE_DATE = "update_close_date"
    ADDRESS_OBJECTION = "address_objection"
    INTERNAL_ESCALATION = "internal_escalation"


# --------------------------------------------------------------------------
# Shared type singletons
#
# A native enum used by more than one table must be the *same* SAEnum object
# in every place, or SQLAlchemy emits CREATE TYPE once per table and the
# second one fails with "type already exists".
# --------------------------------------------------------------------------

deal_stage_enum = pg_enum(DealStage, "deal_stage")
risk_level_enum = pg_enum(RiskLevel, "risk_level")
severity_enum = pg_enum(Severity, "severity")
priority_enum = pg_enum(Priority, "priority")
origin_enum = pg_enum(Origin, "origin")
buying_role_enum = pg_enum(BuyingRole, "buying_role")
influence_level_enum = pg_enum(InfluenceLevel, "influence_level")
sentiment_enum = pg_enum(Sentiment, "sentiment")
meeting_status_enum = pg_enum(MeetingStatus, "meeting_status")
analysis_status_enum = pg_enum(AnalysisStatus, "analysis_status")
task_status_enum = pg_enum(TaskStatus, "task_status")
document_source_type_enum = pg_enum(DocumentSourceType, "document_source_type")
ingest_status_enum = pg_enum(IngestStatus, "ingest_status")
source_kind_enum = pg_enum(SourceKind, "source_kind")
claim_type_enum = pg_enum(ClaimType, "claim_type")
verification_status_enum = pg_enum(VerificationStatus, "verification_status")
verdict_enum = pg_enum(Verdict, "verdict")
validation_method_enum = pg_enum(ValidationMethod, "validation_method")
fact_status_enum = pg_enum(FactStatus, "fact_status")
owner_side_enum = pg_enum(OwnerSide, "owner_side")
commitment_status_enum = pg_enum(CommitmentStatus, "commitment_status")
risk_status_enum = pg_enum(RiskStatus, "risk_status")
recommendation_status_enum = pg_enum(RecommendationStatus, "recommendation_status")
chat_scope_enum = pg_enum(ChatScope, "chat_scope")
chat_role_enum = pg_enum(ChatRole, "chat_role")
message_status_enum = pg_enum(MessageStatus, "message_status")
