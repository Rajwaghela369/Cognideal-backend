"""Model registry.

``alembic/env.py`` does ``from app.models import *``, which respects ``__all__``
below. A model missing from this list is invisible to autogenerate and its
table silently never gets created -- no error, just an absent table discovered
much later. ``tests/test_model_registry.py`` guards against that.
"""

from app.models.account import Account, Contact
from app.models.assertion import Commitment, ExtractedFact, Recommendation, Risk
from app.models.chat import ChatMessage, ChatSession
from app.models.deal import Deal, DealContact, DealStageHistory
from app.models.document import Document, DocumentChunk
from app.models.evidence import ClaimEvidence, ClaimValidation, Evidence
from app.models.meeting import Meeting, MeetingAttendee, MeetingBrief
from app.models.task import Activity, Task

__all__: list[str] = [
    # Layer A -- deal domain
    "Account",
    "Contact",
    "Deal",
    "DealContact",
    "DealStageHistory",
    "Meeting",
    "MeetingAttendee",
    "Task",
    "Activity",
    # Layer B -- knowledge and ingest
    "Document",
    "DocumentChunk",
    # Layer C -- assertions and evidence
    "Evidence",
    "ClaimEvidence",
    "ClaimValidation",
    "ExtractedFact",
    "Commitment",
    "Risk",
    "Recommendation",
    "MeetingBrief",
    "ChatSession",
    "ChatMessage",
]
