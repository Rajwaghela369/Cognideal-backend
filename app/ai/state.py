"""What flows through the analysis graph, and what rides alongside it.

Its own module because the state schema is read far more often than the wiring
is, and because ``stages.py`` and ``graph.py`` both need these types without
importing each other.

Two distinct things live here, and the distinction is load-bearing:

**State** is merged. Every node returns the keys it touched and LangGraph
combines them, so a key written by two nodes in one step needs a reducer saying
how. Only ``facts`` and the two dicts have one; everything else is written by
exactly one node, so last-writer-wins is accurate rather than merely tolerated.

**Context** is not merged, not serialized, and read-only to the nodes. The
``AsyncSession`` goes here because it is neither mergeable nor serializable, and
so does the ``Meeting`` -- an ORM object in merged state would raise a reducer
question with no sensible answer.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated, TypedDict

from app.ai.client import RunBudget
from app.models import Meeting


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Reducer for the two dict-valued keys: later writes win per key."""
    merged = dict(left)
    merged.update(right)
    return merged


@dataclass
class AnalysisContext:
    """Per-run values the nodes read and never write."""

    db: AsyncSession
    meeting: Meeting
    budget: RunBudget
    #: The transcript's `occurred_at`, resolved once rather than per window.
    occurred_at: Optional[str] = None


class AnalysisState(TypedDict, total=False):
    """The merged state.

    ``facts`` is the key the fan-out writes, and the reason this graph exists:
    N parallel ``extract_window`` nodes all append to it, which without a
    reducer is an error by design.
    """

    chunks: List[Any]
    attendees: List[Any]
    facts: Annotated[List[Any], list.__add__]
    gate0_checks: Annotated[Dict[int, Any], merge_dicts]
    surviving_facts: List[Any]
    written_fact_ids: List[Any]
    rejected_facts: List[Any]
    #: Gate 1 outcomes, keyed by written fact id.
    validations: Annotated[Dict[Any, Any], merge_dicts]
    quarantined_fact_ids: List[Any]
    #: Phase 6 outputs. `commitment_proposals` is deliberately not persisted --
    #: see stages.reconcile_commitments.
    commitment_proposals: List[Any]
    superseded: List[Any]
    stale_claims: List[Any]
    detection: Dict[str, Any]
    summary: Optional[str]
    sentiment: Optional[str]
    #: Degradable-stage failures, keyed by stage name. A non-empty value means
    #: the run completed with something missing rather than having failed.
    stage_errors: Annotated[Dict[str, str], merge_dicts]


class WindowInput(TypedDict):
    """One ``Send`` payload.

    A fanned-out node is invoked with the ``Send`` argument as its input rather
    than the whole state, so it needs its own schema. It still *writes* to the
    graph state -- that is what the reducer on ``facts`` is for.
    """

    window: List[Any]
