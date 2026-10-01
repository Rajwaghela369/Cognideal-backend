"""The meeting analysis pipeline.

Thirteen stages, listed in ``docs/ai/README.md`` section 4. They are declared
here in full, from Phase 0 onward, even though every one is still a no-op --
because the *shape* is the design decision and each later phase replaces
exactly one function. A pipeline that grows a stage list as it goes cannot be
read against the document that specifies it.

**The degradation rule.** ``analysis_status='failed'`` means stages 0-4 failed.
Those five produce the facts; everything after them enriches. A failed summary
must not discard facts that already landed, so a late stage failing still
yields ``complete`` with its error logged. That is what ``critical`` encodes.

**Not a LangGraph graph yet, deliberately.** ``docs/ai/README.md`` section 7
puts the pipeline in a ``StateGraph``, and it will be one -- but a graph whose
nodes are all pass-throughs is structure with nothing flowing through it. The
conversion belongs in Phase 3 (task 3.7), when extraction and Gate 0 give the
state object its first real contents and stage 2's fan-out gives the graph
something only a reducer can express. Until then this sequential runner holds
the same contract: named stages, ordered, with the critical/degradable split.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import RunBudget
from app.models import Meeting

logger = logging.getLogger("dealpilot.ai.pipeline")


@dataclass
class PipelineState:
    """What flows between stages.

    Becomes the LangGraph state object in Phase 3. Everything on it is written
    by one stage and read by later ones -- nothing is fetched twice.
    """

    meeting: Meeting
    budget: RunBudget
    chunks: List[Any] = field(default_factory=list)
    attendees: List[Any] = field(default_factory=list)
    facts: List[Any] = field(default_factory=list)
    surviving_facts: List[Any] = field(default_factory=list)
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    stage_errors: Dict[str, str] = field(default_factory=dict)


StageFn = Callable[[AsyncSession, PipelineState], Awaitable[None]]


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    # True for stages 0-4: if one of these fails the run is `failed`, because
    # without them there are no facts and the meeting was not analysed.
    critical: bool
    run: StageFn


async def _todo(db: AsyncSession, state: PipelineState) -> None:
    """A stage that is not built yet. Does nothing, reports nothing."""
    return None


def _stage(index: int, name: str, critical: bool, run: Optional[StageFn] = None) -> Stage:
    return Stage(index=index, name=name, critical=critical, run=run or _todo)


# The thirteen stages of docs/ai/README.md section 4. Phase numbers in comments
# are the task list's; each phase swaps `_todo` for a real implementation.
STAGES: List[Stage] = [
    _stage(0, "parse_transcript", True),      # Phase 2.1
    _stage(1, "roster", True),                # Phase 2.2-2.4
    _stage(2, "extract_facts", True),         # Phase 3.2
    _stage(3, "gate0_span_integrity", True),  # Phase 3.3
    _stage(4, "drop_unevidenced", True),      # Phase 3.4
    _stage(5, "gate1_entailment", False),     # Phase 4.1
    _stage(6, "quarantine", False),           # Phase 4.3
    _stage(7, "reconcile_commitments", False),# Phase 6.1
    _stage(8, "supersede_facts", False),      # Phase 6.2
    _stage(9, "synthesize_summary", False),   # Phase 5.2
    _stage(10, "sentiment", False),           # Phase 5.3
    _stage(11, "finalize", False),            # Phase 5.4
    _stage(12, "redetect_risks", False),      # Phase 7
]


class CriticalStageFailed(RuntimeError):
    """A stage 0-4 failure. The caller records `failed` and does not retry."""

    def __init__(self, stage: Stage, cause: BaseException) -> None:
        super().__init__("stage %d (%s) failed: %s" % (stage.index, stage.name, cause))
        self.stage = stage
        self.cause = cause


async def run_meeting_analysis(
    db: AsyncSession, meeting: Meeting, budget: Optional[RunBudget] = None
) -> PipelineState:
    """Run every stage in order, in the caller's transaction.

    No commit here. The worker owns the transaction so that a crash rolls the
    whole run back and leaves the row claimable again -- see ``worker.py``.
    """
    state = PipelineState(meeting=meeting, budget=budget or RunBudget())

    for stage in STAGES:
        try:
            await stage.run(db, state)
        except Exception as exc:  # noqa: BLE001 -- classified by `critical`
            if stage.critical:
                raise CriticalStageFailed(stage, exc) from exc
            # Degradable: record it and carry on. The facts are already written.
            state.stage_errors[stage.name] = "%s: %s" % (type(exc).__name__, exc)
            logger.warning(
                "pipeline.stage_failed meeting=%s stage=%d:%s error=%r",
                meeting.id, stage.index, stage.name, exc,
            )

    return state
