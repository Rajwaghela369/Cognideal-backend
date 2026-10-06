"""The meeting analysis pipeline as a LangGraph ``StateGraph`` -- task 3.7.

A sequential runner held these thirteen stages first, and was the right thing
until stage 2 had real work to fan out. It then became a second implementation
with its own copy of every stage body, and was removed; the declaration it
carried lives on in ``stage_registry.py``. The fan-out is the whole reason a
graph earns its place here: N extraction windows run in parallel and write one fact list, which
plain code has to merge by hand and a graph expresses as a reducer.

Four things the conversion changes, each of them a consequence of LangGraph's
model rather than a preference:

*   **Nodes return partial updates; they do not mutate.** Every node returns a
    dict of the keys it touched.
*   **``facts`` declares how writes merge.** ``Annotated[List, operator.add]``
    -- without it, two ``extract_window`` nodes writing the same key in one
    step is an error, by design.
*   **The ``AsyncSession`` is not in the state.** State is merged and may be
    serialized; a session is neither. It travels as graph *context*, which is
    read-only to the nodes and never merged.
*   **``retry_policy`` goes on ``extract_window`` only.** It writes nothing
    until Gate 0, so a retry is free. Every other node writes rows, and a node
    that wrote and then retried writes twice.

No checkpointer. ``analysis_status`` is the run state and each stage writes its
own rows, so the durability already exists; adding
``langgraph-checkpoint-postgres`` would mean a second Postgres driver and
tables Alembic would try to drop (docs/ai/README.md section 7).
"""

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Send
from sqlalchemy import select

from app.ai import extract, stages
from app.ai.client import RunBudget
from app.ai.stage_registry import BY_NAME, LATE_STAGES, CriticalStageFailed
from app.ai.state import AnalysisContext, AnalysisState, WindowInput
from app.models import Document, Meeting
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

logger = logging.getLogger("cognideal.ai.graph")


def guarded(name: str):
    """Wrap a node with the critical/degradable rule from the stage registry.

    This is the rule the first conversion lost. The sequential runner caught a
    stage failure and, for a degradable stage, recorded it and carried on --
    "a failed summary must not discard the facts that already landed". A raw
    LangGraph node has no such notion: anything it raises propagates out of
    ``ainvoke`` and rolls back the transaction, verified facts included.

    It cost nothing while stages 5-12 were no-ops that could not fail. Gate 1
    is stage 5 and the first thing that genuinely can, so the wrapper lands
    before it does.
    """
    stage = BY_NAME[name]

    def decorate(fn):
        async def node(state, runtime):
            try:
                return await fn(state, runtime)
            except CriticalStageFailed:
                raise
            except Exception as exc:  # noqa: BLE001 -- classified by `critical`
                if stage.critical:
                    raise CriticalStageFailed(stage, exc) from exc
                logger.warning(
                    "graph.stage_failed stage=%d:%s error=%r", stage.index, name, exc
                )
                return {"stage_errors": {name: "%s: %s" % (type(exc).__name__, exc)}}

        node.__name__ = name
        return node

    return decorate


# --------------------------------------------------------------------------
# nodes -- unpack context, call the stage, return its dict
# --------------------------------------------------------------------------


@guarded("parse_transcript")
async def parse_transcript(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.parse_transcript(ctx.db, meeting=ctx.meeting)


@guarded("roster")
async def roster(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.build_roster(
        ctx.db, meeting=ctx.meeting, chunks=state.get("chunks") or []
    )


def plan_windows(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    """Exists so the fan-out has an origin: a node returns state, an edge
    returns destinations."""
    return {}


async def fan_out(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Any:
    """One ``Send`` per window -- the map half of map-reduce.

    Routes straight to Gate 0 rather than returning no destinations when there
    is nothing to extract: an empty ``Send`` list leaves the graph with no live
    branch.
    """
    ctx = runtime.context
    chunks = state.get("chunks") or []
    if not chunks or not await stages.should_extract(ctx.db, meeting=ctx.meeting):
        return "gate0"
    return [
        Send("extract_window", {"window": window})
        for window in extract.build_windows(chunks)
    ]


@guarded("extract_window")
async def extract_window(state: WindowInput, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.extract_window(
        ctx.db, meeting=ctx.meeting, window=state["window"],
        occurred_at=ctx.occurred_at, budget=ctx.budget,
    )


@guarded("gate0")
async def gate0(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    return await stages.gate0_span_integrity(
        runtime.context.db, facts=state.get("facts") or []
    )


@guarded("drop_unevidenced")
async def drop_unevidenced(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.drop_unevidenced(
        ctx.db, meeting=ctx.meeting, facts=state.get("facts") or [],
        gate0_checks=state.get("gate0_checks") or {},
    )


def anything_survived(state: AnalysisState) -> str:
    """The branch after Gate 0.

    Nothing survived means stages 5-10 have no work: Gate 1 has no claim to
    validate and the summary has no facts to compose from. Not a failure --
    the fact stages ran and found nothing citable.

    It skips to **`finalize`, not to `END`**. Routing straight out would leave
    `analysis_status` unset, so the worker would find the row still `queued`
    and re-run it on every poll, forever. A meeting with no facts is still a
    meeting that was analysed.
    """
    return "gate1_entailment" if state.get("surviving_facts") else "finalize"


@guarded("gate1_entailment")
async def gate1_entailment(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.gate1_entailment(
        ctx.db,
        written_fact_ids=state.get("written_fact_ids") or [],
        surviving_facts=state.get("surviving_facts") or [],
        budget=ctx.budget,
    )


@guarded("quarantine")
async def quarantine(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    return await stages.quarantine(
        runtime.context.db, validations=state.get("validations") or {}
    )


@guarded("reconcile_commitments")
async def reconcile_commitments(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.reconcile_commitments(
        ctx.db, meeting=ctx.meeting,
        surviving_facts=state.get("surviving_facts") or [], budget=ctx.budget,
    )


@guarded("supersede_facts")
async def supersede_facts(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.supersede_facts(
        ctx.db, meeting=ctx.meeting,
        written_fact_ids=state.get("written_fact_ids") or [], budget=ctx.budget,
    )


@guarded("synthesize_summary")
async def synthesize_summary(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.synthesize_summary(
        ctx.db, meeting=ctx.meeting,
        surviving_facts=state.get("surviving_facts") or [],
        written_fact_ids=state.get("written_fact_ids") or [],
        validations=state.get("validations") or {},
        attendees=state.get("attendees") or [],
        budget=ctx.budget,
    )


@guarded("sentiment")
async def sentiment(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.sentiment(
        ctx.db, meeting=ctx.meeting, chunks=state.get("chunks") or [],
        budget=ctx.budget,
    )


@guarded("finalize")
async def finalize(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.finalize(
        ctx.db, meeting=ctx.meeting, stage_errors=state.get("stage_errors") or {}
    )


@guarded("redetect_risks")
async def redetect_risks(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
    ctx = runtime.context
    return await stages.redetect_risks(ctx.db, meeting=ctx.meeting, budget=ctx.budget)


def _todo(name: str):
    """A stage not built yet. Phases 4-7 replace these one at a time."""

    @guarded(name)
    async def node(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> Dict:
        return {}

    return node


#: Node name -> callable, resolved at graph-build time.
#:
#: A registry rather than module globals so there is one patchable seam. A test
#: that wants a node to fail replaces the entry; reaching into module
#: attributes would miss the eight late stages, which are closures rather than
#: top-level functions, and the resulting test passes while asserting nothing.
NODES: Dict[str, Any] = {
    "parse_transcript": parse_transcript,
    "roster": roster,
    "plan_windows": plan_windows,
    "extract_window": extract_window,
    "gate0": gate0,
    "drop_unevidenced": drop_unevidenced,
}
NODES["gate1_entailment"] = gate1_entailment
NODES["quarantine"] = quarantine
NODES["reconcile_commitments"] = reconcile_commitments
NODES["supersede_facts"] = supersede_facts
NODES["synthesize_summary"] = synthesize_summary
NODES["sentiment"] = sentiment
NODES["finalize"] = finalize
NODES["redetect_risks"] = redetect_risks
NODES.update({name: _todo(name) for name in LATE_STAGES if name not in NODES})


def build_graph():
    """Assemble and compile the graph. No checkpointer -- see the docstring."""
    graph = StateGraph(AnalysisState, context_schema=AnalysisContext)

    graph.add_node("parse_transcript", NODES["parse_transcript"])
    graph.add_node("roster", NODES["roster"])
    graph.add_node("plan_windows", NODES["plan_windows"])
    graph.add_node(
        "extract_window",
        NODES["extract_window"],
        input_schema=WindowInput,
        # The only node that writes nothing before it returns, so the only one
        # where a retry cannot duplicate a row.
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=2.0, jitter=True),
    )
    graph.add_node("gate0", NODES["gate0"])
    graph.add_node("drop_unevidenced", NODES["drop_unevidenced"])
    for name in LATE_STAGES:
        graph.add_node(name, NODES[name])

    graph.add_edge(START, "parse_transcript")
    graph.add_edge("parse_transcript", "roster")
    graph.add_edge("roster", "plan_windows")
    # The map: one branch per window, or straight to Gate 0 with no transcript.
    graph.add_conditional_edges("plan_windows", fan_out, ["extract_window", "gate0"])
    # The reduce: every window joins here, and `facts` merges via its reducer.
    graph.add_edge("extract_window", "gate0")
    graph.add_edge("gate0", "drop_unevidenced")
    graph.add_conditional_edges(
        "drop_unevidenced", anything_survived, ["gate1_entailment", "finalize"]
    )
    for current, following in zip(LATE_STAGES, LATE_STAGES[1:]):
        graph.add_edge(current, following)
    graph.add_edge(LATE_STAGES[-1], END)

    return graph.compile()


async def run_meeting_analysis(
    db: AsyncSession, meeting: Meeting, budget: Optional[RunBudget] = None
) -> AnalysisState:
    """Run the graph for one meeting, in the caller's transaction.

    No commit: the worker owns the transaction, so a crash rolls the whole run
    back and leaves the row claimable again.
    """
    occurred_at = None
    if meeting.transcript_document_id is not None:
        value = await db.scalar(
            select(Document.occurred_at).where(
                Document.id == meeting.transcript_document_id
            )
        )
        occurred_at = str(value)[:10] if value else None

    compiled = build_graph()
    return await compiled.ainvoke(
        {"facts": [], "gate0_checks": {}, "stage_errors": {}},
        context=AnalysisContext(
            db=db, meeting=meeting, budget=budget or RunBudget(), occurred_at=occurred_at
        ),
        config={"recursion_limit": 50},
    )
