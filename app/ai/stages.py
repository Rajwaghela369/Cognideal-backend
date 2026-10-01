"""The stage implementations -- the single copy.

Each function takes what it needs and **returns the state keys it changed**.
Nothing here mutates a shared object and nothing here knows about LangGraph:
``graph.py`` holds the nodes and edges, these hold the work. That split is why
the graph nodes are three lines each, and why a stage can be called from a
script or a test without building a graph.

Seven of the thirteen stages call no model at all. The orchestration exists to
sequence deterministic work around a handful of model calls, not to chain model
calls together.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client, detect_ai, extract, reconcile, tiebreak, validate
from app.ai.prompts import get as get_prompt
from app.ai.schemas import SentimentOut, SummaryOut
from app.core.config import settings
from app.models import Deal, Document, DocumentChunk, ExtractedFact, Meeting
from app.models.enums import AnalysisStatus, ClaimType, Origin, ValidationMethod
from app.services import claims
from app.services import facts as facts_service
from app.services import gate0, gate2, roster

logger = logging.getLogger("dealpilot.ai.stages")


async def parse_transcript(db: AsyncSession, *, meeting: Meeting) -> Dict[str, Any]:
    """Stage 0 -- load the transcript's chunks, in order.

    The chunks *are* the text: ``documents.raw_text`` was dropped in 0008
    because the chunks ordered by ``chunk_index`` are the same thing and two
    copies can disagree after a re-ingest. So this reads them rather than
    re-extracting, and every later stage works from offsets that already exist
    in the database.

    A meeting with no transcript is not an error -- there is simply no text.
    Raising would make `failed` mean "no transcript attached", which is a
    different thing from "analysis broke".
    """
    if meeting.transcript_document_id is None:
        logger.info("stage0.no_transcript meeting=%s", meeting.id)
        return {"chunks": []}

    chunks = list(
        (
            await db.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == meeting.transcript_document_id)
                .order_by(DocumentChunk.chunk_index)
            )
        ).scalars()
    )
    logger.info("stage0.chunks meeting=%s chunks=%d", meeting.id, len(chunks))
    return {"chunks": chunks}


def speakers_from_chunks(chunks: Sequence[Any]) -> List[str]:
    """Distinct speakers across the chunks, in order of first appearance.

    Read from ``chunk_metadata['speakers']``, which ingest wrote at upload
    time, rather than re-parsing the text: the parse already happened and must
    not be able to disagree with itself.
    """
    seen: List[str] = []
    for chunk in chunks:
        for speaker in (chunk.chunk_metadata or {}).get("speakers") or []:
            name = roster.normalise(speaker)
            if name not in seen:
                seen.append(name)
    return seen


async def build_roster(
    db: AsyncSession, *, meeting: Meeting, chunks: Sequence[Any]
) -> Dict[str, Any]:
    """Stage 1 -- speaker labels into ``meeting_attendees``, resolved where safe.

    Unblocks the two deterministic risks that read that table. Resolution is
    reluctant by design: an unresolved attendee is the missing-stakeholder
    signal, so NULL is a result rather than a failure.
    """
    speakers = speakers_from_chunks(chunks)
    if not speakers:
        return {"attendees": []}

    account_id = await db.scalar(select(Deal.account_id).where(Deal.id == meeting.deal_id))
    resolutions = await roster.sync_roster(db, meeting.id, account_id, speakers)

    # The model sees only the band similarity could not settle. A score below
    # `roster_candidate_threshold` is not doubt, it is the answer "nobody we
    # know", so it never reaches a provider.
    ambiguous = [r for r in resolutions.values() if r.needs_tiebreak]
    if ambiguous and settings.ai_enabled:
        for resolution in ambiguous:
            await tiebreak.resolve_ambiguous(resolution)
        # Re-run the writer so a newly decided link lands. It upgrades NULL to
        # a contact and never the reverse, so repeating it is safe.
        await roster.sync_roster(db, meeting.id, account_id, speakers)

    logger.info(
        "stage1.roster meeting=%s speakers=%d linked=%d unknown=%s ambiguous=%s",
        meeting.id, len(speakers),
        sum(1 for r in resolutions.values() if r.decision == roster.LINKED),
        [r.raw_name for r in resolutions.values() if r.decision == roster.UNKNOWN] or "none",
        [r.raw_name for r in ambiguous] or "none",
    )
    return {"attendees": list(resolutions.values())}


async def should_extract(db: AsyncSession, *, meeting: Meeting) -> bool:
    """Whether stage 2 has anything to do.

    Re-analysis is idempotent at the **document** level, not the fact level: a
    fact has no natural key, so "this document has already been extracted" is
    the only safe statement. Without this check a second run doubles every
    fact on the deal.
    """
    if meeting.transcript_document_id is None:
        return False
    already = await facts_service.existing_fact_count(db, meeting.transcript_document_id)
    if already:
        logger.info(
            "stage2.skipped meeting=%s document=%s existing_facts=%d",
            meeting.id, meeting.transcript_document_id, already,
        )
        return False
    return True


async def extract_window(
    db: AsyncSession,
    *,
    meeting: Meeting,
    window: Sequence[Any],
    occurred_at: Optional[str],
    budget: Any,
) -> Dict[str, Any]:
    """Stage 2 -- one structured call over one window of chunks.

    Writes nothing to the database, which is what makes it the only stage a
    retry cannot duplicate, and what lets the eval harness score extraction by
    replaying a recorded run rather than inspecting rows.
    """
    candidates, _run = await extract.extract_window(
        window,
        meeting_type=meeting.meeting_type,
        occurred_at=occurred_at or "an unknown date",
        budget=budget,
    )
    return {"facts": candidates}


async def gate0_span_integrity(
    db: AsyncSession, *, facts: Sequence[Any]
) -> Dict[str, Any]:
    """Stage 3 -- does every citation resolve, byte for byte?

    Deterministic and model-free. The outcome is passed to stage 4 rather than
    recomputed there, but ``services/facts.py`` runs it again if a caller omits
    it: the gate is cheap and being skippable would make it worthless.
    """
    checks: Dict[int, Any] = {}
    for index, candidate in enumerate(facts):
        if candidate.rejected:
            continue
        checks[index] = await gate0.check_claim(
            db, candidate.content, [facts_service.citation_for(candidate)]
        )
    logger.info(
        "stage3.gate0 candidates=%d checked=%d passed=%d",
        len(facts), len(checks), sum(1 for c in checks.values() if c.passed),
    )
    return {"gate0_checks": checks}


async def drop_unevidenced(
    db: AsyncSession,
    *,
    meeting: Meeting,
    facts: Sequence[Any],
    gate0_checks: Dict[int, Any],
) -> Dict[str, Any]:
    """Stage 4 -- write what survived, and only that.

    The name is the rule: a claim with zero surviving links does not reach the
    database, let alone the UI. Rejections are returned and logged for the
    metric but never persisted -- a row for something that failed verification
    is a row somebody eventually renders.
    """
    if not facts:
        return {"surviving_facts": [], "written_fact_ids": [], "rejected_facts": []}

    occurred_at = await db.scalar(
        select(Document.occurred_at).where(Document.id == meeting.transcript_document_id)
    )
    result = await facts_service.write_facts(
        db,
        deal_id=meeting.deal_id,
        candidates=facts,
        meeting_id=meeting.id,
        document_id=meeting.transcript_document_id,
        occurred_at=occurred_at,
        checks=gate0_checks,
    )
    return {
        "written_fact_ids": result.written,
        "rejected_facts": result.rejected,
        "surviving_facts": [
            candidate for index, candidate in enumerate(facts)
            if gate0_checks.get(index) is not None and gate0_checks[index].passed
        ],
    }


async def gate1_entailment(
    db: AsyncSession,
    *,
    written_fact_ids: Sequence[Any],
    surviving_facts: Sequence[Any],
    budget: Any = None,
) -> Dict[str, Any]:
    """Stage 5 -- does the cited span actually support the claim?

    One starved call per claim. The validator is handed the claim text and the
    snippet and nothing else; see ``app/ai/validate.py`` on why that is
    enforced by the function signature rather than requested in the prompt.

    ``written_fact_ids`` and ``surviving_facts`` are positionally aligned --
    ``write_facts`` appends to ``written`` in candidate order and stage 4
    filters ``surviving`` by the same predicate. Asserted rather than assumed,
    because a silent misalignment would attach every verdict to the wrong
    claim.
    """
    if not written_fact_ids:
        return {"validations": {}}

    assert len(written_fact_ids) == len(surviving_facts), (
        "written ids and surviving candidates must align: %d vs %d"
        % (len(written_fact_ids), len(surviving_facts))
    )

    verdicts: Dict[Any, Any] = {}
    for fact_id, candidate in zip(written_fact_ids, surviving_facts):
        validation = await validate.validate_claim(
            candidate.content, [candidate.snippet], budget=budget
        )
        if validation is None:
            continue
        await claims.record_validation(
            db,
            claim_type=ClaimType.FACT,
            claim_id=fact_id,
            verdict=validation.verdict,
            method=ValidationMethod.LLM,
            rationale=validation.rationale,
            model=validation.model,
            validator_version=validation.validator_version,
        )
        verdicts[fact_id] = validation
        # The pair docs/schema/README.md section 5 calls the most valuable eval
        # case there is: the generator was certain and an independent check
        # disagreed. Logged at the point both numbers exist.
        if validation.quarantined and (candidate.confidence or 0) >= 0.8:
            logger.warning(
                "gate1.confident_and_wrong fact=%s confidence=%.2f verdict=%s why=%s",
                fact_id, candidate.confidence, validation.verdict.value,
                validation.rationale,
            )

    counts: Dict[str, int] = {}
    for validation in verdicts.values():
        counts[validation.verdict.value] = counts.get(validation.verdict.value, 0) + 1
    logger.info("stage5.gate1 validated=%d by_verdict=%s", len(verdicts), counts or "{}")
    return {"validations": verdicts}


async def quarantine(db: AsyncSession, *, validations: Dict[Any, Any]) -> Dict[str, Any]:
    """Stage 6 -- account for what Gate 1 excluded.

    **It writes nothing, deliberately.** Enforcement is a read-time filter
    (``queries.quarantine_filter``) rather than a column, for two reasons: one
    place cannot be forgotten by a future route, and the newest verdict is the
    one that counts -- a column would have to be rewritten on every
    re-validation and could disagree with `claim_validations`.

    Nor does it set ``extracted_facts.status='rejected'``. That value means a
    *human* rejected the fact, and reusing it for a system quarantine would
    corrupt the human-accept-rate metric, which is one of the few signals here
    that is not self-reported.

    So this stage exists to report: the counts belong in the run log where the
    rest of the run's accounting is.
    """
    quarantined = [
        fact_id for fact_id, validation in (validations or {}).items()
        if validation.quarantined
    ]
    if quarantined:
        logger.info(
            "stage6.quarantined count=%d facts=%s (hidden at read time, rows kept)",
            len(quarantined), [str(f)[:8] for f in quarantined],
        )
    return {"quarantined_fact_ids": quarantined}


def _renderable_facts(
    surviving_facts: Sequence[Any],
    written_fact_ids: Sequence[Any],
    validations: Dict[Any, Any],
) -> List[Any]:
    """The facts a summary may be built from.

    **Gate 1, not merely Gate 0.** A claim whose citation resolves but whose
    span does not support it is quarantined from every reader, and a summary is
    a reader -- composing from it would launder a rejected claim into prose
    that cites nothing. Facts with no verdict yet are included: unvalidated is
    not the same as failed, and Gate 0 already rejected what had no business
    existing.
    """
    keep: List[Any] = []
    for fact_id, candidate in zip(written_fact_ids, surviving_facts):
        validation = (validations or {}).get(fact_id)
        if validation is not None and validation.quarantined:
            continue
        keep.append(candidate)
    return keep


async def synthesize_summary(
    db: AsyncSession,
    *,
    meeting: Meeting,
    surviving_facts: Sequence[Any],
    written_fact_ids: Sequence[Any],
    validations: Dict[Any, Any],
    attendees: Sequence[Any],
    budget: Any = None,
) -> Dict[str, Any]:
    """Stage 9 -- the summary, composed from the facts rather than the transcript.

    Which is the whole point: every sentence then rests on a fact that rests on
    a verified span, and the input is ~1K tokens instead of ~10K. Summarising
    the transcript instead produces a plausible blob that can contradict the
    facts beside it on the same screen.
    """
    usable = _renderable_facts(surviving_facts, written_fact_ids, validations)
    if not usable:
        logger.info("stage9.no_facts meeting=%s", meeting.id)
        return {"summary": None}

    prompt = get_prompt("synthesize")
    rendered = "\n".join(
        "- [%s] %s" % (fact.fact_type, fact.content) for fact in usable
    )
    names = ", ".join(a.raw_name for a in attendees) if attendees else "not recorded"
    result, run = await client.structured(
        SummaryOut,
        prompt.messages(
            meeting_type=meeting.meeting_type,
            occurred_at=str(meeting.scheduled_at)[:10] if meeting.scheduled_at else "an unknown date",
            attendees=names,
            facts=rendered,
        ),
        task="synthesize",
        prompt_version=prompt.version,
        reasoning_effort="medium",
        budget=budget,
    )

    meeting.summary = result.summary.strip()
    meeting.analysis_origin = Origin.AI
    meeting.analysis_model = run.model
    logger.info(
        "stage9.summary meeting=%s from_facts=%d chars=%d",
        meeting.id, len(usable), len(meeting.summary),
    )
    return {"summary": meeting.summary}


async def sentiment(
    db: AsyncSession,
    *,
    meeting: Meeting,
    chunks: Sequence[Any],
    budget: Any = None,
) -> Dict[str, Any]:
    """Stage 10 -- tone, from the transcript, and explicitly not a claim.

    The one stage besides extraction that reads the raw text, because tone is
    not in the fact set. It never acquires a Gate 1 verdict: no span entails
    "the call went badly", so there is nothing for the validator to check it
    against. Rendered as a judgment with illustrative quotes rather than as a
    fact.
    """
    if not chunks:
        return {"sentiment": None}

    prompt = get_prompt("sentiment")
    transcript = extract.render_window(chunks)
    result, run = await client.structured(
        SentimentOut,
        prompt.messages(transcript=transcript),
        task="synthesize",
        prompt_version=prompt.version,
        role=client.ROLE_CHEAP,
        reasoning_effort="low",
        budget=budget,
    )

    meeting.sentiment = result.sentiment
    if meeting.analysis_model is None:
        meeting.analysis_origin = Origin.AI
        meeting.analysis_model = run.model
    logger.info(
        "stage10.sentiment meeting=%s -> %s why=%s quotes=%d",
        meeting.id, result.sentiment.value, result.reasoning[:60], len(result.quotes),
    )
    # The quotes are not persisted: `claim_evidence` needs a `claim_type`, and
    # sentiment is not one of the five claim tables -- correctly, since it is
    # outside the evidence contract. Storing them would need a column on
    # `meetings`; recorded as deferred rather than forced into a table that
    # means something else.
    return {"sentiment": result.sentiment.value}


async def finalize(
    db: AsyncSession, *, meeting: Meeting, stage_errors: Dict[str, str]
) -> Dict[str, Any]:
    """Stage 11 -- stamp the run as done.

    Taken over from the worker, which stamped `analyzed_at` itself with a
    comment saying this stage would claim it. Two writers for one field is how
    they come to disagree.

    `complete` even when degradable stages failed: `analysis_status='failed'`
    means stages 0-4 failed and no facts were produced. A missing summary is
    recorded in `analysis_error` and visible, not fatal.
    """
    meeting.analysis_status = AnalysisStatus.COMPLETE
    meeting.analyzed_at = func.now()
    if stage_errors:
        meeting.analysis_error = "; ".join(
            "%s: %s" % (name, reason) for name, reason in sorted(stage_errors.items())
        )
    logger.info(
        "stage11.finalized meeting=%s degraded=%s",
        meeting.id, sorted(stage_errors) or "none",
    )
    return {}


async def reconcile_commitments(
    db: AsyncSession,
    *,
    meeting: Meeting,
    surviving_facts: Sequence[Any],
    budget: Any = None,
) -> Dict[str, Any]:
    """Stage 7 -- do this call's facts show an open promise was kept?

    Produces **proposals**, and does not write them anywhere yet.

    The design said a proposal lands as a `recommendation`, and that turns out
    not to fit: every `ActionType` describes an action to *take* --
    `schedule_meeting`, `send_document`, `engage_stakeholder` -- and "this
    commitment now looks satisfied" is a proposed **data correction**, not an
    action. Filing it under the nearest action would corrupt the
    `dismissal_reason` and action-type distributions, which are two of the few
    signals here that are not self-reported.

    So the proposals are returned and logged, and the gap is recorded: it needs
    either an `ActionType` value (a one-line CHECK swap, since the set is
    `text + CHECK` precisely for this) or a surface of its own. Closing a
    promise nobody kept is the error that matters, so nothing is written on a
    guess.
    """
    if not surviving_facts:
        return {"commitment_proposals": []}

    proposals = await reconcile.reconcile_commitments(
        db, meeting.deal_id, surviving_facts, budget=budget
    )
    for commitment_id, why in proposals:
        logger.info(
            "stage7.proposal meeting=%s commitment=%s why=%s (not written -- see docstring)",
            meeting.id, commitment_id, why[:70],
        )
    return {"commitment_proposals": proposals}


async def supersede_facts(
    db: AsyncSession,
    *,
    meeting: Meeting,
    written_fact_ids: Sequence[Any],
    budget: Any = None,
) -> Dict[str, Any]:
    """Stage 8 -- Gate 2, both halves.

    **Contradiction** needs a model: does a fact written today replace one the
    deal already holds? The older row is marked `superseded` and kept, with its
    evidence -- never overwritten, never deleted.

    **Staleness** is arithmetic and runs here because it is the same gate: a
    claim whose newest evidence predates the last activity on the deal is
    flagged `stale`. Both answer "is this still current?", so keeping them in
    one stage means a reader never sees one applied without the other.
    """
    if not written_fact_ids:
        stale = await gate2.mark_stale_claims(db, meeting.deal_id, ClaimType.FACT)
        return {"superseded": [], "stale_claims": stale}

    fresh = list(
        (
            await db.execute(
                select(ExtractedFact).where(ExtractedFact.id.in_(list(written_fact_ids)))
            )
        ).scalars()
    )
    superseded = await reconcile.supersede_facts(
        db, meeting.deal_id, fresh, budget=budget
    )
    stale = await gate2.mark_stale_claims(db, meeting.deal_id, ClaimType.FACT)
    logger.info(
        "stage8.gate2 meeting=%s superseded=%d stale_claims=%d",
        meeting.id, len(superseded), len(stale),
    )
    return {"superseded": superseded, "stale_claims": stale}


async def redetect_risks(
    db: AsyncSession, *, meeting: Meeting, budget: Any = None
) -> Dict[str, Any]:
    """Stage 12 -- detect risks, now that the facts exist.

    **Both detectors run, in order.** The four SQL rules go first and are the
    floor: they cannot hallucinate, cannot flicker, and cannot be talked out of
    firing by a prompt-injected transcript. The AI pass then finds those keys
    already open and bumps them rather than duplicating, while adding the six
    semantic types SQL cannot express.

    Keeping the floor is not a hedge. It is what makes the AI detector
    measurable: a rule-detected risk the model did not propose is a recall miss
    with no human labelling, and that measurement regenerates on every run.
    """
    from app.services import detect as rules

    deal = await db.get(Deal, meeting.deal_id)
    deterministic = await rules.run(db, deal)
    outcome = {"rules": deterministic}

    if settings.ai_enabled:
        proposals, dossier = await detect_ai.propose(db, meeting.deal_id, budget=budget)
        applied = await detect_ai.apply(db, meeting.deal_id, proposals, dossier)
        outcome.update(
            inserted=len(applied.inserted), bumped=len(applied.bumped),
            suppressed=len(applied.suppressed), resolved=len(applied.resolved),
            rejected=len(applied.rejected),
        )
        logger.info(
            "stage12.detection meeting=%s rules=%s ai_inserted=%d ai_bumped=%d "
            "suppressed=%d resolved=%d rejected=%d",
            meeting.id, deterministic, len(applied.inserted), len(applied.bumped),
            len(applied.suppressed), len(applied.resolved), len(applied.rejected),
        )
    return {"detection": outcome}
