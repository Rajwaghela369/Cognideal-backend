"""AI risk detection -- tasks 7.3 to 7.10.

The model proposes; this module decides what is written. Everything between
those two things is deterministic, and that is the design: four properties the
SQL detector gets for free have to be re-established in Python, because a
language model breaks all four by default.

**Identity (7.8).** ``risk_type`` is a closed enum, so the partial unique index
still sees that two runs reported one problem. For ``other`` the model proposes
a slug and it is canonicalised here -- ``champion_going_quiet`` and
``champion_disengaged`` are one problem, and no constraint can know that.

**Citation (7.3).** An ``evidence_ref`` outside the dossier is rejected before
any write. Combined with handle-only referencing, a fabricated citation is
structurally impossible rather than merely unlikely.

**Resolution (7.9).** Asked, never inferred from silence, and requiring two
consecutive agreeing verdicts before a risk actually closes. A model pass is
not exhaustive; one that forgot to mention a risk is not evidence it is gone.

**No re-nagging (7.10).** The partial unique index is scoped to
``status='suggested'``, deliberately, so a dismissal does not block a fresh
suggestion forever -- which also means it cannot stop one tomorrow. The
cooldown policy does that, in Python.

Plus **severity hysteresis (7.6)**, which is not about correctness but about
trust: a badge that oscillates between `high` and `critical` on identical data
is read as broken.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client, dossier as dossier_mod, validate
from app.ai.prompts import get as get_prompt
from app.ai.schemas import DetectionOut, RiskOut
from app.models import Recommendation, Risk
from app.models.enums import (
    ActionType,
    DismissalReason,
    Origin,
    Priority,
    RecommendationStatus,
    RiskStatus,
    RiskType,
    Severity,
)
from app.services import claims as claims_service, gate0
from app.models.enums import ClaimType, SourceKind, ValidationMethod

logger = logging.getLogger("cognideal.ai.detect")

DETECTOR_VERSION = "detect@1"

_SLUG = re.compile(r"[^a-z0-9]+")

#: How long a dismissal suppresses the same suggestion. Per reason, because the
#: reasons mean different things: `already_handled` is "we are on it" and ages
#: out; `not_relevant` is "this does not apply here" and does not.
COOLDOWN_DAYS: Dict[str, Optional[int]] = {
    DismissalReason.ALREADY_HANDLED.value: 30,
    DismissalReason.NOT_RELEVANT.value: None,   # indefinite
    DismissalReason.WRONG.value: None,          # indefinite, and a detector bug
    DismissalReason.BAD_TIMING.value: 14,
    DismissalReason.OTHER.value: 30,
}

#: Lowering a severity needs new evidence or this long. Raising is immediate.
SEVERITY_HYSTERESIS_DAYS = 7

_SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


@dataclass
class Proposal:
    """One risk the model proposed, after validation and canonicalisation."""

    risk_type: str
    risk_key: str
    title: str
    description: str
    severity: Severity
    confidence: float
    evidence: List[dossier_mod.Entry]
    recommendation: Any

    @property
    def key(self) -> Tuple[str, str]:
        return (self.risk_type, self.risk_key)


@dataclass
class DetectionResult:
    proposed: List[Proposal] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    inserted: List[uuid.UUID] = field(default_factory=list)
    bumped: List[uuid.UUID] = field(default_factory=list)
    suppressed: List[Tuple[str, str]] = field(default_factory=list)
    resolved: List[uuid.UUID] = field(default_factory=list)
    verdicts: Dict[str, str] = field(default_factory=dict)
    verdict_evidence: Dict[str, List[dossier_mod.Entry]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 7.8 identity
# --------------------------------------------------------------------------


def canonical_key(slug: Optional[str]) -> str:
    """Normalise a model-proposed slug. Empty for the ten known types."""
    if not slug:
        return ""
    return _SLUG.sub("_", slug.strip().lower()).strip("_")[:60]


def match_existing_key(proposed: str, existing: Sequence[str]) -> str:
    """Reuse an existing key when the proposal is the same problem reworded.

    Token overlap rather than string distance: ``champion_going_quiet`` and
    ``champion_disengaged`` share no prefix and are not close by edit distance,
    but they share the subject -- which is what makes them one card. Half the
    tokens matching is a deliberately low bar, because two cards for one
    problem is the failure that fills the panel with noise, and the cost of a
    wrong merge is that one risk carries a slightly odd slug.
    """
    if not proposed:
        return ""
    want = set(proposed.split("_"))
    for candidate in existing:
        have = set(candidate.split("_"))
        shared = want & have
        if shared and len(shared) >= max(1, min(len(want), len(have)) // 2):
            return candidate
    return proposed


# --------------------------------------------------------------------------
# 7.6 severity
# --------------------------------------------------------------------------


def computed_severity(
    risk_type: str, dossier: dossier_mod.Dossier
) -> Optional[Severity]:
    """The severity a rule can work out, for the types where one can.

    Where a number decides it, a number decides it -- the model's opinion is
    clamped to this band. Dwell time and days-to-close are not matters of
    judgement.
    """
    text = dossier.render()
    if risk_type == RiskType.STALLED_STAGE.value:
        match = re.search(r"\((\d+) days\)", text)
        if match:
            days = int(match.group(1))
            return Severity.CRITICAL if days > 90 else Severity.HIGH if days > 45 else Severity.MEDIUM
    if risk_type == RiskType.CLOSE_DATE_AT_RISK.value:
        match = re.search(r"\((-?\d+) days from today\)", text)
        if match:
            days = int(match.group(1))
            return Severity.CRITICAL if days < 0 else Severity.HIGH if days < 21 else Severity.MEDIUM
    return None


def apply_severity_policy(
    proposed: Severity, existing: Optional[Risk], computed: Optional[Severity]
) -> Severity:
    """Clamp to a computed band, then apply hysteresis.

    Raising is immediate: a deal getting worse should say so at once. Lowering
    waits, because a badge that drops and climbs again on unchanged data is
    read as the tool being unreliable rather than the deal improving.
    """
    severity = computed or proposed
    if existing is None:
        return severity

    was = _SEVERITY_ORDER.index(existing.severity)
    now = _SEVERITY_ORDER.index(severity)
    if now >= was:
        return severity

    last_seen = existing.last_seen_at or existing.first_detected_at
    if last_seen is None:
        return severity
    age = datetime.now(last_seen.tzinfo) - last_seen
    if age < timedelta(days=SEVERITY_HYSTERESIS_DAYS):
        return existing.severity
    return severity


# --------------------------------------------------------------------------
# 7.10 dismissals
# --------------------------------------------------------------------------


def suppressed_by_dismissal(
    key: Tuple[str, str], action_type: str, dossier: dossier_mod.Dossier
) -> Optional[str]:
    """Why this suggestion must not be made again, if it must not be."""
    for reason, action, when in dossier.dismissals.get(key, []):
        cooldown = COOLDOWN_DAYS.get(reason, 30)
        if cooldown is None:
            return "%s (indefinite)" % reason
        if when is None:
            continue
        age = datetime.now(when.tzinfo) - when
        if age < timedelta(days=cooldown):
            # Scoped to the same action for `already_handled` -- they are doing
            # that particular thing, not every possible thing.
            if reason != DismissalReason.ALREADY_HANDLED.value or action == action_type:
                return "%s (%d of %d days)" % (reason, age.days, cooldown)
    return None


# --------------------------------------------------------------------------
# 7.3 the call, and handle validation
# --------------------------------------------------------------------------


async def propose(
    db: AsyncSession,
    deal_id: uuid.UUID,
    *,
    budget: Optional[client.RunBudget] = None,
) -> Tuple[DetectionResult, dossier_mod.Dossier]:
    """Ask the model what is at risk. Writes nothing.

    Separated from the write so task 7.4's shadow run can score the proposals
    against the deterministic detector before anything reaches the database.
    """
    dossier = await dossier_mod.build(db, deal_id)
    result = DetectionResult()
    if not dossier.entries:
        return result, dossier

    prompt = get_prompt("detect")
    answer, run = await client.structured(
        DetectionOut,
        prompt.messages(
            dossier=dossier.render(),
            open_risks=dossier.render_open_risks(),
            dismissals=dossier.render_dismissals(),
        ),
        task="detect",
        prompt_version=prompt.version,
        # `medium`, measured on a reasoning model rather than assumed. At
        # `high` the model spent its output budget reasoning and never emitted
        # the JSON; at `low` it found two of the three risks the SQL rules
        # confirm; `medium` found all three, in ~2.6K output tokens. Only sent
        # when the primary role is a reasoning model -- see
        # `client.is_reasoning_model`.
        reasoning_effort="medium",
        budget=budget,
    )

    known_keys = [
        risk.risk_key for risk in dossier.open_risks.values() if risk.risk_key
    ]

    for risk in answer.risks:
        entries, unknown = _resolve_refs(risk.evidence_refs, dossier)
        if unknown:
            # The whole reason the model answers with handles: an invented one
            # is detectable. Rejected rather than dropped-and-written, because
            # a risk citing nothing is a risk the UI must not show.
            result.rejected.append(
                (risk.risk_type, "cited unknown handles: %s" % ", ".join(sorted(unknown)))
            )
            continue
        if not entries:
            result.rejected.append((risk.risk_type, "no evidence cited"))
            continue
        _, rec_unknown = _resolve_refs(risk.recommendation.evidence_refs, dossier)
        if rec_unknown:
            result.rejected.append(
                (risk.risk_type, "recommendation cited unknown handles: %s" %
                 ", ".join(sorted(rec_unknown)))
            )
            continue

        risk_key = canonical_key(risk.risk_key) if risk.risk_type == RiskType.OTHER else ""
        if risk.risk_type == RiskType.OTHER:
            if not risk_key:
                result.rejected.append((risk.risk_type, "'other' with no risk_key"))
                continue
            risk_key = match_existing_key(risk_key, known_keys)

        result.proposed.append(
            Proposal(
                risk_type=risk.risk_type.value,
                risk_key=risk_key,
                title=risk.title.strip(),
                description=risk.description.strip(),
                severity=risk.severity,
                confidence=max(0.0, min(1.0, risk.confidence)),
                evidence=entries,
                recommendation=risk.recommendation,
            )
        )

    for verdict in answer.open_risk_verdicts:
        if verdict.risk_id in dossier.open_risks:
            entries, unknown = _resolve_refs(verdict.evidence_refs, dossier)
            if unknown:
                logger.warning(
                    "detect.verdict_unknown_handles risk=%s handles=%s",
                    verdict.risk_id, sorted(unknown),
                )
                continue
            result.verdicts[verdict.risk_id] = verdict.verdict
            result.verdict_evidence[verdict.risk_id] = entries
        else:
            logger.warning("detect.verdict_for_unknown_risk id=%s", verdict.risk_id)

    logger.info(
        "detect.proposed deal=%s risks=%d rejected=%d verdicts=%d tokens=%s",
        deal_id, len(result.proposed), len(result.rejected), len(result.verdicts),
        run.total_tokens,
    )
    return result, dossier


def _resolve_refs(
    refs: Sequence[str], dossier: dossier_mod.Dossier
) -> Tuple[List[dossier_mod.Entry], Set[str]]:
    entries, unknown = [], set()
    for ref in refs:
        entry = dossier.entries.get(ref.strip())
        if entry is None:
            unknown.add(ref.strip())
        else:
            entries.append(entry)
    return entries, unknown


# --------------------------------------------------------------------------
# 7.5 the write, and 7.9 resolution
# --------------------------------------------------------------------------


async def apply(
    db: AsyncSession,
    deal_id: uuid.UUID,
    result: DetectionResult,
    dossier: dossier_mod.Dossier,
    *,
    budget: Optional[client.RunBudget] = None,
) -> DetectionResult:
    """Upsert the proposals and act on the verdicts. No model call.

    Upsert, never insert: ``uq_risks_open_key`` exists so a re-run bumps
    ``last_seen_at`` rather than adding a fifth copy, and this is the code that
    honours it.
    """
    existing = {
        (risk.risk_type, risk.risk_key or ""): risk
        for risk in (
            await db.execute(
                select(Risk).where(Risk.deal_id == deal_id, Risk.status == RiskStatus.OPEN)
            )
        ).scalars()
    }

    for proposal in result.proposed:
        action = proposal.recommendation.action_type.value
        reason = suppressed_by_dismissal(proposal.key, action, dossier)
        if reason:
            result.suppressed.append((("%s/%s" % proposal.key).rstrip("/"), reason))
            continue

        risk_claim = "%s. %s" % (proposal.title, proposal.description)
        risk_grounding = await _ground(
            db, risk_claim, proposal.evidence, budget=budget
        )
        rec = proposal.recommendation
        rec_entries, _ = _resolve_refs(rec.evidence_refs, dossier)
        rec_entries = rec_entries or proposal.evidence
        rec_claim = "%s. %s %s" % (rec.title, rec.description, rec.rationale)
        rec_grounding = await _ground(db, rec_claim, rec_entries, budget=budget)
        if risk_grounding is None or rec_grounding is None:
            result.rejected.append((proposal.risk_type, "failed Gate 0 or Gate 1"))
            continue

        current = existing.get(proposal.key)
        severity = apply_severity_policy(
            proposal.severity, current, computed_severity(proposal.risk_type, dossier)
        )

        if current is not None:
            current.last_seen_at = func.now()
            current.severity = severity
            # Proposed again means present: a pending first strike towards
            # resolution no longer stands.
            current.description = _without_marker(current.description)
            current.model = client.model_for(client.ROLE_PRIMARY)
            current.detector_version = DETECTOR_VERSION
            await _persist_grounding(
                db, deal_id, ClaimType.RISK, current.id, risk_grounding
            )
            result.bumped.append(current.id)
            risk = current
        else:
            risk = Risk(
                deal_id=deal_id,
                risk_type=proposal.risk_type,
                risk_key=proposal.risk_key,
                title=proposal.title,
                description=proposal.description,
                severity=severity,
                status=RiskStatus.OPEN,
                origin=Origin.AI,
                confidence=proposal.confidence,
                model=client.model_for(client.ROLE_PRIMARY),
                detector_version=DETECTOR_VERSION,
            )
            db.add(risk)
            await db.flush()
            await _persist_grounding(
                db, deal_id, ClaimType.RISK, risk.id, risk_grounding
            )
            result.inserted.append(risk.id)

        recommendation = await db.scalar(
            select(Recommendation).where(
                Recommendation.deal_id == deal_id,
                Recommendation.source_risk_id == risk.id,
                Recommendation.status == RecommendationStatus.SUGGESTED,
            )
        )
        if recommendation is None:
            recommendation = Recommendation(
                deal_id=deal_id,
                source_risk_id=risk.id,
                status=RecommendationStatus.SUGGESTED,
                origin=Origin.AI,
            )
            db.add(recommendation)
        recommendation.title = rec.title.strip()
        recommendation.description = rec.description.strip()
        recommendation.rationale = rec.rationale.strip()
        recommendation.action_type = rec.action_type.value
        recommendation.priority = rec.priority
        recommendation.confidence = proposal.confidence
        recommendation.model = client.model_for(client.ROLE_PRIMARY)
        recommendation.detector_version = DETECTOR_VERSION
        await db.flush()
        await _persist_grounding(
            db, deal_id, ClaimType.RECOMMENDATION, recommendation.id, rec_grounding
        )

    await _apply_verdicts(db, result, dossier, budget=budget)
    await db.flush()
    logger.info(
        "detect.applied deal=%s inserted=%d bumped=%d suppressed=%d resolved=%d",
        deal_id, len(result.inserted), len(result.bumped),
        len(result.suppressed), len(result.resolved),
    )
    return result


def _citation(entry: dossier_mod.Entry) -> Dict[str, Any]:
    return {
        "source_kind": entry.source_kind,
        "snippet": entry.snippet,
        "record_ref": entry.record_ref,
        "document_id": entry.document_id,
        "chunk_id": entry.chunk_id,
        "char_start": entry.char_start,
        "char_end": entry.char_end,
        "speaker": entry.speaker,
        "occurred_at": entry.occurred_at,
    }


async def _ground(db, claim_text, entries, *, budget=None):
    """Run both grounding gates before an assertion reaches the database."""
    check = await gate0.check_claim(
        db, claim_text, [_citation(entry) for entry in entries]
    )
    if not check.passed:
        logger.info(
            "detect.gate0_rejected claim=%r reason=%s", claim_text[:80], check.reason
        )
        return None
    surviving = [
        (entry, link) for entry, link in zip(entries, check.links) if link.verified
    ]
    validation = await validate.validate_claim(
        claim_text, [entry.snippet for entry, _ in surviving], budget=budget
    )
    if validation is None or validation.quarantined:
        logger.info(
            "detect.gate1_rejected claim=%r verdict=%s",
            claim_text[:80], validation.verdict if validation else None,
        )
        return None
    return surviving, validation


def _evidence_key(source_kind, chunk_id, char_start, char_end, record_ref, snippet):
    """What makes two citations the same citation, whatever types they arrive in."""
    return (
        str(getattr(source_kind, "value", source_kind)),
        str(chunk_id) if chunk_id else None,
        char_start,
        char_end,
        json.dumps(record_ref, sort_keys=True, default=str) if record_ref else None,
        snippet,
    )


async def _persist_grounding(db, deal_id, claim_type, claim_id, grounding) -> None:
    """Attach the grounding's citations and record its verdict.

    Citations the claim already carries are skipped. A risk re-detected on
    every run used to gain a fresh copy of each citation every time --
    ``attach_evidence`` always inserts -- so a daily sweep piled duplicates
    onto the panel and into the chat's ``list_risks``. The verdict is still
    recorded each run: that table is a history, and the newest one counts.
    """
    surviving, validation = grounding
    have = {
        _evidence_key(row.source_kind, row.chunk_id, row.char_start, row.char_end,
                      row.record_ref, row.snippet)
        for row in await claims_service.evidence_for(db, claim_type, claim_id)
    }
    added = 0
    for entry, link in surviving:
        key = _evidence_key(entry.source_kind, entry.chunk_id, entry.char_start,
                            entry.char_end, entry.record_ref, entry.snippet)
        if key in have:
            continue
        have.add(key)
        added += 1
        await claims_service.attach_evidence(
            db,
            claim_type=claim_type,
            claim_id=claim_id,
            deal_id=deal_id,
            source_kind=entry.source_kind,
            snippet=entry.snippet,
            record_ref=entry.record_ref,
            document_id=entry.document_id,
            chunk_id=entry.chunk_id,
            char_start=entry.char_start,
            char_end=entry.char_end,
            speaker=entry.speaker,
            occurred_at=entry.occurred_at,
            verification_status=link.status,
        )
    if added:
        logger.debug(
            "detect.evidence_attached claim=%s:%s added=%d", claim_type.value, claim_id, added
        )
    await claims_service.record_validation(
        db,
        claim_type=claim_type,
        claim_id=claim_id,
        verdict=validation.verdict,
        method=ValidationMethod.LLM,
        rationale=validation.rationale,
        model=validation.model,
        validator_version=validation.validator_version,
    )


async def _apply_verdicts(db, result: DetectionResult, dossier, *, budget=None) -> None:
    """Resolve only on a repeated, cited `resolved`.

    One `resolved` is not enough: a single pass saying a risk is gone is the
    same evidence as a single pass forgetting to mention it, and the cost of
    being wrong is a risk that disappears from the panel while still being
    true. Two consecutive agreeing verdicts is the cheapest check that a single
    sampling fluke cannot pass, recorded on the row itself rather than in a
    side table.
    """
    for risk_id, verdict in result.verdicts.items():
        risk = dossier.open_risks.get(risk_id)
        if risk is None or risk.risk_type in _RULE_TYPES:
            continue
        if verdict != "resolved":
            # Two *consecutive* resolutions: anything else in between breaks
            # the run, or two lucky passes months apart could close a risk
            # that was reported present in between.
            if _RESOLUTION_MARKER in (risk.description or ""):
                risk.description = _without_marker(risk.description)
                logger.info("detect.resolution_strike_reset risk=%s verdict=%s", risk.id, verdict)
            continue
        entries = result.verdict_evidence.get(risk_id) or []
        grounding = await _ground(
            db,
            "The risk '%s' is resolved." % risk.title,
            entries,
            budget=budget,
        )
        if grounding is None:
            logger.info("detect.resolution_rejected risk=%s", risk.id)
            continue
        # `description` carries the strike because there is nowhere else to put
        # it without a migration; replace with a column if this outlives Phase 7.
        if _RESOLUTION_MARKER in (risk.description or ""):
            risk.status = RiskStatus.RESOLVED
            risk.resolved_at = func.now()
            risk.description = _without_marker(risk.description)
            logger.info("detect.risk_resolved risk=%s title=%r", risk.id, risk.title)
            await _persist_grounding(
                db, risk.deal_id, ClaimType.RISK, risk.id, grounding
            )
            result.resolved.append(risk.id)
        else:
            risk.description = "%s %s" % (risk.description or "", _RESOLUTION_MARKER)
            logger.info("detect.resolution_first_strike risk=%s", risk.id)


_RESOLUTION_MARKER = "[resolution-proposed]"


def _without_marker(description: Optional[str]) -> Optional[str]:
    if not description or _RESOLUTION_MARKER not in description:
        return description
    return description.replace(_RESOLUTION_MARKER, "").strip()


# --------------------------------------------------------------------------
# 7.4 shadow mode
# --------------------------------------------------------------------------


async def shadow_run(
    db: AsyncSession,
    deal_id: uuid.UUID,
    *,
    budget: Optional[client.RunBudget] = None,
) -> Dict[str, Any]:
    """Propose, score against the SQL rules, write nothing.

    The dress rehearsal, and the reason the deterministic detector stays once
    this exists. The six SQL rules cannot hallucinate and cannot miss, so
    **any rule-detected risk the model did not propose is a measured recall
    miss with no human labelling** -- free ground truth that regenerates on
    every run, forever.

    Run it twice on an unchanged deal for the stability number, which is the
    one to watch first: a panel that changes between two identical runs reads
    as broken regardless of either run's precision.
    """
    from app.services import detect as rules

    from app.models import Deal

    result, dossier = await propose(db, deal_id, budget=budget)
    # The rules evaluated but not written: `detect.run` upserts, and a shadow
    # run must leave the database exactly as it found it.
    deal = await db.get(Deal, deal_id)
    rule_findings = [
        f for f in [await rule(db, deal) for rule in rules.RULES] if f is not None
    ]

    proposed_keys = {p.key for p in result.proposed}

    # A risk the model affirmed as `still_present` has been found, even though
    # it was not re-proposed -- and not re-proposing an already-open risk is
    # correct behaviour, since the prompt asks for a verdict on it instead.
    #
    # Counting only `proposed` scored this run at **recall 0.0** while the
    # model had in fact affirmed every rule-detected risk. The metric was
    # wrong, not the detector, which is a reminder that a shadow run measures
    # the measurement too.
    affirmed_keys = {
        (dossier.open_risks[rid].risk_type, dossier.open_risks[rid].risk_key or "")
        for rid, verdict in result.verdicts.items()
        if verdict == "still_present" and rid in dossier.open_risks
    }
    ai_keys = proposed_keys | affirmed_keys

    rule_keys = {(f.risk_type, "") for f in rule_findings}
    overlapping = {k for k in rule_keys if k[0] in _RULE_TYPES}

    found = overlapping & ai_keys
    missed = overlapping - ai_keys
    return {
        "proposed": sorted("%s/%s" % k for k in proposed_keys),
        "affirmed_open": sorted("%s/%s" % k for k in affirmed_keys),
        "rejected": result.rejected,
        "rule_risks": sorted("%s/%s" % k for k in overlapping),
        "recall": round(len(found) / len(overlapping), 4) if overlapping else None,
        "missed": sorted("%s/%s" % k for k in missed),
        "ai_only": sorted("%s/%s" % k for k in (ai_keys - overlapping)),
        "_recall_note": "proposed + affirmed-as-still-present, vs the six SQL rules",
        "verdicts": result.verdicts,
    }


#: The six types the SQL rules cover, and therefore the only ones recall can
#: be measured on without a human writing labels.
_RULE_TYPES = {
    RiskType.NO_ECONOMIC_BUYER.value,
    RiskType.SINGLE_THREADED.value,
    RiskType.STALLED_STAGE.value,
    RiskType.CLOSE_DATE_AT_RISK.value,
    RiskType.MISSED_COMMITMENT.value,
    RiskType.GONE_QUIET.value,
}
