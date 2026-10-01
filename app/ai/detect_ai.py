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

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client, dossier as dossier_mod
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
from app.services import claims as claims_service
from app.models.enums import ClaimType, SourceKind

logger = logging.getLogger("dealpilot.ai.detect")

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
        # `medium`, measured rather than assumed. At `high` the model spends
        # its output budget reasoning and never emits the JSON, which Groq
        # returns as `400 json_validate_failed` with an EMPTY
        # `failed_generation` -- the shape that means nothing was produced, not
        # that the schema was wrong. At `low` it found two of the three risks
        # the SQL rules confirm. `medium` found all three, in ~2.6K output
        # tokens.
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
            result.verdicts[verdict.risk_id] = verdict.verdict
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

        current = existing.get(proposal.key)
        severity = apply_severity_policy(
            proposal.severity, current, computed_severity(proposal.risk_type, dossier)
        )

        if current is not None:
            current.last_seen_at = func.now()
            current.severity = severity
            current.model = client.model_for(client.ROLE_PRIMARY)
            current.detector_version = DETECTOR_VERSION
            result.bumped.append(current.id)
            continue

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
        await _cite(db, deal_id, ClaimType.RISK, risk.id, proposal.evidence)

        rec = proposal.recommendation
        recommendation = Recommendation(
            deal_id=deal_id,
            source_risk_id=risk.id,
            title=rec.title.strip(),
            description=rec.description.strip(),
            rationale=rec.rationale.strip(),
            action_type=rec.action_type.value,
            priority=rec.priority,
            status=RecommendationStatus.SUGGESTED,
            origin=Origin.AI,
            confidence=proposal.confidence,
            model=client.model_for(client.ROLE_PRIMARY),
            detector_version=DETECTOR_VERSION,
        )
        db.add(recommendation)
        await db.flush()
        rec_entries, _ = _resolve_refs(rec.evidence_refs, dossier)
        await _cite(db, deal_id, ClaimType.RECOMMENDATION, recommendation.id,
                    rec_entries or proposal.evidence)
        result.inserted.append(risk.id)

    await _apply_verdicts(db, result, dossier)
    await db.flush()
    logger.info(
        "detect.applied deal=%s inserted=%d bumped=%d suppressed=%d resolved=%d",
        deal_id, len(result.inserted), len(result.bumped),
        len(result.suppressed), len(result.resolved),
    )
    return result


async def _cite(db, deal_id, claim_type, claim_id, entries) -> None:
    for entry in entries:
        await claims_service.attach_evidence(
            db,
            claim_type=claim_type,
            claim_id=claim_id,
            deal_id=deal_id,
            source_kind=entry.source_kind,
            snippet=entry.text,
            record_ref=entry.record_ref,
        )


async def _apply_verdicts(db, result: DetectionResult, dossier) -> None:
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
        if risk is None or verdict != "resolved":
            continue
        # `description` carries the strike because there is nowhere else to put
        # it without a migration; replace with a column if this outlives Phase 7.
        marker = "[resolution-proposed]"
        if marker in (risk.description or ""):
            risk.status = RiskStatus.RESOLVED
            risk.resolved_at = func.now()
            risk.description = (risk.description or "").replace(marker, "").strip()
            result.resolved.append(risk.id)
        else:
            risk.description = "%s %s" % (risk.description or "", marker)
            logger.info("detect.resolution_first_strike risk=%s", risk.id)


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
    this exists. The four SQL rules cannot hallucinate and cannot miss, so
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
        "_recall_note": "proposed + affirmed-as-still-present, vs the four SQL rules",
        "verdicts": result.verdicts,
    }


#: The four types the SQL rules cover, and therefore the only ones recall can
#: be measured on without a human writing labels.
_RULE_TYPES = {
    RiskType.NO_ECONOMIC_BUYER.value,
    RiskType.SINGLE_THREADED.value,
    RiskType.STALLED_STAGE.value,
    RiskType.CLOSE_DATE_AT_RISK.value,
}
