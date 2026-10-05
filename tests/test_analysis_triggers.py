"""Phase 10: event tiers, debounce, loop guards, and clock sweep."""

import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import select

import worker
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import ClaimEvidence, Commitment, Deal, Evidence, Risk
from app.models.enums import (
    ClaimType,
    CommitmentStatus,
    DealStage,
    Origin,
    OwnerSide,
    RecommendationStatus,
    RiskStatus,
    RiskType,
    SourceKind,
    VerificationStatus,
)
from app.services import analysis, deal as deal_service, risk as risk_service


async def _deal_state(deal_id):
    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        return (
            deal.analysis_dirty_first_at,
            deal.analysis_dirty_last_at,
            deal.analysis_dirty_reason,
            deal.analysis_swept_at,
        )


async def test_dirty_first_is_stable_and_last_moves(deal_id):
    async with SessionLocal() as db:
        await analysis.mark_dirty(db, deal_id, "first edit")
        await db.commit()
    first_at, first_last, reason, _ = await _deal_state(deal_id)
    assert first_at is not None and first_last is not None
    assert reason == "first edit"

    async with SessionLocal() as db:
        await analysis.mark_dirty(db, deal_id, "second edit")
        await db.commit()
    second_first, second_last, reason, _ = await _deal_state(deal_id)
    assert second_first == first_at
    assert second_last >= first_last
    assert reason == "second edit"


async def test_ai_origin_and_bulk_mode_do_not_enqueue(deal_id):
    async with SessionLocal() as db:
        assert not await analysis.mark_dirty(
            db, deal_id, "detector write", origin=Origin.AI
        )
        with analysis.suppress_tier2():
            for index in range(200):
                assert not await analysis.mark_dirty(
                    db, deal_id, "bulk row %d" % index
                )
        await db.commit()

    first_at, last_at, reason, _ = await _deal_state(deal_id)
    assert (first_at, last_at, reason) == (None, None, None)


async def test_stage_change_reverifies_and_marks_dirty_in_one_transaction(db, deal_id):
    evidence = Evidence(
        deal_id=deal_id,
        source_kind=SourceKind.RECORD,
        record_ref={"table": "deals", "id": str(deal_id), "field": "stage"},
        snippet=DealStage.DISCOVERY.value,
    )
    db.add(evidence)
    await db.flush()
    link = ClaimEvidence(
        claim_type=ClaimType.RISK,
        claim_id=uuid.uuid4(),
        evidence_id=evidence.id,
        verification_status=VerificationStatus.VERIFIED,
    )
    db.add(link)
    await db.flush()

    deal = await db.get(Deal, deal_id)
    await deal_service.apply_stage_change(db, deal, DealStage.NEGOTIATION)

    # No commit or second request: Tier 0 and dirty marking share the write.
    await db.refresh(link)
    await db.refresh(deal)
    assert link.verification_status == VerificationStatus.VALUE_DRIFTED
    assert deal.analysis_dirty_first_at is not None
    await db.commit()


async def test_non_relevant_edit_does_not_mark_dirty(db, deal_id):
    deal = await db.get(Deal, deal_id)
    deal.name = "A display-only rename"
    await db.commit()
    first_at, last_at, reason, _ = await _deal_state(deal_id)
    assert (first_at, last_at, reason) == (None, None, None)


async def test_deterministic_run_and_dismissal_leave_clean(db, deal_id):
    # Detector-authored risks/recommendations never feed the trigger back into
    # itself. A dismissal is deliberately another no-op for invalidation.
    await analysis.run_deal_analysis(db, deal_id)
    rec = SimpleNamespace(
        status=RecommendationStatus.SUGGESTED,
        dismissal_reason=None,
        dismissal_note=None,
        decided_at=None,
    )
    body = SimpleNamespace(reason=SimpleNamespace(value="not_relevant"), note=None)
    await risk_service.dismiss_recommendation(db, rec, body)
    await db.commit()
    first_at, last_at, reason, _ = await _deal_state(deal_id)
    assert (first_at, last_at, reason) == (None, None, None)


async def test_debounce_consumes_five_edits_once(monkeypatch, deal_id):
    for index in range(5):
        async with SessionLocal() as db:
            await analysis.mark_dirty(db, deal_id, "edit %d" % index)
            await db.commit()

    calls = []

    async def fake_run(db, claimed_id, budget=None):
        calls.append(claimed_id)

    monkeypatch.setattr(settings, "analysis_debounce_seconds", 0)
    monkeypatch.setattr(analysis, "run_deal_analysis", fake_run)
    assert await worker.poll_dirty_deals() is True
    assert await worker.poll_dirty_deals() is False
    assert calls == [deal_id]
    first_at, last_at, reason, swept_at = await _deal_state(deal_id)
    assert (first_at, last_at, reason) == (None, None, None)
    assert swept_at is not None


async def test_dirty_claim_is_single_flight(monkeypatch, deal_id):
    async with SessionLocal() as db:
        await analysis.mark_dirty(db, deal_id, "ready")
        await db.commit()
    monkeypatch.setattr(settings, "analysis_debounce_seconds", 0)

    async with SessionLocal() as first, SessionLocal() as second:
        claimed = await worker._claim_dirty_deal(first)
        skipped = await worker._claim_dirty_deal(second)
        assert claimed is not None and claimed.id == deal_id
        assert skipped is None


async def test_nightly_sweep_finds_a_time_only_missed_commitment(
    db, deal_id, monkeypatch
):
    commitment = Commitment(
        deal_id=deal_id,
        description="Send the signed security addendum",
        owner_side=OwnerSide.US,
        due_date=date.today() - timedelta(days=1),
        status=CommitmentStatus.PENDING,
        origin=Origin.USER,
    )
    db.add(commitment)
    deal = await db.get(Deal, deal_id)
    deal.analysis_swept_at = datetime.now(timezone.utc) - timedelta(days=2)
    await db.commit()

    # Keep this integration test scoped to its fixture even when a developer's
    # local database contains older real deals eligible for the global sweep.
    async def claim_fixture(session):
        return await session.scalar(
            select(Deal)
            .where(Deal.id == deal_id)
            .with_for_update(skip_locked=True)
        )

    monkeypatch.setattr(worker, "_claim_sweep_deal", claim_fixture)

    assert await worker.poll_sweep() is True
    async with SessionLocal() as check:
        missed = await check.scalar(
            select(Risk).where(
                Risk.deal_id == deal_id,
                Risk.risk_type == RiskType.MISSED_COMMITMENT.value,
                Risk.status == RiskStatus.OPEN,
            )
        )
        swept_at = await check.scalar(
            select(Deal.analysis_swept_at).where(Deal.id == deal_id)
        )
    assert missed is not None
    assert swept_at is not None


async def test_gone_quiet_is_part_of_deterministic_floor(db, deal_id, monkeypatch):
    monkeypatch.setattr(settings, "gone_quiet_days", 7)
    deal = await db.get(Deal, deal_id)
    deal.last_activity_at = datetime.now(timezone.utc) - timedelta(days=8)
    await analysis.refresh_deterministic(db, deal_id)
    risk = await db.scalar(
        select(Risk).where(
            Risk.deal_id == deal_id,
            Risk.risk_type == RiskType.GONE_QUIET.value,
            Risk.status == RiskStatus.OPEN,
        )
    )
    assert risk is not None
    await db.commit()
