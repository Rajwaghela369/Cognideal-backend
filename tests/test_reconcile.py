"""Supersession candidate selection -- can the right old fact be seen at all?

``supersede_facts`` asks a model to pick, from a list we supply, the old fact a
new one replaces. The model can only ever be as good as that list: a candidate
the query never selected is a candidate the model cannot choose, and the old
fact then stays ``accepted`` forever while a newer one contradicts it. The
panel reports both, each with a valid citation, and nothing is wrong with the
evidence -- only with the shortlist.

So these tests hold the model constant and vary the shortlist. The stub is a
perfect oracle: it reads the numbered list out of the rendered prompt and
answers with the number of the fact carrying the marker, or null when the
marker is absent. Every failure here is therefore a retrieval failure, not a
judgement one -- which is the only kind of failure the selection query can fix.
"""

import re
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.ai import client, reconcile
from app.models import Commitment, ExtractedFact
from app.models.enums import CommitmentStatus, FactStatus, FactType, OwnerSide

#: Appears in exactly one stored fact -- the one the new fact revises.
MARKER = "annual contract value is 220000 USD"

#: Shares the fact_type and the shape, shares no distinguishing content.
NOISE = "procurement needs a signed W-9 before invoice %d can be raised"


@pytest.fixture
def oracle(monkeypatch):
    """A model that is always right, so only the shortlist can be wrong."""
    calls = []

    async def fake_structured(schema, messages, **kwargs):
        rendered = "\n".join(content for _role, content in messages)
        calls.append(rendered)
        # The new fact and the shortlist are both in the rendered user message.
        # Split them: a marker in the shortlist alone means nothing -- the
        # answer is only yes when the *new* fact is the one revising it.
        new_fact, _, shortlist = rendered.partition("Existing facts of the same kind")
        choice = None
        if MARKER in new_fact:
            for number, text in re.findall(r"^(\d+)\. (.*)$", shortlist, re.MULTILINE):
                if MARKER in text:
                    choice = int(number)
                    break
        return (
            reconcile.Choice(choice=choice, reasoning="oracle"),
            client.AIRun("validate", "test-model", "supersede@1"),
        )

    monkeypatch.setattr(reconcile.client, "structured", fake_structured)
    return calls


@pytest_asyncio.fixture
async def seed(db, deal_id):
    """Seeds facts, and rolls back however the test ends.

    The rollback is a finalizer rather than a line at the end of each test
    because a *failing* assertion must still release the transaction: these
    rows reference ``deals``, and the ``deal_id`` fixture tears down by
    deleting the account from a second session. An open transaction holding a
    foreign-key lock turns a one-line assertion failure into a test run that
    hangs until the database gives up -- which is what happened when this file
    was first written.
    """

    async def _seed(*, noise_count: int):
        return await _plant(db, deal_id, noise_count=noise_count)

    try:
        yield _seed
    finally:
        await db.rollback()


async def _plant(db, deal_id, *, noise_count: int):
    """One revisable fact, older than ``noise_count`` unrelated ones.

    The marker fact is deliberately the oldest: recency ordering is exactly
    what pushes it out of the shortlist, and a fact being old is not evidence
    that it has stopped mattering.
    """
    base = datetime.now(timezone.utc) - timedelta(days=noise_count + 2)
    target = ExtractedFact(
        deal_id=deal_id,
        fact_type=FactType.BUDGET.value,
        content="The %s." % MARKER,
        status=FactStatus.ACCEPTED,
        extracted_at=base,
    )
    db.add(target)
    for i in range(noise_count):
        db.add(
            ExtractedFact(
                deal_id=deal_id,
                fact_type=FactType.BUDGET.value,
                content=NOISE % i,
                status=FactStatus.ACCEPTED,
                extracted_at=base + timedelta(days=i + 1),
            )
        )
    new_fact = ExtractedFact(
        deal_id=deal_id,
        fact_type=FactType.BUDGET.value,
        content="The %s -- revised, the CFO approved 265000 USD instead." % MARKER,
        status=FactStatus.PENDING,
        extracted_at=datetime.now(timezone.utc),
    )
    db.add(new_fact)
    await db.flush()
    return target, new_fact


@pytest.mark.asyncio
async def test_a_revision_supersedes_the_fact_it_revises(db, deal_id, oracle, seed):
    """The baseline: few enough facts that every one is a candidate."""
    target, new_fact = await seed(noise_count=3)

    result = await reconcile.supersede_facts(db, deal_id, [new_fact])

    assert result == [(target.id, new_fact.id)]
    assert target.status == FactStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_the_fact_to_supersede_is_found_past_the_ten_most_recent(
    db, deal_id, oracle, seed
):
    """The gap. Twelve accepted budget facts, the revisable one the oldest.

    Under ``ORDER BY extracted_at DESC LIMIT 10`` the shortlist is twelve
    rows' worth of noise truncated to ten, and the one fact that is actually
    about the same thing as the new one is not in it. Relevance, not recency,
    is what the limit should be applied to.
    """
    target, new_fact = await seed(noise_count=12)

    result = await reconcile.supersede_facts(db, deal_id, [new_fact])

    assert result == [(target.id, new_fact.id)], (
        "the fact being revised was not offered to the model: "
        "candidate selection dropped it"
    )
    assert target.status == FactStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_an_unrelated_new_fact_supersedes_nothing(db, deal_id, oracle, seed):
    """The floor the oracle protects: no marker in the list, no supersession.

    Worth asserting alongside the gap above, because the obvious way to make
    that test pass -- widen the shortlist until everything is in it -- is also
    the way to make the model's job harder. A relevance-ordered shortlist must
    still be allowed to come back empty.
    """
    _target, _new = await seed(noise_count=3)
    unrelated = ExtractedFact(
        deal_id=deal_id,
        fact_type=FactType.BUDGET.value,
        content="Legal asked whether the MSA permits monthly rather than annual billing.",
        status=FactStatus.PENDING,
        extracted_at=datetime.now(timezone.utc),
    )
    db.add(unrelated)
    await db.flush()

    assert await reconcile.supersede_facts(db, deal_id, [unrelated]) == []

    remaining = list((await db.scalars(
        select(ExtractedFact.status).where(
            ExtractedFact.deal_id == deal_id,
            ExtractedFact.status == FactStatus.SUPERSEDED,
        )
    )).all())
    assert remaining == []


# --------------------------------------------------------------------------
# Commitment reconciliation: the same shortlist, one call instead of many
# --------------------------------------------------------------------------

#: The one commitment these facts show satisfied.
KEPT = "circulate the completed SOC 2 Type II report to the security reviewers"


@pytest.mark.asyncio
async def test_a_satisfied_commitment_survives_the_cap(db, deal_id, monkeypatch):
    """Thirty open commitments, one of them answered, and a list of ten.

    The cap is what makes this worth a test. Before it the prompt grew with the
    deal and the satisfied commitment was always somewhere in it; now it has to
    be in the ten that relevance ordering keeps -- and the due date it happens
    to carry must not decide that, which is why this one has none while the
    noise is all due sooner.
    """
    for i in range(30):
        db.add(
            Commitment(
                deal_id=deal_id,
                description="Confirm the invoicing address for subsidiary %d" % i,
                owner_side=OwnerSide.CUSTOMER,
                due_date=date(2026, 11, 1) + timedelta(days=i),
                status=CommitmentStatus.PENDING,
            )
        )
    kept = Commitment(
        deal_id=deal_id,
        description="Dana to %s" % KEPT,
        owner_side=OwnerSide.CUSTOMER,
        due_date=None,
        status=CommitmentStatus.PENDING,
    )
    db.add(kept)
    await db.flush()

    shortlists = []

    async def fake_structured(schema, messages, **kwargs):
        rendered = "\n".join(content for _role, content in messages)
        items = re.findall(r"^(\d+)\. (.*)$", rendered, re.MULTILINE)
        shortlists.append(items)
        choice = next((int(n) for n, text in items if KEPT in text), None)
        return (
            reconcile.Choice(choice=choice, reasoning="oracle"),
            client.AIRun("validate", "test-model", "reconcile@1"),
        )

    monkeypatch.setattr(reconcile.client, "structured", fake_structured)

    fact = ExtractedFact(
        deal_id=deal_id,
        fact_type=FactType.COMMITMENT.value,
        content="Dana sent over the completed SOC 2 Type II report this morning.",
        status=FactStatus.PENDING,
    )
    db.add(fact)
    await db.flush()

    try:
        result = await reconcile.reconcile_commitments(db, deal_id, [fact])

        assert result and result[0][0] == kept.id, (
            "the answered commitment was not on the shortlist"
        )
        # Both halves matter: the prompt is bounded, and the one commitment
        # worth reading is inside the bound. Either alone would pass for the
        # wrong reason.
        commitments = [text for _n, text in shortlists[0] if "due" in text]
        assert len(commitments) <= reconcile.CANDIDATES
    finally:
        await db.rollback()
