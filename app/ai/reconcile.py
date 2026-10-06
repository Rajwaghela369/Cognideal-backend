"""Cross-meeting comparison: commitments kept, and facts superseded. Phase 6.

Both tasks here look backwards, which makes them different from extraction:
they compare what this call said against what the deal already holds. Both
answer with an **index into a list we supplied**, never an id -- the dossier's
handle pattern, so an impossible answer is detectable.

Neither writes the thing it concludes.

*Reconciliation* produces a **proposal**, not a status change. Closing a
commitment is a statement that someone did what they said, and
``risk -> recommendation -> [human accepts] -> task`` is the product: the model
may notice, a person decides. A model that silently closed commitments would
make ``missed_commitment`` unreliable in the one direction that matters.

*Supersession* does write ``status='superseded'`` on the older fact, and that
is deliberate rather than inconsistent: it hides nothing and deletes nothing.
Both rows stay, both keep their evidence, and the schema is explicit that
contradiction must never overwrite (docs/schema/README.md section 5).
"""

import logging
import uuid
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client
from app.ai.prompts import get as get_prompt
from app.models import Commitment, ExtractedFact
from app.models.enums import CommitmentStatus, FactStatus

logger = logging.getLogger("cognideal.ai.reconcile")

#: How many old facts a supersession decision may choose between. The model
#: reads the whole list, so this is a prompt-size and attention budget, not a
#: database one.
CANDIDATES = 10

#: Trigram floor for being on that list at all. Deliberately low, and the
#: numbers are measured rather than guessed: on real fact pairs a verbatim
#: revision scores ~0.57, a revision reworded end to end ~0.19, two facts that
#: merely share deal vocabulary ~0.13, and unrelated ones ~0.08. 0.3 -- the
#: pg_trgm default -- would discard the reworded revision, which is the case
#: this whole function exists for.
#:
#: The gap between the weakest keep (0.19) and the strongest drop (0.13) is
#: narrow, so treat the floor as pruning the long tail, not as a decision: the
#: ordering is what puts the likely answer in front of the model, and the model
#: still answers null whenever it is unsure.
SIMILARITY_FLOOR = 0.15


class Choice(BaseModel):
    """An index into a numbered list, or null. Shared by both tasks."""

    model_config = ConfigDict(extra="forbid")

    choice: Optional[int] = Field(description="The item's number, or null if none")
    reasoning: str = Field(description="One sentence")


def _numbered(items: Sequence[str]) -> str:
    return "\n".join("%d. %s" % (i, text) for i, text in enumerate(items, start=1))


def _validated_choice(answer: Choice, size: int, what: str) -> Optional[int]:
    """Turn the model's number into a zero-based index, or None.

    Out of range is treated as "none" rather than clamped. A model that cannot
    count is not one to trust with the decision, and this is the whole reason
    the answer is an integer against a list we supplied.
    """
    if answer.choice is None:
        return None
    if not 1 <= answer.choice <= size:
        logger.warning("%s.out_of_range choice=%s size=%d", what, answer.choice, size)
        return None
    return answer.choice - 1


async def reconcile_commitments(
    db: AsyncSession,
    deal_id: uuid.UUID,
    facts: Sequence[Any],
    *,
    budget: Optional[client.RunBudget] = None,
) -> List[Tuple[uuid.UUID, str]]:
    """Which open commitments do these facts show satisfied?

    Returns ``[(commitment_id, why)]`` -- proposals for a human, written by the
    caller as recommendations. One call for the whole fact set rather than one
    per commitment: the comparison needs to see them together, and a per-pair
    loop would cost a call per commitment for a worse answer.

    That single call is why the list is capped. It was unbounded, which is fine
    for a deal with six open commitments and not fine for one with sixty: the
    prompt grows without limit, every item competes for the same attention, and
    the answer is one index into all of it. Ordering is by best trigram
    similarity to any of these facts, so when the cap does cut, what survives
    is what these facts are about.

    **No similarity floor here**, unlike `supersede_facts`. A commitment and
    the fact that shows it satisfied routinely share almost no wording -- "send
    the SOC 2 report" against "Dana circulated the security pack" -- so a floor
    would discard exactly the pairs worth catching. Trigram only orders; the
    due-date tiebreak decides the rest, and every open commitment stays
    eligible.
    """
    if not facts:
        return []

    # `greatest` over the facts rather than against them concatenated: a
    # commitment matching one fact strongly should outrank one matching all of
    # them weakly, and a blob's trigram union depresses every row alike.
    relevance = func.greatest(
        *[func.similarity(Commitment.description, f.content) for f in facts]
    )
    open_commitments = list(
        (
            await db.execute(
                select(Commitment)
                .where(
                    Commitment.deal_id == deal_id,
                    Commitment.status == CommitmentStatus.PENDING,
                )
                .order_by(relevance.desc(), Commitment.due_date.nulls_last())
                .limit(CANDIDATES)
            )
        ).scalars()
    )
    if not open_commitments:
        return []

    prompt = get_prompt("reconcile")
    answer, _run = await client.structured(
        Choice,
        prompt.messages(
            commitments=_numbered([
                "%s (owner: %s, due %s)" % (c.description, c.owner_name or c.owner_side, c.due_date)
                for c in open_commitments
            ]),
            facts=_numbered(["[%s] %s" % (f.fact_type, f.content) for f in facts]),
        ),
        task="validate",
        prompt_version=prompt.version,
        reasoning_effort="medium",
        budget=budget,
    )

    index = _validated_choice(answer, len(open_commitments), "reconcile")
    if index is None:
        logger.info("reconcile.none deal=%s why=%s", deal_id, answer.reasoning[:70])
        return []

    commitment = open_commitments[index]
    logger.info(
        "reconcile.proposed deal=%s commitment=%s why=%s",
        deal_id, commitment.id, answer.reasoning[:70],
    )
    return [(commitment.id, answer.reasoning.strip())]


async def supersede_facts(
    db: AsyncSession,
    deal_id: uuid.UUID,
    new_facts: Sequence[Any],
    *,
    budget: Optional[client.RunBudget] = None,
) -> List[Tuple[uuid.UUID, uuid.UUID]]:
    """Mark older facts superseded where a new one replaces them.

    Returns ``[(old_fact_id, new_fact_id)]``. Compares only within a
    ``fact_type`` -- a budget fact cannot supersede a deadline -- and only
    against **accepted** facts, because a `pending` fact is itself unreviewed
    and superseding one proposal with another would decide nothing.

    The shortlist is chosen **per new fact, by trigram similarity to it**, and
    that is the load-bearing part. It was once `extracted_at DESC LIMIT 10`:
    recency as a proxy for relevance, which holds only while a deal has fewer
    than ten accepted facts of a type. Past that the one fact genuinely about
    the same subject can fall outside the window, the model never sees it, and
    it stays `accepted` while a newer fact contradicts it -- both on the panel,
    both correctly cited. See `tests/test_reconcile.py`.

    The call count does not change: it was already one per new fact. It can
    only fall, because a new fact whose shortlist comes back empty now skips
    the call entirely instead of being compared against ten unrelated facts.
    """
    if not new_facts:
        return []

    superseded: List[Tuple[uuid.UUID, uuid.UUID]] = []
    prompt = get_prompt("supersede")
    # Never offer a fact this run has already retired, and never offer the new
    # facts themselves. The second was always needed; the first became needed
    # when selection moved per-fact, because two new facts of one type now get
    # two queries and the same old row would otherwise be marked twice.
    excluded = {f.id for f in new_facts}

    for new_fact in new_facts:
        relevance = func.similarity(ExtractedFact.content, new_fact.content)
        existing = list(
            (
                await db.execute(
                    select(ExtractedFact)
                    .where(
                        ExtractedFact.deal_id == deal_id,
                        ExtractedFact.fact_type == new_fact.fact_type,
                        ExtractedFact.status == FactStatus.ACCEPTED,
                        ExtractedFact.id.notin_(excluded),
                        relevance > SIMILARITY_FLOOR,
                    )
                    .order_by(relevance.desc())
                    .limit(CANDIDATES)
                )
            ).scalars()
        )
        if not existing:
            continue

        answer, _run = await client.structured(
            Choice,
            prompt.messages(
                fact_type=new_fact.fact_type,
                new_fact=new_fact.content,
                existing=_numbered([f.content for f in existing]),
            ),
            task="validate",
            prompt_version=prompt.version,
            reasoning_effort="medium",
            budget=budget,
        )
        index = _validated_choice(answer, len(existing), "supersede")
        if index is None:
            continue

        old = existing[index]
        # Marked, never overwritten or deleted: both rows stay, both keep
        # their evidence. "Do not overwrite -- mark the old claim
        # superseded, link the new evidence, keep both."
        old.status = FactStatus.SUPERSEDED
        old.reviewed_at = func.now()
        excluded.add(old.id)
        superseded.append((old.id, new_fact.id))
        logger.info(
            "supersede.marked old=%s new=%s type=%s why=%s",
            old.id, new_fact.id, new_fact.fact_type, answer.reasoning[:70],
        )

    if superseded:
        await db.flush()
    return superseded
