"""The model half of attendee name resolution -- task 2.4.

Called only on what ``services/roster.py`` could not settle: a similarity score
in the band between "clearly nobody we know" and "clearly this person". On the
fixture corpus that is nothing at all, which is the intended shape -- the model
is the exception path, not the mechanism.

The deterministic half stays in ``services/roster.py`` on purpose. Resolution
has to work with the provider turned off, and ``no_economic_buyer`` cannot
depend on a model being reachable.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.ai import client
from app.ai.prompts import get as get_prompt
from app.services.roster import Candidate, Resolution

logger = logging.getLogger("cognideal.ai.tiebreak")


class SpeakerMatch(BaseModel):
    """The tiebreak's answer.

    ``choice`` is an index into the numbered candidate list, never a contact id.
    A model asked to echo a uuid will sometimes produce a plausible one that
    belongs to nobody; an integer is validated against a list we supplied, so an
    out-of-range answer is detectable rather than silently wrong.

    Both fields are required -- strict mode permits no optional keys -- so
    "no match" is expressed as ``choice: null``. ``extra="forbid"`` renders as
    ``additionalProperties: false``, which strict mode also requires.
    """

    model_config = ConfigDict(extra="forbid")

    choice: Optional[int] = Field(description="Candidate number, or null if none match")
    reasoning: str = Field(description="One sentence. Why this candidate, or why none.")


def _render(candidates: List[Candidate]) -> str:
    return "\n".join(
        "%d. %s (name similarity %.2f)" % (i, c.full_name, c.similarity)
        for i, c in enumerate(candidates, start=1)
    )


async def resolve_ambiguous(
    resolution: Resolution, *, budget: Optional[client.RunBudget] = None
) -> Resolution:
    """Ask the model to settle one ambiguous name. Mutates nothing in the DB.

    Returns the resolution unchanged if AI is off, if there is nothing to
    choose from, or if the answer is out of range -- in every one of those cases
    leaving ``contact_id`` NULL, which is the safe outcome.
    """
    if not resolution.candidates:
        return resolution

    prompt = get_prompt("roster_tiebreak")
    try:
        match, run = await client.structured(
            SpeakerMatch,
            prompt.messages(
                raw_name=resolution.raw_name, candidates=_render(resolution.candidates)
            ),
            task="resolve",
            prompt_version=prompt.version,
            role=client.ROLE_CHEAP,
            reasoning_effort="low",
            budget=budget,
        )
    except client.AIDisabled:
        # Not an error. Resolution is meant to work with the provider off.
        logger.info("tiebreak.skipped name=%r reason=ai_disabled", resolution.raw_name)
        return resolution

    if match.choice is None:
        logger.info("tiebreak.none name=%r why=%s", resolution.raw_name, match.reasoning)
        return resolution

    if not 1 <= match.choice <= len(resolution.candidates):
        # The one thing indices make detectable. Treat it as "no match": a
        # model that cannot count is not a model to trust with an attribution.
        logger.warning(
            "tiebreak.out_of_range name=%r choice=%s candidates=%d",
            resolution.raw_name, match.choice, len(resolution.candidates),
        )
        return resolution

    chosen = resolution.candidates[match.choice - 1]
    logger.info(
        "tiebreak.linked name=%r -> %s why=%s",
        resolution.raw_name, chosen.full_name, match.reasoning,
    )
    resolution.contact_id = chosen.contact_id
    resolution.similarity = chosen.similarity
    return resolution
