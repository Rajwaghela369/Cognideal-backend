"""Gate 1 -- the Evidence Validator. Task 4.1.

Gate 0 proved the citation resolves. This asks the separate question of whether
what it says actually supports the claim, which needs judgement and so needs a
model.

**The starved context is the mechanism, and it is enforced here rather than
requested in the prompt.** :func:`validate_claim` takes a claim string and a
list of snippets. It has no access to the transcript, the deal or the
extraction that produced the claim, because it is never given them -- a
prompt-level instruction to ignore context the caller supplied anyway is not a
constraint, it is a hope.

Independent of ``confidence``. The generator's self-report never reaches this
function: it is a weak signal, and letting a confident claim skip a gate is the
specific failure ``docs/schema/README.md`` section 5 warns about. The two are
*crossed* afterwards, in the metrics, where high confidence plus
``contradicted`` is the most valuable eval case there is.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.ai import client
from app.ai.prompts import get as get_prompt
from app.models.enums import Verdict

logger = logging.getLogger("cognideal.ai.validate")

#: Verdicts that must never render. `contradicted` is evidence of the opposite;
#: `unsupported` means the citation does not address the claim at all.
QUARANTINED = (Verdict.CONTRADICTED, Verdict.UNSUPPORTED)


class ValidationOut(BaseModel):
    """The validator's answer.

    ``verdict`` before ``rationale`` on purpose: the model writes fields in
    order, and a model that commits to a verdict first then justifies it is
    measurably less prone to reasoning its way into agreement than one that
    writes a paragraph and concludes from it.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict = Field(description="supported, partial, contradicted or unsupported")
    rationale: str = Field(description="One sentence. Name the assertion that failed, if one did.")


@dataclass
class Validation:
    verdict: Verdict
    rationale: str
    model: str
    validator_version: str

    @property
    def quarantined(self) -> bool:
        return self.verdict in QUARANTINED


def render_evidence(snippets: Sequence[str]) -> str:
    """Number the spans so a rationale can refer to one of them."""
    return "\n".join('%d. "%s"' % (i, s) for i, s in enumerate(snippets, start=1))


async def validate_claim(
    claim: str,
    snippets: Sequence[str],
    *,
    budget: Optional[client.RunBudget] = None,
) -> Optional[Validation]:
    """One claim, its spans, one verdict.

    Returns ``None`` when there is nothing to validate -- a claim with no
    surviving span should never have reached here, since Gate 0 rejects it, so
    this is a guard rather than a path.
    """
    if not snippets:
        return None

    prompt = get_prompt("validate")
    result, run = await client.structured(
        ValidationOut,
        prompt.messages(claim=claim, evidence=render_evidence(snippets)),
        task="validate",
        prompt_version=prompt.version,
        # The cheaper model would do for a narrow task, but this gate protects
        # every claim in the product and its prompt is tiny -- the strongest
        # model available is also nearly free here.
        role=client.ROLE_PRIMARY,
        reasoning_effort="medium",
        budget=budget,
    )
    return Validation(
        verdict=result.verdict,
        rationale=result.rationale.strip(),
        model=run.model,
        validator_version=prompt.version,
    )
