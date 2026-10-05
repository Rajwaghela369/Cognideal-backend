"""The extractor's wire contract, and the per-type payload shapes -- task 3.1.

Two layers, deliberately:

**The wire schema** (:class:`ExtractionResult`) is what the model must satisfy,
and it is flat. Groq's strict mode constrains what a schema may look like:
every field must appear in ``required`` and every object must set
``additionalProperties: false``. That rules out the obvious design -- a
free-form ``Dict[str, str]`` payload renders as ``additionalProperties:
{"type": "string"}``, which is a schema rather than ``false`` -- and it also
rules out defaulted optionals, since a Pydantic field with ``= None`` is
omitted from ``required``. So every field here is declared *without* a default
and typed ``Optional`` where absence is meaningful: present-but-null, never
missing.

**The payload models** are the semantic contract, checked in Python after the
response parses. They are what makes a ``budget`` fact promotable into a real
amount later, and they catch the failure a flat schema cannot express: a
``commitment`` with no owner, a ``budget`` with no figure.

Note what is *not* here. The literal rule (docs/schema/README.md section 5)
reads the claim text and the cited span, not the payload -- so Gate 0 does not
depend on any of this. Payload structure matters at promotion time, which is
Gate 3.
"""

from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    ActionType,
    FactType,
    OwnerSide,
    Priority,
    RiskType,
    Sentiment,
    Severity,
)

# --------------------------------------------------------------------------
# the wire: what the model returns
# --------------------------------------------------------------------------


class FactPayload(BaseModel):
    """The union of fields any fact type needs, flat and all nullable.

    One flat object rather than a discriminated union per ``fact_type``: a 20B
    model asked to pick a branch and fill it correctly gets both wrong more
    often than it gets one wrong, and a union's schema is large enough to crowd
    out the transcript it is meant to be reading.

    Nine fields, none defaulted -- see the module docstring on why a default
    would break strict mode.
    """

    model_config = ConfigDict(extra="forbid")

    what: Optional[str] = Field(description="The thing itself, in a few words")
    amount: Optional[str] = Field(description="A money figure, exactly as spoken or written")
    currency: Optional[str] = Field(description="Currency if stated, else null")
    date: Optional[str] = Field(description="A date or deadline, exactly as spoken")
    owner_side: Optional[str] = Field(description="'us' or 'customer', for a commitment")
    owner_name: Optional[str] = Field(description="Who owns a commitment")
    party: Optional[str] = Field(description="A person or company named: stakeholder or competitor")
    role: Optional[str] = Field(description="That person's role, if stated")
    blocking: Optional[bool] = Field(description="True if an objection blocks progress")


class ExtractedFactOut(BaseModel):
    """One fact as the model reports it.

    ``snippet`` is the load-bearing field and the one the prompt works hardest
    on: it must be the **verbatim** quote, because Gate 0 locates it in the
    chunk by substring search and computes the offsets from that. The model is
    never asked for character positions -- index arithmetic is a needless
    failure mode, and offsets that are computed rather than asserted cannot
    drift (docs/ai/README.md section 4).
    """

    model_config = ConfigDict(extra="forbid")

    fact_type: FactType = Field(description="One of the allowed fact types")
    content: str = Field(description="The claim, in one sentence, in your own words")
    snippet: str = Field(description="The exact words from the transcript that support it, copied character for character")
    speaker: Optional[str] = Field(description="Who said it, as labelled in the transcript")
    confidence: float = Field(description="0.0 to 1.0, your own confidence")
    payload: FactPayload


class ExtractionResult(BaseModel):
    """What one extraction call returns.

    A list rather than one fact per call: a window holds several facts and
    re-sending the same text eight times to ask about one type each would cost
    eight times as much for a worse result, since the model could not see that
    a budget figure and its board threshold are the same exchange.
    """

    model_config = ConfigDict(extra="forbid")

    facts: List[ExtractedFactOut]


# --------------------------------------------------------------------------
# the semantics: per-type payload shapes, validated after parsing
# --------------------------------------------------------------------------


class RequirementPayload(BaseModel):
    what: str


class ObjectionPayload(BaseModel):
    what: str
    blocking: Optional[bool]


class StakeholderPayload(BaseModel):
    party: str
    role: Optional[str]


class CommitmentPayload(BaseModel):
    what: str
    owner_side: OwnerSide
    owner_name: Optional[str]
    date: Optional[str]


class DeadlinePayload(BaseModel):
    what: str
    date: str


class BudgetPayload(BaseModel):
    amount: str
    currency: Optional[str]


class CompetitorPayload(BaseModel):
    party: str


class DecisionCriteriaPayload(BaseModel):
    what: str


PAYLOAD_MODELS: Dict[str, Type[BaseModel]] = {
    FactType.REQUIREMENT.value: RequirementPayload,
    FactType.OBJECTION.value: ObjectionPayload,
    FactType.STAKEHOLDER.value: StakeholderPayload,
    FactType.COMMITMENT.value: CommitmentPayload,
    FactType.DEADLINE.value: DeadlinePayload,
    FactType.BUDGET.value: BudgetPayload,
    FactType.COMPETITOR.value: CompetitorPayload,
    FactType.DECISION_CRITERIA.value: DecisionCriteriaPayload,
}


def narrow_payload(fact_type: str, payload: FactPayload) -> Tuple[Optional[dict], Optional[str]]:
    """Validate a flat payload against its fact type and drop the irrelevant keys.

    Returns ``(payload_dict, None)`` or ``(None, reason)``. A rejection is not
    fatal to the fact: ``extracted_facts.payload`` is nullable, and a budget
    fact whose figure the model failed to isolate is still a cited claim a
    human can read. It simply cannot be *promoted* into a commitment or an
    amount, which is the thing the payload exists for.
    """
    model = PAYLOAD_MODELS.get(fact_type)
    if model is None:
        return None, "unknown fact_type %r" % fact_type

    supplied = payload.model_dump()
    relevant = {k: supplied.get(k) for k in model.model_fields}
    try:
        return model(**relevant).model_dump(mode="json"), None
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as a reason
        return None, "%s: %s" % (model.__name__, exc)


# --------------------------------------------------------------------------
# strict-mode conformance, asserted locally
# --------------------------------------------------------------------------


def strict_schema_problems(model: Type[BaseModel]) -> List[str]:
    """Why Groq would reject this schema under ``strict: true``.

    Checked here rather than discovered as a 400 in the container: this
    interpreter's langchain-groq cannot send ``strict`` at all
    (docs/ai/README.md section 7), so without a local check the constraint
    would go untested until a deployment that can.
    """
    problems: List[str] = []

    def walk(schema: dict, path: str) -> None:
        if schema.get("type") == "object" or "properties" in schema:
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            missing = sorted(set(properties) - required)
            if missing:
                problems.append("%s: not required: %s" % (path or ".", ", ".join(missing)))
            if schema.get("additionalProperties") is not False:
                problems.append("%s: additionalProperties must be false" % (path or "."))
            for name, child in properties.items():
                walk(child, "%s.%s" % (path, name))
        for key in ("items", "$ref"):
            if key == "items" and isinstance(schema.get("items"), dict):
                walk(schema["items"], path + "[]")
        for branch in schema.get("anyOf", []) or []:
            if isinstance(branch, dict):
                walk(branch, path)

    root = model.model_json_schema()
    definitions = root.get("$defs", {})

    def resolve(schema: dict, path: str) -> None:
        ref = schema.get("$ref")
        if ref and ref.startswith("#/$defs/"):
            resolve(definitions[ref.split("/")[-1]], path)
            return
        walk(schema, path)
        for name, child in (schema.get("properties") or {}).items():
            if isinstance(child, dict) and child.get("$ref"):
                resolve(child, "%s.%s" % (path, name))
        items = schema.get("items")
        if isinstance(items, dict):
            resolve(items, path + "[]")

    resolve(root, "")
    return sorted(set(problems))


# --------------------------------------------------------------------------
# Phase 5 -- meeting synthesis
# --------------------------------------------------------------------------


class SummaryOut(BaseModel):
    """The meeting summary, composed from the fact set."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Three to five sentences. Only what the facts say.")


class SentimentOut(BaseModel):
    """The one judgment outside the evidence contract.

    ``quotes`` are illustrative, for a human to disagree with -- not proof. No
    span *entails* "the call went badly", which is why sentiment never acquires
    a Gate 1 verdict.
    """

    model_config = ConfigDict(extra="forbid")

    sentiment: Sentiment = Field(description="positive, neutral, negative or unknown")
    quotes: List[str] = Field(description="Two or three short passages, copied exactly")
    reasoning: str = Field(description="One sentence on why")


# --------------------------------------------------------------------------
# Phase 7 -- AI risk and recommendation detection
# --------------------------------------------------------------------------


class RecommendationOut(BaseModel):
    """The action a risk implies. Produced in the same call as its risk.

    One call rather than two because the two enums were designed as a pair and
    a card is assembled from both -- and because two separate triggers can
    disagree about what a deal needs.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(description="One of the allowed action types")
    title: str = Field(description="The action, imperative, under ten words")
    description: str = Field(description="What to do, one or two sentences")
    rationale: str = Field(description="Why, quoting the deal's own specifics")
    priority: Priority = Field(description="low, medium, high or urgent")
    evidence_refs: List[str] = Field(description="Handles from the dossier only")


class RiskOut(BaseModel):
    """One proposed risk.

    ``risk_type`` is a closed enum and ``risk_key`` only applies to ``other``:
    together they are the identity the partial unique index keys on, which is
    what stops a re-run inserting a fifth copy of a risk it has already
    reported under different prose.
    """

    model_config = ConfigDict(extra="forbid")

    risk_type: RiskType = Field(description="One of the allowed risk types, or 'other'")
    risk_key: Optional[str] = Field(
        description="A short snake_case slug, ONLY when risk_type is 'other'. Else null."
    )
    title: str = Field(description="Under ten words")
    description: str = Field(description="What the risk is, two or three sentences")
    severity: Severity = Field(description="low, medium, high or critical")
    confidence: float = Field(description="0.0 to 1.0")
    evidence_refs: List[str] = Field(
        description="Handles from the dossier that justify this. Use only handles given."
    )
    recommendation: RecommendationOut


class OpenRiskVerdict(BaseModel):
    """Whether a risk already open is still true.

    Asked rather than inferred from silence. A model pass is not exhaustive --
    it may simply not mention a risk this run -- and resolving on absence makes
    the panel flicker, which reads as brokenness regardless of precision.
    """

    model_config = ConfigDict(extra="forbid")

    risk_id: str = Field(description="The id exactly as given in the open-risk list")
    verdict: str = Field(description="still_present, resolved or unclear")
    evidence_refs: List[str] = Field(description="Handles supporting the verdict")


class DetectionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risks: List[RiskOut]
    open_risk_verdicts: List[OpenRiskVerdict]
    proactive: List[RecommendationOut]


# --------------------------------------------------------------------------
# Phase 8 -- meeting briefs
# --------------------------------------------------------------------------


class BriefOut(BaseModel):
    """A pre-meeting brief assembled from already-prefetched deal state."""

    model_config = ConfigDict(extra="forbid")

    objectives: List[str] = Field(description="Two to five concrete objectives")
    key_risks: List[str] = Field(description="The risks most relevant to this meeting")
    recommended_questions: List[str] = Field(
        description="Questions the seller should ask, phrased verbatim"
    )
    context_summary: str = Field(
        description="A concise summary of the deal context relevant to this meeting"
    )


class ChatTitleOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="A short conversation title, at most six words")
