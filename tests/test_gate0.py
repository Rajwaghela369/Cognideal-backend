"""Gate 0 and the extraction schemas -- tasks 3.1, 3.2, 3.3. Mostly no database.

The literal rule's tests are the ones to read first: they pin the distinction
the whole fixture corpus is built around. A figure the model invented must be
rejected; a figure spoken aloud must not be looked for in the first place.
"""

import pytest
from pydantic import BaseModel

from app.ai import extract
from app.ai.schemas import (
    ExtractionResult,
    FactPayload,
    narrow_payload,
    strict_schema_problems,
)
from app.models.enums import VerificationStatus
from app.services import gate0


def payload(**values):
    """A FactPayload with everything null except what is named."""
    base = dict.fromkeys(FactPayload.model_fields, None)
    base.update(values)
    return FactPayload(**base)


class _Chunk:
    def __init__(self, chunk_id, index, content):
        self.id = chunk_id
        self.chunk_index = index
        self.content = content
        self.chunk_metadata = {"char_start": 0, "char_end": len(content)}


# --------------------------------------------------------------------------
# 3.1 the wire schema
# --------------------------------------------------------------------------


def test_the_wire_schema_satisfies_openai_strict_mode():
    """Checked locally so a schema change that breaks strict mode fails here.

    Otherwise it would surface only as a 400 from OpenAI at request time.
    """
    assert strict_schema_problems(ExtractionResult) == []


def test_every_payload_field_is_required_and_nullable():
    """Strict mode has no concept of an optional field.

    A Pydantic field with `= None` is omitted from `required`, so absence must
    be expressed as present-but-null instead.
    """
    schema = ExtractionResult.model_json_schema()["$defs"]["FactPayload"]
    assert set(schema["required"]) == set(schema["properties"])
    for name, spec in schema["properties"].items():
        assert "null" in [b.get("type") for b in spec["anyOf"]], name


def test_payload_narrowing_keeps_only_the_relevant_keys():
    narrowed, error = narrow_payload("budget", payload(amount="a hundred and fifty thousand", what="x"))
    assert error is None
    assert narrowed == {"amount": "a hundred and fifty thousand", "currency": None}


def test_payload_narrowing_rejects_a_budget_with_no_figure():
    narrowed, error = narrow_payload("budget", payload(what="something"))
    assert narrowed is None and "amount" in error


def test_payload_narrowing_validates_against_the_real_enum():
    _, error = narrow_payload("commitment", payload(what="send it", owner_side="both"))
    assert error is not None, "owner_side must be one of OwnerSide"


# --------------------------------------------------------------------------
# 3.2 locating the snippet
# --------------------------------------------------------------------------


def test_locate_finds_an_exact_quote():
    chunks = [_Chunk("a", 0, "Priya Raman: Our auditors want twelve months.")]
    chunk, index, start, end = extract.locate("Our auditors want twelve months.", chunks)
    assert (index, chunks[0].content[start:end]) == (0, "Our auditors want twelve months.")


def test_locate_tolerates_case_but_stores_the_document_text():
    """A model quoting mid-sentence lowercases the first letter.

    That is the same quote, so it must resolve -- but what gets stored is the
    document's characters, because Gate 0 re-checks them forever.
    """
    chunks = [_Chunk("a", 0, "Tom Alvarez: We'd have to put it out to tender.")]
    found = extract.locate("we'd have to put it out to tender.", chunks)
    assert found is not None
    _, _, start, end = found
    assert chunks[0].content[start:end].startswith("We'd")


def test_locate_refuses_a_paraphrase():
    chunks = [_Chunk("a", 0, "Priya Raman: Our auditors want twelve months.")]
    assert extract.locate("The auditors require twelve months of retention", chunks) is None


def test_windows_are_even_not_greedy():
    """A greedy split leaves a one-chunk remainder, the worst input shape for a
    task that depends on reading the surrounding exchange."""
    chunks = [_Chunk(i, i, "x") for i in range(5)]
    assert [len(w) for w in extract.build_windows(chunks, size=4)] == [3, 2]


def test_render_window_trims_chunk_overlap():
    """Joining overlapping chunks raw repeats a passage at every boundary,
    which invites the model to extract the same fact twice."""
    a = _Chunk("a", 0, "Hello world abcdef")
    b = _Chunk("b", 1, "abcdef and then more")
    b.chunk_metadata = {"char_start": 12, "char_end": 32}
    assert extract.render_window([a, b]) == "Hello world abcdef\n and then more"


# --------------------------------------------------------------------------
# 3.3 the literal rule
# --------------------------------------------------------------------------


def test_the_literal_rule_catches_figures():
    assert gate0.literals_in("Budget approved at $150,000 on 2026-07-28") == [
        "$150,000", "2026-07-28",
    ]


def test_the_literal_rule_ignores_spoken_numbers():
    """The rule exists to catch a model *converting* speech into a
    precise-looking literal that was never said. "A hundred and fifty
    thousand" is what the transcript actually contains, so there is nothing to
    verify against -- looking for it would reject true claims."""
    assert gate0.literals_in("Security budget is about a hundred and fifty thousand") == []
    assert gate0.literals_in("The SOC 2 is due by the seventh of August") == []


def test_normalising_lets_punctuation_differ():
    assert gate0._normalise("$180,000") == gate0._normalise("180000")


# --------------------------------------------------------------------------
# 3.3 / 3.4 the claim-level verdict
# --------------------------------------------------------------------------


async def test_a_document_span_verifies(db):
    """Needs no document: an unknown chunk id is the failure path."""
    check = await gate0.check_document_span(db, None, "anything", 0, 5)
    assert check.status == VerificationStatus.SPAN_MISSING


async def test_an_unresolvable_table_is_refused(db):
    """`record_ref` is jsonb a model wrote. Resolving an arbitrary table name
    from it would be an injection point into our own schema."""
    check = await gate0.check_record_ref(
        db, {"table": "pg_user", "id": "x", "field": "passwd"}, "secret"
    )
    assert check.status == VerificationStatus.SPAN_MISSING
    assert "not resolvable" in (check.detail or "")


async def test_a_derived_reference_with_no_row_verifies(db):
    """"No economic buyer is listed" points at a table and a field but no row.
    There is nothing that can drift."""
    check = await gate0.check_record_ref(
        db, {"table": "deal_contacts", "field": "buying_role"}, "none listed"
    )
    assert check.status == VerificationStatus.VERIFIED


async def test_a_claim_with_no_citations_fails(db):
    check = await gate0.check_claim(db, "The budget is approved", [])
    assert not check.passed
    assert check.reason == "no evidence cited"


async def test_an_unsupported_literal_fails_the_whole_claim(db):
    """The claim cites a real span, but names a figure the span does not."""
    check = await gate0.check_claim(
        db,
        "Budget approved at $150,000",
        [{"source_kind": "record", "record_ref": {"table": "deals", "field": "value"},
          "snippet": "a hundred and fifty thousand"}],
    )
    assert not check.passed
    assert "$150,000" in (check.reason or "")
