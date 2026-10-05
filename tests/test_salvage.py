"""The stray-element salvage, from the shapes task 11.9 actually observed.

Three detect runs in ten failed because the model put a spurious scalar inside
the `risks` array -- a bare `""` between two valid risks, or a truncated
`'{""risk_type"'` -- and two or three good risks were discarded with it. These
tests use those shapes, not invented ones, and pin both the recovery and the
limits on it: the salvage must not turn a response the model got badly wrong
into data.
"""

import json

import pytest
from pydantic import BaseModel

from app.ai import client


class Risk(BaseModel):
    risk_type: str
    title: str


class Detection(BaseModel):
    risks: list


class StrictDetection(BaseModel):
    risks: list[Risk]


def _payload(*risks):
    return {"risks": list(risks)}


GOOD_A = {"risk_type": "security_review_pending", "title": "Security sign-off pending"}
GOOD_B = {"risk_type": "missed_commitment", "title": "SOC 2 report overdue"}


# --------------------------------------------------------------------------
# The stripper
# --------------------------------------------------------------------------


def test_drops_the_empty_string_between_two_valid_risks():
    """Runs 4 and 5: `[{...}, "", {...}]`."""
    cleaned, dropped = client._strip_stray_elements(_payload(GOOD_A, "", GOOD_B))
    assert cleaned == _payload(GOOD_A, GOOD_B)
    assert len(dropped) == 1
    assert "$.risks" in dropped[0]


def test_drops_the_truncated_object_fragment():
    """Run 1: a half-written `'{""risk_type"'` as an array element."""
    cleaned, dropped = client._strip_stray_elements(
        _payload(GOOD_A, '{""risk_type"', GOOD_B)
    )
    assert cleaned["risks"] == [GOOD_A, GOOD_B]
    assert len(dropped) == 1


def test_a_clean_payload_is_returned_unchanged_and_reports_nothing():
    payload = _payload(GOOD_A, GOOD_B)
    cleaned, dropped = client._strip_stray_elements(payload)
    assert cleaned == payload
    assert dropped == []


def test_a_list_of_scalars_is_left_alone():
    """`evidence_refs` is a list of strings and must not be emptied."""
    payload = {"risks": [{"risk_type": "x", "title": "y", "evidence_refs": ["f1", "f2"]}]}
    cleaned, dropped = client._strip_stray_elements(payload)
    assert cleaned == payload
    assert dropped == []


def test_strays_nested_inside_an_object_are_also_dropped():
    payload = {"risks": [{"risk_type": "x", "title": "y", "sub": [GOOD_A, 7, GOOD_B]}]}
    cleaned, dropped = client._strip_stray_elements(payload)
    assert cleaned["risks"][0]["sub"] == [GOOD_A, GOOD_B]
    assert len(dropped) == 1


def test_an_empty_list_is_not_treated_as_a_list_of_objects():
    payload = {"risks": [], "open_risk_verdicts": []}
    assert client._strip_stray_elements(payload) == (payload, [])


# --------------------------------------------------------------------------
# The salvage, end to end
# --------------------------------------------------------------------------


def test_salvage_revalidates_and_returns_the_model():
    raw = json.dumps(_payload(GOOD_A, "", GOOD_B))
    parsed, dropped = client._salvage(StrictDetection, raw, "detect")
    assert parsed is not None
    assert [r.risk_type for r in parsed.risks] == [
        "security_review_pending", "missed_commitment",
    ]
    assert len(dropped) == 1


def test_salvage_declines_a_clean_payload_that_simply_failed():
    """No strays means the failure was something else -- do not mask it."""
    raw = json.dumps({"risks": [{"risk_type": "x"}]})  # missing `title`
    parsed, _ = client._salvage(StrictDetection, raw, "detect")
    assert parsed is None


def test_salvage_declines_unparseable_json():
    """A truncated *response* is out of scope: repairing it means guessing."""
    parsed, dropped = client._salvage(StrictDetection, '{"risks": [{"risk_t', "detect")
    assert parsed is None
    assert dropped == []


def test_salvage_declines_when_there_is_nothing_to_read():
    assert client._salvage(StrictDetection, None, "detect") == (None, [])
    assert client._salvage(StrictDetection, "", "detect") == (None, [])


def test_too_many_strays_is_a_lost_response_not_a_salvage():
    """A model that lost the shape must fail, not be averaged into data."""
    raw = json.dumps(_payload(GOOD_A, "", "", "", "", GOOD_B))
    parsed, dropped = client._salvage(StrictDetection, raw, "detect")
    assert parsed is None
    assert len(dropped) > client._MAX_SALVAGED_DROPS


# --------------------------------------------------------------------------
# Reading the rejected text off each failure shape
# --------------------------------------------------------------------------


def test_failed_generation_is_read_from_a_groq_400():
    exc = Exception("400")
    exc.body = {"error": {"code": "json_validate_failed", "failed_generation": '{"risks":[]}'}}
    assert client._failed_generation(exc) == '{"risks":[]}'


def test_an_unrelated_400_carries_nothing_to_salvage():
    exc = Exception("400")
    exc.body = {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}
    assert client._failed_generation(exc) is None
    assert client._failed_generation(Exception("boom")) is None


def test_content_is_read_from_a_completion_that_arrived():
    class Raw:
        content = '{"risks": []}'

    assert client._content_of(Raw()) == '{"risks": []}'

    class Parts:
        content = [{"text": '{"risks"'}, {"text": ": []}"}]

    assert client._content_of(Parts()) == '{"risks": []}'
    assert client._content_of(object()) is None
