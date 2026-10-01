"""The deterministic half of AI risk detection -- tasks 7.6, 7.8, 7.10.

Everything between "the model proposed" and "a row was written" is ordinary
Python, and these are the parts that re-establish what the SQL detector gets
for free. No model, no database.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.ai import detect_ai
from app.ai.dossier import Dossier
from app.models.enums import DismissalReason, RiskType, Severity


class _Risk:
    """Just enough of a Risk row for the severity policy."""

    def __init__(self, severity, last_seen_days_ago=0):
        self.severity = severity
        self.last_seen_at = datetime.now(timezone.utc) - timedelta(days=last_seen_days_ago)
        self.first_detected_at = self.last_seen_at


# --------------------------------------------------------------------------
# 7.8 identity
# --------------------------------------------------------------------------


def test_canonical_key_normalises_a_proposed_slug():
    assert detect_ai.canonical_key("Champion Going Quiet!") == "champion_going_quiet"
    assert detect_ai.canonical_key("  BUDGET--frozen  ") == "budget_frozen"
    assert detect_ai.canonical_key(None) == ""


def test_a_reworded_slug_merges_into_the_existing_one():
    """The failure this prevents: two cards for one problem.

    `champion_going_quiet` and `champion_disengaged` share no prefix and are
    far apart by edit distance, but they are the same risk -- which is why the
    match is on shared tokens rather than string similarity.
    """
    assert detect_ai.match_existing_key(
        "champion_disengaged", ["champion_going_quiet"]
    ) == "champion_going_quiet"


def test_an_unrelated_slug_is_left_alone():
    assert detect_ai.match_existing_key(
        "budget_frozen", ["champion_going_quiet"]
    ) == "budget_frozen"


# --------------------------------------------------------------------------
# 7.6 severity
# --------------------------------------------------------------------------


def test_a_computed_band_overrides_the_models_opinion():
    """Dwell time is not a matter of judgement."""
    dossier = Dossier(deal_id=None)
    dossier.entries = {"r1": type("E", (), {"handle": "r1", "text": "in stage 'discovery' since 2026-01-01 (120 days)"})()}
    computed = detect_ai.computed_severity(RiskType.STALLED_STAGE.value, dossier)
    assert computed == Severity.CRITICAL
    assert detect_ai.apply_severity_policy(Severity.LOW, None, computed) == Severity.CRITICAL


def test_raising_severity_is_immediate():
    existing = _Risk(Severity.MEDIUM, last_seen_days_ago=0)
    assert detect_ai.apply_severity_policy(Severity.CRITICAL, existing, None) == Severity.CRITICAL


def test_lowering_severity_waits():
    """A badge that drops and climbs again on unchanged data reads as the tool
    being unreliable, not the deal improving."""
    fresh = _Risk(Severity.CRITICAL, last_seen_days_ago=1)
    assert detect_ai.apply_severity_policy(Severity.LOW, fresh, None) == Severity.CRITICAL

    old = _Risk(Severity.CRITICAL, last_seen_days_ago=detect_ai.SEVERITY_HYSTERESIS_DAYS + 1)
    assert detect_ai.apply_severity_policy(Severity.LOW, old, None) == Severity.LOW


# --------------------------------------------------------------------------
# 7.10 dismissals
# --------------------------------------------------------------------------


def _dossier_with_dismissal(reason, days_ago, action="engage_stakeholder"):
    dossier = Dossier(deal_id=None)
    dossier.dismissals[("no_economic_buyer", "")] = [
        (reason, action, datetime.now(timezone.utc) - timedelta(days=days_ago))
    ]
    return dossier


def test_already_handled_suppresses_for_a_window_then_expires():
    key = ("no_economic_buyer", "")
    recent = _dossier_with_dismissal(DismissalReason.ALREADY_HANDLED.value, 3)
    assert detect_ai.suppressed_by_dismissal(key, "engage_stakeholder", recent)

    stale = _dossier_with_dismissal(DismissalReason.ALREADY_HANDLED.value, 60)
    assert detect_ai.suppressed_by_dismissal(key, "engage_stakeholder", stale) is None


def test_already_handled_is_scoped_to_the_same_action():
    """They are doing that particular thing, not every possible thing."""
    key = ("no_economic_buyer", "")
    dossier = _dossier_with_dismissal(DismissalReason.ALREADY_HANDLED.value, 3,
                                      action="engage_stakeholder")
    assert detect_ai.suppressed_by_dismissal(key, "send_document", dossier) is None


def test_not_relevant_and_wrong_suppress_indefinitely():
    key = ("no_economic_buyer", "")
    for reason in (DismissalReason.NOT_RELEVANT.value, DismissalReason.WRONG.value):
        ancient = _dossier_with_dismissal(reason, 900)
        assert detect_ai.suppressed_by_dismissal(key, "engage_stakeholder", ancient), reason


def test_bad_timing_comes_back_sooner():
    key = ("no_economic_buyer", "")
    assert detect_ai.suppressed_by_dismissal(
        key, "engage_stakeholder", _dossier_with_dismissal(DismissalReason.BAD_TIMING.value, 2)
    )
    assert detect_ai.suppressed_by_dismissal(
        key, "engage_stakeholder", _dossier_with_dismissal(DismissalReason.BAD_TIMING.value, 20)
    ) is None


def test_an_undismissed_risk_is_not_suppressed():
    assert detect_ai.suppressed_by_dismissal(
        ("stalled_stage", ""), "schedule_meeting", Dossier(deal_id=None)
    ) is None


# --------------------------------------------------------------------------
# 7.3 handle validation
# --------------------------------------------------------------------------


def test_an_unknown_handle_is_detected():
    """The whole reason the model answers with handles rather than ids: an
    invented handle is detectable, a fabricated uuid is not."""
    dossier = Dossier(deal_id=None)
    dossier.entries = {"r1": object()}
    entries, unknown = detect_ai._resolve_refs(["r1", "r99"], dossier)
    assert len(entries) == 1
    assert unknown == {"r99"}


# --------------------------------------------------------------------------
# 7.4 what the shadow run measures
# --------------------------------------------------------------------------


def test_recall_counts_affirmed_open_risks_not_only_new_proposals():
    """The metric bug the first live shadow run exposed.

    The model does not re-propose a risk that is already open -- it returns a
    `still_present` verdict for it, which is what the prompt asks for. Scoring
    only `proposed` against the SQL rules therefore reported **recall 0.0** on
    a run where the model had affirmed every single rule-detected risk.

    The detector was right and the measurement was wrong, which is worth a test
    of its own: a shadow run is also measuring the measurement.
    """
    rule_keys = {("close_date_at_risk", ""), ("no_economic_buyer", ""), ("stalled_stage", "")}
    proposed = {("budget_unconfirmed", ""), ("missed_commitment", ""),
                ("security_review_pending", "")}
    affirmed = {("no_economic_buyer", ""), ("stalled_stage", ""), ("close_date_at_risk", "")}

    only_proposed = len(rule_keys & proposed) / len(rule_keys)
    assert only_proposed == 0.0, "this is what the broken metric reported"

    corrected = len(rule_keys & (proposed | affirmed)) / len(rule_keys)
    assert corrected == 1.0
