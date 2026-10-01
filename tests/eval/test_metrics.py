"""The metrics, scored against a recorded run. No database, no API call.

Every expected number here is derived by hand from
``recorded/extraction-discovery-example.json`` and asserted exactly. That is
what makes a scoring bug distinguishable from a model regression: if these fail,
the metric is wrong, because the input never changes.
"""

import json
import pathlib

from tests.eval import metrics

RECORDING = json.loads(
    (pathlib.Path(__file__).parent / "recorded" / "extraction-discovery-example.json").read_text()
)
LABELS = json.loads(
    (pathlib.Path(__file__).parents[1] / "fixtures" / "labels" / "01-discovery-2026-07-28.json").read_text()
)


def test_labels_carry_resolved_offsets():
    """resolve_labels.py must have run; unresolved labels score nothing."""
    for fact in LABELS["facts"]:
        assert "chunk_index" in fact, "%s has no offsets -- run resolve_labels.py" % fact["id"]
        assert fact["char_end"] > fact["char_start"]
        assert "unciteable" not in fact, "%s: %s" % (fact["id"], fact.get("unciteable"))


def test_exact_span_matches():
    scores = metrics.extraction_scores(RECORDING["predicted_facts"], LABELS["facts"])
    assert scores["predicted"] == 4
    assert scores["labelled"] == 11
    # d1 exactly, d3 on overlap. The other two predictions match nothing.
    assert scores["matched"] == 2
    assert scores["false_positives"] == 2


def test_overlap_matching_tolerates_a_shifted_span():
    """The second prediction is offset by six characters and must still match.

    The model's span will rarely be byte-identical to a hand label. Requiring
    exact offsets would report a correct extraction as both a miss and a false
    positive -- double-penalising the one thing that was right.
    """
    predicted = [RECORDING["predicted_facts"][1]]
    labelled = [f for f in LABELS["facts"] if f["id"] == "d3"]
    m = metrics.match_facts(predicted, labelled)
    assert len(m["matched"]) == 1
    assert m["matched"][0]["label_id"] == "d3"
    assert m["matched"][0]["overlap"] > 0


def test_right_span_wrong_type_is_not_a_match():
    """Prediction 3 cites d6's exact span but calls it a requirement.

    fact_type is part of the identity. A competitor mention filed as a
    requirement is wrong in a way that matters: it reaches a different
    downstream rule.
    """
    predicted = [RECORDING["predicted_facts"][2]]
    labelled = [f for f in LABELS["facts"] if f["id"] == "d6"]
    m = metrics.match_facts(predicted, labelled)
    assert m["matched"] == []
    assert m["missed"] == ["d6"]


def test_recall_is_broken_out_by_fact_type():
    scores = metrics.extraction_scores(RECORDING["predicted_facts"], LABELS["facts"])
    by_type = scores["by_fact_type"]
    assert by_type["requirement"] == "1/2"
    assert by_type["budget"] == "1/1"
    assert by_type["competitor"] == "0/1"
    assert by_type["commitment"] == "0/1"


def test_matching_is_one_to_one():
    """Two predictions on one label may not both count."""
    label = [f for f in LABELS["facts"] if f["id"] == "d1"]
    twice = [dict(label[0]), dict(label[0])]
    m = metrics.match_facts(twice, label)
    assert len(m["matched"]) == 1
    assert len(m["false_positives"]) == 1


def test_gate0_pass_rate_separates_the_failure_kinds():
    g = metrics.gate0_pass_rate(RECORDING["claim_evidence_links"])
    assert g["links"] == 5
    assert g["verified"] == 2
    assert g["pass_rate"] == 0.4
    # Two fabrications and one record that moved are different bugs.
    assert g["by_status"] == {"span_missing": 2, "value_drifted": 1, "verified": 2}


def test_verdict_distribution():
    v = metrics.verdict_distribution(RECORDING["validations"])
    assert v["validations"] == 5
    assert v["by_verdict"] == {"contradicted": 1, "partial": 1, "supported": 2, "unsupported": 1}
    assert v["share_supported"] == 0.4
    assert v["share_rejected"] == 0.4


def test_confidence_verdict_matrix_surfaces_confident_errors():
    m = metrics.confidence_verdict_matrix(RECORDING["validations"])
    # 0.88 + contradicted: certain and wrong.
    assert m["high_confidence_rejected"] == 1
    # The high band is >= 0.8, so 0.91/0.88/0.84 all land here.
    assert m["matrix"]["high"] == {"contradicted": 1, "partial": 1, "supported": 1}
    assert m["matrix"]["mid"] == {"supported": 1, "unsupported": 1}


def test_risk_stability_ignores_prose_and_catches_flicker():
    s = metrics.risk_stability(RECORDING["risk_run_a"], RECORDING["risk_run_b"])
    # Two keys survive different wording; the invented slug does not.
    assert s["intersection"] == 2
    assert s["union"] == 4
    assert s["jaccard"] == 0.5
    assert s["only_in_a"] == ["other/champion_going_quiet"]
    assert s["only_in_b"] == ["other/champion_disengaged"]


def test_identical_runs_are_perfectly_stable():
    run = RECORDING["risk_run_a"]
    assert metrics.risk_stability(run, run)["jaccard"] == 1.0


def test_risk_recall_against_the_deterministic_rules():
    r = metrics.risk_recall_against_rules(RECORDING["risk_run_a"], RECORDING["rule_risks"])
    assert r["rule_risks"] == 3
    assert r["found"] == 2
    assert round(r["recall"], 4) == 0.6667
    # Free ground truth: SQL found it, the model did not.
    assert r["missed"] == ["single_threaded/"]
    assert r["ai_only"] == ["other/champion_going_quiet"]


def test_empty_inputs_do_not_divide_by_zero():
    assert metrics.extraction_scores([], [])["f1"] == 0.0
    assert metrics.gate0_pass_rate([])["pass_rate"] == 0.0
    assert metrics.verdict_distribution([])["validations"] == 0
    assert metrics.risk_stability([], [])["jaccard"] == 1.0
    assert metrics.risk_recall_against_rules([], [])["recall"] == 1.0
