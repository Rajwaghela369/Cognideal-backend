"""The replay reader, pinned. No model, no key, no database.

These exist because the reader silently scored the wrong file. It required
`predicted_facts` at the top level, so the kept baseline -- whose entire
purpose is to be the thing a prompt change is compared against -- was skipped,
and the only recording scored was the hand-designed metric fixture, pooled
against all three transcripts' labels for a corpus recall of 0.08. A replay
that reports a number from the wrong input is worse than one that crashes, so
the shapes are asserted rather than trusted.
"""

import json
import pathlib

import pytest

from tests.eval import run_extraction_eval as runner

RECORDED = pathlib.Path(runner.HERE) / "recorded"


def test_the_kept_baseline_is_readable_and_scores_as_recorded():
    """The regression: `by_transcript` recordings must not be skipped."""
    scored = runner.replay(RECORDED)
    assert "openai/gpt-oss-120b" in scored, (
        "the baseline recording was skipped -- replay cannot read its shape"
    )
    total = scored["openai/gpt-oss-120b"]["TOTAL"]
    # The numbers the recording's own comment claims for extract@2.
    assert round(total["recall"], 2) == 0.73
    assert round(total["f1"], 2) == 0.54


def test_baseline_per_type_recall_is_the_11_10_starting_point():
    """What task 11.10 actually has to move, as measured rather than recalled."""
    by_type = runner.replay(RECORDED)["openai/gpt-oss-120b"]["TOTAL"]["by_fact_type"]
    # The open regression.
    assert by_type["objection"] == "0/3"
    # NOT a regression: extract@2's precedence list already fixed this, and
    # TASKS.md 11.10 described extract@1's figure. Pinned so the claim cannot
    # drift back.
    assert by_type["commitment"] == "3/3"


def test_the_metric_fixture_is_not_scored_as_a_model_run():
    """It is a designed story, not something a model returned.

    One exact hit, one shifted span, one wrong type, one invention -- scoring
    that as a model run reports a prompt regression that never happened. The
    assertion is that its *facts* are absent from every arm, not that the
    directory holds some number of files: recordings accumulate as prompts and
    models are compared, so a count here would fail on every new run and say
    nothing about the contract.
    """
    fixture = json.loads((RECORDED / "extraction-discovery-example.json").read_text())
    assert fixture["_not_a_model_run"] is True

    designed = {f["content"] for f in fixture["predicted_facts"]}
    scored = runner.replay(RECORDED)
    assert scored, "no recordings scored at all"
    for arm, rows in scored.items():
        for stem, row in rows.items():
            # `matched`/`missed` carry label ids, so compare on what the
            # fixture would contribute: its invented content string.
            assert not (designed & _contents(RECORDED, arm, stem)), (
                "the designed fixture leaked into arm %s" % arm
            )


def _contents(directory, arm, stem):
    """Predicted `content` strings for one arm and transcript, from disk."""
    out = set()
    for path in sorted(directory.glob("*.json")):
        recording = json.loads(path.read_text())
        if recording.get("_not_a_model_run") or recording.get("model") != arm:
            continue
        for key, facts in (recording.get("by_transcript") or {}).items():
            if key == stem:
                out.update(f.get("content", "") for f in facts)
    return out


def test_a_single_transcript_recording_still_reads(tmp_path):
    """The older per-transcript shape keeps working."""
    labels = runner.load_labels()
    stem = sorted(labels)[0]
    fact = dict(labels[stem]["facts"][0])
    (tmp_path / "one.json").write_text(json.dumps({
        "model": "some-model", "transcript": "%s.txt" % stem,
        "predicted_facts": [fact],
    }))
    scored = runner.replay(tmp_path)
    assert scored["some-model"][stem]["matched"] == 1


def test_an_unscoreable_recording_refuses_rather_than_being_dropped(tmp_path):
    (tmp_path / "junk.json").write_text(json.dumps({"model": "m", "notes": "hi"}))
    with pytest.raises(SystemExit, match="neither"):
        runner.replay(tmp_path)
