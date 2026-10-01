"""Score extraction on the fixture corpus, for one model or all three.

    # today: score recorded runs, no API calls, no key
    python tests/eval/run_extraction_eval.py --replay tests/eval/recorded

    # once Phase 3.1/3.2 exist: call Groq and score what comes back
    python tests/eval/run_extraction_eval.py --live --model all

``--model all`` is the challenger comparison task 1.5 asks for. The Qwen model
belongs **here and nowhere else**: it is Preview tier and several times the
primary's price (docs/ai/README.md section 7), so a second opinion on the eval
set is exactly the right amount of exposure to it.

The live path deliberately refuses to improvise. It imports the real extractor,
and if Phase 3 has not built it yet it says so rather than inventing a
throwaway prompt -- an eval scored against a prompt nothing else uses measures
nothing.
"""

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, ".")

from tests.eval import metrics

HERE = pathlib.Path(__file__).parent
LABELS_DIR = HERE.parent / "fixtures" / "labels"

ROLES = ("primary", "cheap", "challenger")


def load_labels():
    out = {}
    for path in sorted(LABELS_DIR.glob("*.json")):
        labels = json.loads(path.read_text())
        unresolved = [f["id"] for f in labels["facts"] if "chunk_index" not in f]
        if unresolved:
            raise SystemExit(
                "%s has unresolved labels %s -- run tests/fixtures/resolve_labels.py"
                % (path.name, unresolved)
            )
        out[path.stem] = labels
    return out


def score(predictions_by_transcript, labels_by_transcript):
    """Per-transcript scores plus a corpus total."""
    rows, pooled_pred, pooled_labels = {}, [], []
    for stem, labels in labels_by_transcript.items():
        predicted = predictions_by_transcript.get(stem, [])
        rows[stem] = metrics.extraction_scores(predicted, labels["facts"])
        pooled_pred.extend(predicted)
        pooled_labels.extend(labels["facts"])
    # Pooled rather than averaged: a per-transcript mean would weight a
    # three-fact transcript the same as an eleven-fact one.
    rows["TOTAL"] = metrics.extraction_scores(pooled_pred, pooled_labels)
    return rows


def report(by_model):
    width = max(len(m) for m in by_model) if by_model else 10
    print("\n%-*s  %-34s %5s %5s %5s %4s %4s" % (width, "model", "transcript",
                                                 "prec", "rec", "f1", "fp", "miss"))
    print("-" * (width + 66))
    for model, rows in by_model.items():
        for stem, s in rows.items():
            print("%-*s  %-34s %5.2f %5.2f %5.2f %4d %4d"
                  % (width, model if stem == "TOTAL" else "", stem,
                     s["precision"], s["recall"], s["f1"],
                     s["false_positives"], len(s["missed"])))
    print()
    for model, rows in by_model.items():
        total = rows["TOTAL"]
        print("%-*s  recall by fact_type: %s" % (width, model, total["by_fact_type"]))
        if total["missed"]:
            print("%-*s  missed: %s" % (width, "", ", ".join(map(str, total["missed"]))))


def replay(directory):
    """Score whatever is on disk. No model, no key, no network."""
    by_model = {}
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        recording = json.loads(path.read_text())
        if "predicted_facts" not in recording:
            continue
        model = recording.get("model", path.stem)
        stem = pathlib.Path(recording.get("transcript", path.stem)).stem
        by_model.setdefault(model, {}).setdefault(stem, []).extend(recording["predicted_facts"])
    return {model: score(preds, load_labels()) for model, preds in by_model.items()}


async def live(models):
    try:
        from app.ai import extract  # noqa: F401
    except ImportError:
        raise SystemExit(
            "no extractor yet: app/ai/extract.py arrives with task 3.2, and the "
            "payload schemas with 3.1. Use --replay until then; this runner will "
            "not invent a prompt to score against."
        )
    raise SystemExit("live scoring is wired in task 3.6, against the real extractor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", metavar="DIR", help="score recorded runs instead of calling a model")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="primary", choices=ROLES + ("all",))
    args = parser.parse_args()

    if args.replay:
        report(replay(args.replay))
        return 0
    if args.live:
        models = list(ROLES) if args.model == "all" else [args.model]
        return asyncio.get_event_loop().run_until_complete(live(models))
    parser.error("pass --replay DIR or --live")


if __name__ == "__main__":
    sys.exit(main())
