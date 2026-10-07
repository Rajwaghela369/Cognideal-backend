"""Score extraction on the fixture corpus, for one model or all three.

    # score recorded runs: no API calls, no key
    python tests/eval/run_extraction_eval.py --replay tests/eval/recorded

    # task 11.11's three-way comparison, on the current prompt
    python tests/eval/run_extraction_eval.py --live --model all

    # task 11.10: one prompt version against another, same model
    python tests/eval/run_extraction_eval.py --live --prompt extract_v3

``--model all`` is the challenger comparison task 1.5 (reopened as 11.11) asks
for. The Qwen model belongs **here and nowhere else**: it is Preview tier and
several times the primary's price (docs/ai/README.md section 7), so a second
opinion on the eval set is exactly the right amount of exposure to it.

The live path does not improvise. It calls ``app.ai.extract`` -- the extractor
the pipeline uses -- with an overridden model role and prompt, and reads the
transcripts' chunks out of the database rather than re-chunking them, because
every label's offsets point into those exact rows. An eval scored against a
prompt or a chunking nothing else uses measures nothing.

It needs the fixture corpus ingested (``tests/fixtures/load.py``) and postgres
up; it writes recordings into ``recorded/`` so a comparison can be re-scored
later with no key.
"""

import argparse
import asyncio
import collections
import json
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from sqlalchemy import select

from tests.eval import metrics

HERE = pathlib.Path(__file__).parent
LABELS_DIR = HERE.parent / "fixtures" / "labels"

ROLES = ("primary", "cheap", "challenger")

#: The transcripts that have hand labels, by filename stem. A transcript with
#: no labels cannot be scored, so an extra ingested document is skipped rather
#: than counted as all-false-positives.
LABEL_STEMS = frozenset(p.stem for p in LABELS_DIR.glob("*.json"))


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
    """Score whatever is on disk. No model, no key, no network.

    Two recording shapes, because two exist and only one used to be read:

    ``by_transcript``  {stem: [facts]} -- a whole-corpus run, which is what
                       ``--live`` writes and what the kept baseline is.
    ``predicted_facts`` [facts] + ``transcript`` -- one transcript.

    The earlier version of this function required ``predicted_facts`` at the
    top level, so ``extract-baseline-gpt-oss-120b.json`` -- the recording whose
    whole purpose is to be the thing a prompt change is compared against -- was
    skipped in silence, and the only file scored was the three-fact metric
    fixture, pooled against all three transcripts' labels for a corpus recall
    of 0.08. A replay that silently scores the wrong file is worse than one
    that refuses, so unreadable recordings now raise instead of being dropped.
    """
    by_model, skipped = {}, []
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        recording = json.loads(path.read_text())
        # The metric fixture is a designed story (one exact hit, one shifted
        # span, one wrong type, one invention), not something a model returned.
        # Scoring it as a model run reports a prompt regression that never
        # happened. See recorded/regenerate.py.
        if recording.get("_not_a_model_run"):
            skipped.append(path.name)
            continue
        model = recording.get("model", path.stem)
        bucket = by_model.setdefault(model, {})
        if "by_transcript" in recording:
            for stem, facts in recording["by_transcript"].items():
                bucket.setdefault(pathlib.Path(stem).stem, []).extend(facts)
        elif "predicted_facts" in recording:
            stem = pathlib.Path(recording.get("transcript", path.stem)).stem
            bucket.setdefault(stem, []).extend(recording["predicted_facts"])
        else:
            raise SystemExit(
                "%s has neither `by_transcript` nor `predicted_facts` -- it is "
                "not a scoreable recording. Mark it `_not_a_model_run: true` if "
                "that is deliberate." % path.name
            )
    if skipped:
        print("not model runs, skipped: %s" % ", ".join(skipped))
    labels = load_labels()
    return {model: score(preds, labels) for model, preds in by_model.items()}


#: What the extractor actually reads off a chunk -- it takes ``Sequence[Any]``
#: and never queries through one. Detached copies are used instead of the ORM
#: rows because a repeated run has to survive between iterations: the first
#: ``--repeat 3`` lost runs 2 and 3 to `MissingGreenlet`, since the rollback
#: between runs expires every ORM attribute and the next iteration's first
#: ``chunk.content`` then tried to lazy-load outside a greenlet. Plain objects
#: cannot expire, so there is no session state to get wrong.
DetachedChunk = collections.namedtuple(
    "DetachedChunk", "id chunk_index content chunk_metadata"
)


async def load_corpus_chunks(db):
    """The fixture transcripts' chunks, keyed by label stem.

    Read from the database rather than re-chunked here, for the reason
    ``stages.load_chunks`` gives: the chunks *are* the text, and every label in
    ``fixtures/labels/`` carries offsets into these exact rows. A second
    chunker in the eval harness would score the model against coordinates the
    running system never produces.

    Returned detached, as plain values -- see :data:`DetachedChunk`.
    """
    from app.models import Document, Meeting
    from app.models.document import DocumentChunk

    out = {}
    meetings = (
        await db.execute(
            select(Meeting)
            .join(Document, Document.id == Meeting.transcript_document_id)
            .order_by(Meeting.scheduled_at)
        )
    ).scalars()
    for meeting in meetings:
        # `occurred_at` is the document's, not the meeting's -- the meeting
        # column is `scheduled_at`. Read from the same place `stages.py:218`
        # reads it, so the prompt sees the date the pipeline would send.
        filename, occurred_at = (
            await db.execute(
                select(Document.original_filename, Document.occurred_at)
                .where(Document.id == meeting.transcript_document_id)
            )
        ).one()
        stem = pathlib.Path(filename or "").stem
        if stem not in LABEL_STEMS:
            continue
        chunks = [
            DetachedChunk(c.id, c.chunk_index, c.content, c.chunk_metadata)
            for c in (await db.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == meeting.transcript_document_id)
                .order_by(DocumentChunk.chunk_index)
            )).scalars()
        ]
        out[stem] = (
            meeting.meeting_type,
            chunks,
            str(occurred_at)[:10] if occurred_at else None,
        )
    return out


async def run_one(db, role, prompt, corpus):
    """Extract the whole corpus once, under one model role and one prompt."""
    from app.ai import extract

    by_transcript, stats = {}, {"reported": 0, "located": 0, "unlocatable": 0,
                                "payload_errors": 0, "failed_transcripts": {}}
    for stem, (meeting_type, chunks, occurred_at) in sorted(corpus.items()):
        try:
            candidates = await extract.extract_facts(
                chunks,
                meeting_type=meeting_type,
                occurred_at=occurred_at or "an unknown date",
                role=role,
                prompt=prompt,
            )
        except Exception as exc:
            # Recorded, not raised. A model that cannot return parseable JSON
            # for one transcript still has a score on the other two, and
            # "failed here" is itself a comparison result -- which is the
            # whole point of task 11.11.
            stats["failed_transcripts"][stem] = "%s: %s" % (type(exc).__name__, exc)
            by_transcript[stem] = []
            print("    %-34s FAILED  %s" % (stem, str(exc)[:80]))
            continue

        located = [c for c in candidates if c.located]
        stats["reported"] += len(candidates)
        stats["located"] += len(located)
        stats["unlocatable"] += sum(1 for c in candidates if c.rejected == extract.UNLOCATABLE)
        stats["payload_errors"] += sum(1 for c in candidates if c.payload_error)
        by_transcript[stem] = [
            {
                "fact_type": c.fact_type,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "content": c.content,
                "confidence": c.confidence,
            }
            for c in located
        ]
        print("    %-34s reported=%-3d located=%d" % (stem, len(candidates), len(located)))
    return by_transcript, stats


def aggregate(runs_scored, labels):
    """Summarise N scored runs of one arm.

    Exists because a single run cannot decide a prompt here. Task 11.10's first
    comparison put `extract@3` at f1 0.56 against the *recorded* `@2` at 0.54
    and called it an improvement; a same-day `@2` run then scored 0.58, and its
    objection recall came in at 1/3 where the recording said 0/3. The variance
    between two runs of the same prompt was larger than the difference between
    the two prompts, so one sample per arm measures the sampling, not the
    change.

    Per-label hit *rates* are the useful output: "found in 3 of 3 runs" and
    "found in 1 of 3" are different claims about the same label, and a single
    run flattens both to "found".
    """
    totals = [r["TOTAL"] for r in runs_scored]
    n = len(totals)

    def spread(key):
        values = [t[key] for t in totals]
        return {
            "mean": round(sum(values) / n, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    all_label_ids = sorted(
        f["id"] for labels_one in labels.values() for f in labels_one["facts"]
    )
    # A label is "missed" in a run if it appears in that run's missed list.
    hit_rate = {}
    for label_id in all_label_ids:
        hits = sum(1 for t in totals if label_id not in t["missed"])
        hit_rate[label_id] = {"found_in": hits, "of": n, "rate": round(hits / n, 4)}

    by_type = {}
    for labels_one in labels.values():
        for fact in labels_one["facts"]:
            by_type.setdefault(fact["fact_type"], []).append(fact["id"])
    per_type = {
        fact_type: {
            "mean_found": round(
                sum(hit_rate[i]["found_in"] for i in ids) / n, 2
            ),
            "of": len(ids),
            "per_label": {i: hit_rate[i]["rate"] for i in sorted(ids)},
        }
        for fact_type, ids in sorted(by_type.items())
    }
    return {
        "runs": n,
        "precision": spread("precision"),
        "recall": spread("recall"),
        "f1": spread("f1"),
        "per_type": per_type,
        "hit_rate": hit_rate,
        "never_found": sorted(i for i, h in hit_rate.items() if h["found_in"] == 0),
        "always_found": sorted(i for i, h in hit_rate.items() if h["found_in"] == n),
    }


def report_aggregate(by_arm):
    for arm, agg in by_arm.items():
        print("\n=== %s  (n=%d) ===" % (arm, agg["runs"]))
        for key in ("precision", "recall", "f1"):
            s = agg[key]
            print("  %-10s mean %.3f   min %.3f   max %.3f"
                  % (key, s["mean"], s["min"], s["max"]))
        print("  recall by fact_type (mean found / labelled):")
        for fact_type, row in agg["per_type"].items():
            flickers = [i for i, r in row["per_label"].items() if 0.0 < r < 1.0]
            print("    %-18s %4.2f/%d%s" % (
                fact_type, row["mean_found"], row["of"],
                "   unstable: %s" % ", ".join(flickers) if flickers else ""))
        print("  never found in any run: %s" % (agg["never_found"] or "none"))


async def live(models, prompt_name, out_dir, repeat=1):
    from app.ai import extract  # noqa: F401  -- fail loudly if it is missing
    from app.ai.prompts import get as get_prompt
    from app.core.config import settings
    from app.db.session import SessionLocal

    if not settings.ai_enabled or not settings.openai_api_key:
        raise SystemExit("ai_enabled is off or OPENAI_API_KEY is unset -- use --replay")

    prompt = get_prompt(prompt_name)
    labels = load_labels()
    by_arm, by_model_last = {}, {}

    async with SessionLocal() as db:
        corpus = await load_corpus_chunks(db)
        missing = sorted(set(labels) - set(corpus))
        if missing:
            raise SystemExit(
                "no ingested transcript for %s -- load the fixtures first:\n"
                "  docker compose up -d postgres minio\n"
                "  uvicorn app.main:app --port 8099 &\n"
                "  python tests/fixtures/load.py" % ", ".join(missing)
            )

        for role in models:
            model_name = _model_for(role, settings)
            label = "%s/%s" % (model_name, prompt.version)
            print("\n%s (%s), prompt %s, %d run(s)"
                  % (role, model_name, prompt.version, repeat))
            scored_runs, raw_runs = [], []
            for attempt in range(1, repeat + 1):
                if repeat > 1:
                    print("  run %d/%d" % (attempt, repeat))
                by_transcript, stats = await run_one(db, role, prompt, corpus)
                scored = score(by_transcript, labels)
                scored_runs.append(scored)
                raw_runs.append({"by_transcript": by_transcript, "totals": stats})
                print("    recall %.2f  f1 %.2f  objection %s" % (
                    scored["TOTAL"]["recall"], scored["TOTAL"]["f1"],
                    scored["TOTAL"]["by_fact_type"].get("objection", "-")))

            by_arm[label] = aggregate(scored_runs, labels)
            by_model_last[label] = scored_runs[-1]
            out = pathlib.Path(out_dir) / ("extract-%s-%s.json" % (
                prompt.version.replace("@", "-"), role))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "_comment": "Live run by run_extraction_eval.py --live.",
                "model": label, "role": role, "prompt_version": prompt.version,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "repeat": repeat,
                "runs": raw_runs,
                # The last run also at the top level, so --replay (which reads
                # one `by_transcript`) can still score this file.
                "by_transcript": raw_runs[-1]["by_transcript"],
                "totals": raw_runs[-1]["totals"],
            }, indent=2) + "\n")
            print("    -> %s" % out)

    if repeat > 1:
        report_aggregate(by_arm)
    else:
        report(by_model_last)
    return 0


def _model_for(role, settings):
    return {
        "primary": settings.ai_model_primary,
        "cheap": settings.ai_model_cheap,
        "challenger": settings.ai_model_challenger,
    }[role]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", metavar="DIR", help="score recorded runs instead of calling a model")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="primary", choices=ROLES + ("all",))
    parser.add_argument("--prompt", default="extract",
                        help="registered prompt name (default: extract)")
    parser.add_argument("--out", default=str(HERE / "recorded"),
                        help="where to save live recordings")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="run each arm N times and report the spread "
                             "(n=1 cannot separate a prompt change from "
                             "run-to-run variance -- see `aggregate`)")
    args = parser.parse_args()

    if args.replay:
        report(replay(args.replay))
        return 0
    if args.live:
        models = list(ROLES) if args.model == "all" else [args.model]
        return asyncio.run(live(models, args.prompt, args.out, args.repeat))
    parser.error("pass --replay DIR or --live")


if __name__ == "__main__":
    sys.exit(main())
