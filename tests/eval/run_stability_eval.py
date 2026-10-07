"""Task 11.9: how stable is risk detection on an unchanged deal?

Runs ``detect_ai.shadow_run`` N times against one deal and scores risk-key
stability plus per-risk verdict flicker. **Writes nothing** -- ``shadow_run``
proposes and scores against the deterministic rules without touching the
database, which is the whole reason it exists (task 7.4).

    # the task's sample: ten runs on the fixture deal
    python tests/eval/run_stability_eval.py --deal "SecureFlow Enterprise" -n 10

    # re-score a saved run with no API calls and no key
    python tests/eval/run_stability_eval.py --replay recorded/stability/stability-<n>.json

Why ten and not two: 7.4 had n=2 and reported Jaccard 0.67, which is one
disagreement on one key and cannot distinguish an occasional wobble from a
detector that alternates between two answers. Ten runs make the per-key
appearance rate readable, which is the number that says *which* card flickers.

The runs are sequential, not gathered. Concurrency would have them contend for
the same governor bucket and turn a stability measurement into a throttling
measurement, and a throttled extraction window already waits ~77 seconds.
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, ".")

from sqlalchemy import select

from tests.eval import metrics

HERE = pathlib.Path(__file__).parent

#: Its own directory, not `recorded/`. These are detection runs, and
#: `run_extraction_eval.py --replay` globs `recorded/*.json` for *extraction*
#: recordings -- a detect run sitting there is neither skippable nor scoreable
#: by it, so it would make the extraction replay refuse outright.
STABILITY_DIR = HERE / "recorded" / "stability"


def _as_risk_dicts(keys):
    """``["stalled_stage/", "other/champion_quiet"]`` -> metric input dicts.

    ``shadow_run`` reports keys already joined for human reading; the metrics
    want the pair back. Split once from the left: ``risk_type`` never contains
    a slash and ``canonical_key`` slugifies, so one split is lossless.
    """
    out = []
    for key in keys:
        risk_type, _, risk_key = key.partition("/")
        out.append({"risk_type": risk_type, "risk_key": risk_key})
    return out


def score(runs):
    """Score N recorded shadow runs. Pure -- this is the replay path.

    Failed runs are reported, then excluded from the stability and flicker
    figures. Both choices matter. Reporting them is the point: a run that
    raised produced no panel at all, which is worse than any amount of key
    churn. Excluding them from Jaccard is what keeps the two numbers readable
    -- folding an empty panel in as "a run that agreed with nothing" would
    drive the key metrics toward zero and bury *which* keys actually wobble
    under the fact that one request 400'd.
    """
    ok = [r for r in runs if not r.get("error")]
    failures = [r for r in runs if r.get("error")]

    # The panel's contents: what the model proposed plus what it affirmed as
    # still present. The same union shadow_run scores recall on, and for the
    # same reason -- an affirmed open risk is a card on screen.
    panels = [
        _as_risk_dicts(list(r["proposed"]) + list(r["affirmed_open"]))
        for r in ok
    ]
    recalls = [r["recall"] for r in ok if r.get("recall") is not None]
    runs, all_runs = ok, runs
    return {
        "sample_size": len(all_runs),
        "scored_runs": len(ok),
        "failed_runs": {
            "count": len(failures),
            # The headline for this section: one in ten is a blank panel.
            "rate": round(len(failures) / len(all_runs), 4) if all_runs else 0.0,
            "errors": sorted({f["error"][:160] for f in failures}),
        },
        "stability": metrics.risk_stability_n(panels),
        "flicker": metrics.verdict_flicker([r["verdicts"] for r in runs]),
        "recall": {
            "runs_scored": len(recalls),
            "min": min(recalls) if recalls else None,
            "max": max(recalls) if recalls else None,
            "mean": round(sum(recalls) / len(recalls), 4) if recalls else None,
            # A rule-detected risk missed in *any* run is the interesting case:
            # mean recall 0.97 and "never missed anything" are different claims.
            "ever_missed": sorted({m for r in runs for m in r.get("missed", [])}),
        },
        "rejections": sorted({
            "%s: %s" % (t, why) for r in runs for t, why in r.get("rejected", [])
        }),
    }


async def live(deal_name, n):
    from app.ai import detect_ai
    from app.core.config import settings
    from app.db.session import SessionLocal
    from app.models import Deal

    if not settings.ai_enabled or not settings.openai_api_key:
        raise SystemExit(
            "ai_enabled is off or OPENAI_API_KEY is unset -- this task needs live "
            "calls. Use --replay to re-score a saved run."
        )

    runs = []
    async with SessionLocal() as db:
        deal = await db.scalar(select(Deal).where(Deal.name == deal_name))
        if deal is None:
            names = (await db.scalars(select(Deal.name))).all()
            raise SystemExit("no deal named %r. Found: %s" % (deal_name, sorted(set(names))))
        # Plain values, read before the loop. The per-run rollback below
        # expires every ORM attribute, and `deal.id` on the next iteration
        # would then lazy-load -- which outside a greenlet raises
        # MissingGreenlet rather than simply refetching.
        deal_id, deal_label = deal.id, deal.name
        print("deal %s (%s), %d runs, model %s\n" % (
            deal_label, deal_id, n, settings.ai_model_primary))

        for i in range(1, n + 1):
            started = time.monotonic()
            try:
                result = await detect_ai.shadow_run(db, deal_id)
            except Exception as exc:
                # A run that raises is a *measurement*, not a reason to abandon
                # the sample -- and it is the most consequential kind of
                # instability there is. The first ten-run sample lost run 3 to
                # Groq `json_validate_failed` (the model emitted a stray
                # `"{""risk_type"` mid-array), which means the panel renders
                # nothing at all on that pass. Aborting here would have thrown
                # away the two good runs and, worse, reported the detector as
                # unmeasurable rather than as failing one pass in ten.
                result = {
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "proposed": [], "affirmed_open": [], "rejected": [],
                    "rule_risks": [], "recall": None, "missed": [],
                    "ai_only": [], "verdicts": {},
                }
            # shadow_run writes nothing, but `propose` builds a dossier through
            # this session. Roll back rather than trusting that: an identity
            # map holding a risk row across ten iterations is exactly how a
            # "stability" run would start measuring its own cache. Also clears
            # a failed transaction before the next iteration.
            await db.rollback()
            result["_elapsed_s"] = round(time.monotonic() - started, 1)
            runs.append(result)
            if result.get("error"):
                print("run %2d/%d  %4.1fs  FAILED  %s" % (
                    i, n, result["_elapsed_s"], result["error"][:110]))
            else:
                print("run %2d/%d  %4.1fs  panel=%d  verdicts=%d  recall=%s" % (
                    i, n, result["_elapsed_s"],
                    len(result["proposed"]) + len(result["affirmed_open"]),
                    len(result["verdicts"]), result["recall"],
                ))
    return runs


def report(scored):
    s, f, r = scored["stability"], scored["flicker"], scored["recall"]
    fail = scored["failed_runs"]
    print("\n--- runs ---")
    print("attempted %d, scored %d, failed %d (rate %s)" % (
        scored["sample_size"], scored["scored_runs"], fail["count"], fail["rate"]))
    for err in fail["errors"]:
        print("  %s" % err)
    print("\n--- stability, n=%d scored ---" % scored["scored_runs"])
    print("mean pairwise jaccard : %s" % s["mean_pairwise_jaccard"])
    print("unanimous jaccard     : %s" % s["unanimous_jaccard"])
    print("keys per run          : %s" % s["keys_per_run"])
    print("stable keys           : %s" % (s["stable_keys"] or "none"))
    print("unstable keys         : %s" % (s["unstable_keys"] or "none"))
    for key, rate in sorted(s["per_key"].items(), key=lambda kv: (-kv[1], kv[0])):
        print("  %-44s %s" % (key, rate))

    print("\n--- verdict flicker ---")
    print("risks with a verdict  : %d" % f["risks"])
    print("flickering            : %d  (rate %s)" % (f["flicker_count"], f["flicker_rate"]))
    if f["flickering"]:
        for rid in f["flickering"]:
            print("  %s  %s" % (rid, f["per_risk"][rid]["verdicts"]))
    if f["missing_verdicts"]:
        print("missing a verdict in some run: %s" % f["missing_verdicts"])

    print("\n--- recall vs the SQL rules ---")
    print("mean %s  (min %s, max %s)" % (r["mean"], r["min"], r["max"]))
    print("ever missed: %s" % (r["ever_missed"] or "nothing"))
    if scored["rejections"]:
        print("\nrejected proposals seen: %s" % scored["rejections"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deal", default="SecureFlow Enterprise", help="deal name")
    ap.add_argument("-n", type=int, default=10, help="number of runs")
    ap.add_argument("--replay", help="score a saved runs file instead of calling the model")
    ap.add_argument("--out", help="where to save the raw runs (default: recorded/stability-<n>.json)")
    args = ap.parse_args()

    if args.replay:
        payload = json.loads(pathlib.Path(args.replay).read_text())
        runs = payload["runs"] if isinstance(payload, dict) else payload
    else:
        runs = asyncio.run(live(args.deal, args.n))
        out = pathlib.Path(args.out) if args.out else STABILITY_DIR / ("stability-%d.json" % len(runs))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "_comment": "Task 11.9. Raw shadow runs, scored by run_stability_eval.score.",
            "deal": args.deal,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "runs": runs,
        }, indent=2, default=str))
        print("\nraw runs -> %s" % out)

    report(score(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
