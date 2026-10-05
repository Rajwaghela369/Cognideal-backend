"""Eval metrics. Pure functions over plain dicts -- no database, no API call.

That constraint is the point. A metric that needs a live model to run cannot be
tested, and a scoring bug then looks exactly like a model regression. Everything
here consumes a *recorded* run: whatever the extractor, validator or detector
returned, saved as JSON. The same functions score a live run and a replay.

The five metrics are the ones docs/ai/README.md section 10 asks for.

The matching rule deserves its own note, because it is the decision inside this
module most likely to be got wrong: a predicted fact matches a labelled one on
**(fact_type, span overlap)**, never on text equality. The model's ``content``
phrasing legitimately varies between runs; the span it cites should not. Scoring
on strings would mark correct extractions as misses and make the number useless
for hill-climbing.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Fact extraction: precision and recall against hand labels
# ---------------------------------------------------------------------------


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """Characters shared by two spans, or 0 if they are in different chunks."""
    if a.get("chunk_index") != b.get("chunk_index"):
        return 0
    lo = max(a.get("char_start", 0), b.get("char_start", 0))
    hi = min(a.get("char_end", 0), b.get("char_end", 0))
    return max(0, hi - lo)


def match_facts(
    predicted: Sequence[Dict[str, Any]],
    labelled: Sequence[Dict[str, Any]],
    min_overlap: int = 1,
) -> Dict[str, Any]:
    """Pair predictions to labels on (fact_type, span overlap).

    Greedy by overlap size, largest first, one-to-one: a prediction cannot
    satisfy two labels and a label cannot be satisfied twice. Greedy rather
    than optimal because the spans are short and rarely contested, and an
    assignment algorithm here would be precision the inputs do not have.
    """
    candidates: List[Tuple[int, int, int]] = []
    for i, p in enumerate(predicted):
        for j, l in enumerate(labelled):
            if p.get("fact_type") != l.get("fact_type"):
                continue
            shared = _overlap(p, l)
            if shared >= min_overlap:
                candidates.append((shared, i, j))
    candidates.sort(reverse=True)

    used_p, used_l, pairs = set(), set(), []
    for shared, i, j in candidates:
        if i in used_p or j in used_l:
            continue
        used_p.add(i)
        used_l.add(j)
        pairs.append({"predicted": i, "labelled": j, "overlap": shared,
                      "label_id": labelled[j].get("id")})

    return {
        "matched": pairs,
        "false_positives": [i for i in range(len(predicted)) if i not in used_p],
        "missed": [labelled[j].get("id", j) for j in range(len(labelled)) if j not in used_l],
    }


def extraction_scores(
    predicted: Sequence[Dict[str, Any]], labelled: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Precision, recall and F1 over one transcript."""
    m = match_facts(predicted, labelled)
    tp = len(m["matched"])
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(labelled) if labelled else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "predicted": len(predicted), "labelled": len(labelled), "matched": tp,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "false_positives": len(m["false_positives"]), "missed": m["missed"],
        "by_fact_type": _recall_by_type(m, labelled),
    }


def _recall_by_type(m: Dict[str, Any], labelled: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Recall per fact_type -- where a single number hides which kind it misses."""
    found = {p["label_id"] for p in m["matched"]}
    per: Dict[str, List[int]] = {}
    for l in labelled:
        hit, total = per.setdefault(l["fact_type"], [0, 0])
        per[l["fact_type"]] = [hit + (1 if l.get("id") in found else 0), total + 1]
    return {k: "%d/%d" % (v[0], v[1]) for k, v in sorted(per.items())}


# ---------------------------------------------------------------------------
# Gate 0: do the citations resolve?
# ---------------------------------------------------------------------------


def gate0_pass_rate(links: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Share of claim->evidence links that verified.

    Counts every ``verification_status`` separately rather than collapsing to
    pass/fail: ``span_missing`` is a fabricated or paraphrased citation,
    ``value_drifted`` is a record that moved under a true claim. Same number,
    different bug.
    """
    counts = Counter(link.get("verification_status", "unverified") for link in links)
    total = sum(counts.values())
    verified = counts.get("verified", 0)
    return {
        "links": total,
        "verified": verified,
        "pass_rate": round(verified / total, 4) if total else 0.0,
        "by_status": dict(sorted(counts.items())),
    }


# ---------------------------------------------------------------------------
# Gate 1: what the validator said, and whether it is just agreeing
# ---------------------------------------------------------------------------


def verdict_distribution(validations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per verdict.

    Watched for a degenerate shape: a validator returning ``supported`` for
    everything has the same distribution as a perfect extractor and costs the
    same to run, so near-100% ``supported`` is a reason to distrust the gate
    rather than to celebrate the extractor.
    """
    counts = Counter(v.get("verdict", "unknown") for v in validations)
    total = sum(counts.values())
    return {
        "validations": total,
        "by_verdict": dict(sorted(counts.items())),
        "share_supported": round(counts.get("supported", 0) / total, 4) if total else 0.0,
        "share_rejected": round(
            (counts.get("contradicted", 0) + counts.get("unsupported", 0)) / total, 4
        ) if total else 0.0,
    }


CONFIDENCE_BANDS = ((0.0, 0.5, "low"), (0.5, 0.8, "mid"), (0.8, 1.01, "high"))


def _band(confidence: Optional[float]) -> str:
    if confidence is None:
        return "none"
    for lo, hi, name in CONFIDENCE_BANDS:
        if lo <= confidence < hi:
            return name
    return "none"


def confidence_verdict_matrix(claims: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Self-reported confidence against the independent verdict.

    ``confidence`` is the generator's own guess and ``verdict`` is a separate
    check, so the interesting cell is **high confidence + contradicted**: the
    model was certain and wrong. docs/schema/README.md section 5 calls those the
    most valuable eval cases there are, and this is how they get counted.
    """
    matrix: Dict[str, Counter] = {}
    for claim in claims:
        row = matrix.setdefault(_band(claim.get("confidence")), Counter())
        row[claim.get("verdict", "unknown")] += 1

    high_and_wrong = sum(
        matrix.get("high", Counter()).get(v, 0) for v in ("contradicted", "unsupported")
    )
    return {
        "matrix": {band: dict(sorted(row.items())) for band, row in sorted(matrix.items())},
        "high_confidence_rejected": high_and_wrong,
    }


# ---------------------------------------------------------------------------
# Risk detection: does it say the same thing twice in a row?
# ---------------------------------------------------------------------------


def risk_key(risk: Dict[str, Any]) -> Tuple[str, str]:
    """A risk's identity: (risk_type, risk_key) -- never its prose.

    The same identity the partial unique index uses. Scoring on titles would
    report a stable detector as unstable the moment it rephrased a card.
    """
    return (risk.get("risk_type", ""), risk.get("risk_key", "") or "")


def risk_stability(run_a: Sequence[Dict[str, Any]], run_b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Jaccard overlap of open risk keys across two runs on an unchanged deal.

    The flicker metric, and the one to watch first: a risk panel that changes
    between two runs over identical data reads as broken regardless of how
    precise either run was.
    """
    a = {risk_key(r) for r in run_a}
    b = {risk_key(r) for r in run_b}
    union = a | b
    return {
        "run_a": len(a), "run_b": len(b),
        "intersection": len(a & b), "union": len(union),
        "jaccard": round(len(a & b) / len(union), 4) if union else 1.0,
        "only_in_a": sorted("%s/%s" % k for k in (a - b)),
        "only_in_b": sorted("%s/%s" % k for k in (b - a)),
    }


def risk_recall_against_rules(
    ai_risks: Sequence[Dict[str, Any]],
    rule_risks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Did the AI pass find what the deterministic detector found?

    Free ground truth: the four SQL rules cannot miss and cannot hallucinate,
    so any rule-detected risk the AI pass did not propose is a measured recall
    miss with no human labelling. This is what task 7.4's shadow run reports.
    """
    ai = {risk_key(r) for r in ai_risks}
    rules = {risk_key(r) for r in rule_risks}
    missed = rules - ai
    return {
        "rule_risks": len(rules), "ai_risks": len(ai),
        "found": len(rules & ai),
        "recall": round(len(rules & ai) / len(rules), 4) if rules else 1.0,
        "missed": sorted("%s/%s" % k for k in missed),
        "ai_only": sorted("%s/%s" % k for k in (ai - rules)),
    }


# ---------------------------------------------------------------------------
# Risk detection over N runs: task 11.9
# ---------------------------------------------------------------------------
#
# `risk_stability` above compares two runs, which is what task 7.4 needed and
# all it had (n=2, Jaccard 0.67). Two runs cannot distinguish "the detector
# wobbles on one risk" from "the detector is bimodal", and 7.4's own note says
# the figure is worth re-measuring. These two functions take N runs instead.
#
# They are kept separate from the pairwise version rather than replacing it:
# `risk_stability`'s `only_in_a` / `only_in_b` lists are what you read when a
# single pair disagrees, and collapsing them into an N-run summary would lose
# the detail that makes a flicker diagnosable.


def risk_stability_n(runs: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Risk-key stability across N runs on an unchanged deal.

    Reports three different things, because they fail differently:

    ``mean_pairwise_jaccard`` is comparable to 7.4's two-run number -- the same
    statistic, averaged over all pairs instead of the single pair that existed.

    ``unanimous_jaccard`` is the harsher one: the keys every run agreed on,
    over the keys any run produced. With n=2 it equals the pairwise figure; as
    n grows it falls faster, because one run inventing one key once is enough
    to drop a key out of the intersection.

    ``per_key`` is the diagnosis. A detector at Jaccard 0.8 because one key
    appears half the time is a different product problem from one at 0.8
    because ten keys each appear nine times out of ten, and only this column
    tells them apart.
    """
    if not runs:
        return {"runs": 0, "mean_pairwise_jaccard": 1.0, "unanimous_jaccard": 1.0,
                "stable_keys": [], "unstable_keys": [], "per_key": {}}

    sets = [{risk_key(r) for r in run} for run in runs]
    union = set().union(*sets)
    intersection = set.intersection(*sets) if sets else set()

    pairs = [
        (len(a & b) / len(a | b)) if (a | b) else 1.0
        for i, a in enumerate(sets) for b in sets[i + 1:]
    ]

    counts = Counter(k for s in sets for k in s)
    n = len(sets)
    return {
        "runs": n,
        "mean_pairwise_jaccard": round(sum(pairs) / len(pairs), 4) if pairs else 1.0,
        "unanimous_jaccard": round(len(intersection) / len(union), 4) if union else 1.0,
        "keys_per_run": [len(s) for s in sets],
        "stable_keys": sorted("%s/%s" % k for k in intersection),
        "unstable_keys": sorted("%s/%s" % k for k in (union - intersection)),
        # appearances / runs, per key -- 1.0 means every run found it.
        "per_key": {
            "%s/%s" % k: round(counts[k] / n, 4) for k in sorted(union)
        },
    }


def verdict_flicker(verdicts_per_run: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """Did each open risk get the same verdict every time?

    The number task 11.9 says matters more than Jaccard, and for a reason worth
    stating: a *proposal* that comes and goes affects a card the human has not
    seen yet, but a *verdict* that flips moves a card the human is already
    looking at between "still a problem" and "resolved". The second is the one
    that reads as brokenness.

    A risk missing from a run is not scored as a flip -- it is scored as a
    missing verdict and counted separately. Treating absence as disagreement
    would conflate "the model changed its mind" with "the model said nothing",
    which have different fixes.
    """
    ids = sorted({rid for run in verdicts_per_run for rid in run})
    n = len(verdicts_per_run)
    per_risk, flickering, incomplete = {}, [], []
    for rid in ids:
        seen = [run[rid] for run in verdicts_per_run if rid in run]
        distinct = sorted(set(seen))
        per_risk[rid] = {
            "verdicts": dict(Counter(seen)),
            "answered_in": len(seen),
            "of_runs": n,
            "stable": len(distinct) == 1,
        }
        if len(distinct) > 1:
            flickering.append(rid)
        if len(seen) != n:
            incomplete.append(rid)
    return {
        "runs": n,
        "risks": len(ids),
        "flickering": sorted(flickering),
        "flicker_count": len(flickering),
        # The headline: zero is the pass condition task 11.9 states.
        "flicker_rate": round(len(flickering) / len(ids), 4) if ids else 0.0,
        "missing_verdicts": sorted(incomplete),
        "per_risk": per_risk,
    }
