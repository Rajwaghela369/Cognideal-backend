"""Rebuild the example recording from the current resolved labels.

The recording's spans are **chunk coordinates**, so changing how documents are
chunked invalidates them -- which is what happened when task 2.1 moved to
speaker-aware splitting. In production that cannot happen: chunks are immutable
and a re-ingest is delete-and-re-upload (docs/schema/README.md section 3). In
the fixtures it can, so this script exists.

It preserves the designed story rather than recording whatever a model said:
one exact hit, one hit with a shifted span, one right-span-wrong-type, one
invention. The metric tests assert exact counts against that story.

    python tests/eval/recorded/regenerate.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
LABELS = HERE.parents[1] / "fixtures" / "labels" / "01-discovery-2026-07-28.json"


def main() -> None:
    by_id = {f["id"]: f for f in json.loads(LABELS.read_text())["facts"]}
    for key in ("d1", "d3", "d6"):
        if "chunk_index" not in by_id[key]:
            raise SystemExit("labels are unresolved -- run tests/fixtures/resolve_labels.py")
    d1, d3, d6 = by_id["d1"], by_id["d3"], by_id["d6"]

    path = HERE / "extraction-discovery-example.json"
    rec = json.loads(path.read_text())
    rec["predicted_facts"][0].update(chunk_index=d1["chunk_index"],
                                     char_start=d1["char_start"], char_end=d1["char_end"])
    # Deliberately offset: a six-character drift must still match.
    rec["predicted_facts"][1].update(chunk_index=d3["chunk_index"],
                                     char_start=d3["char_start"] + 6, char_end=d3["char_end"] + 5)
    # d6's exact span, filed under the wrong fact_type.
    rec["predicted_facts"][2].update(chunk_index=d6["chunk_index"],
                                     char_start=d6["char_start"], char_end=d6["char_end"])
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print("regenerated", path.name)


if __name__ == "__main__":
    main()
