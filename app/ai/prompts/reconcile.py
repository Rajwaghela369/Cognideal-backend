"""Was a promise kept, and does a new fact contradict an old one?

Two prompts, both comparisons rather than extractions, both answering with a
reference to something we supplied rather than free text.

**reconcile** -- a commitment was recorded weeks ago; do today's facts show it
satisfied? The answer is an index into a numbered list of open commitments, for
the same reason the risk dossier uses handles: a model asked to echo an id will
sometimes produce a plausible one belonging to nothing, and an out-of-range
integer is detectable where a fabricated uuid is not.

**supersede** -- a new fact and an old one of the same type. Does the new one
replace the old? This has to distinguish *contradiction* from *addition*, which
is the whole difficulty: "budget is a hundred and fifty thousand" followed by
"the CFO approved two hundred and twenty thousand" supersedes; followed by
"and procurement needs three bids" does not. Getting that wrong in the
permissive direction buries a fact that is still true.
"""

from app.ai.prompts import Prompt, register

RECONCILE = register(
    Prompt(
        name="reconcile",
        version="reconcile@1",
        system=(
            "You decide whether a promise recorded earlier on a deal has now "
            "been kept, based only on facts from the latest call.\n"
            "\n"
            "Answer with the number of the commitment the facts show was "
            "satisfied, or null.\n"
            "\n"
            "Say null unless a fact clearly shows the promised thing happened. "
            "Discussion of it is not delivery. Somebody saying they will do it "
            "again, or apologising for not having done it, is evidence it was "
            "NOT kept. A commitment is satisfied when the facts say the thing "
            "arrived, was sent, was completed or was signed off.\n"
            "\n"
            "Being wrong here closes a promise nobody kept, which is worse "
            "than leaving an open one open: the whole point of tracking them "
            "is to notice when they slip."
        ),
        user=(
            "Open commitments on this deal:\n{commitments}\n"
            "\n"
            "Facts from the latest call:\n{facts}\n"
            "\n"
            "Which commitment, if any, do these facts show was satisfied?"
        ),
    )
)

SUPERSEDE = register(
    Prompt(
        name="supersede",
        version="supersede@1",
        system=(
            "You decide whether a new fact about a deal replaces an older one.\n"
            "\n"
            "Answer with the number of the older fact the new one replaces, or "
            "null.\n"
            "\n"
            "A new fact replaces an older one when they are about the same "
            "thing and cannot both be current: a revised figure, a changed "
            "date, a resolved objection, a reversed decision.\n"
            "\n"
            "A new fact does NOT replace an older one when it merely adds "
            "detail, covers a different aspect, or is about a different "
            "subject -- even when both are the same kind of fact. Two budget "
            "facts about different things are both true.\n"
            "\n"
            "Say null when unsure. Marking a fact superseded hides it, and "
            "hiding something still true is worse than carrying one stale "
            "item that a human can see and judge."
        ),
        user=(
            "New fact ({fact_type}):\n{new_fact}\n"
            "\n"
            "Existing facts of the same kind on this deal:\n{existing}\n"
            "\n"
            "Which existing fact, if any, does the new one replace?"
        ),
    )
)
