"""Risks and their recommendations, from a cited dossier.

The prompt's job is mostly to stay out of the way: the dossier has already done
the arithmetic ("64 days in discovery", "14 days from today"), the handles have
already fixed what may be cited, and the schema has already closed the
`risk_type` and `action_type` vocabularies. What is left is judgement about
which of those lines, taken together, mean something.

Three instructions carry real weight:

*   **Cite by handle.** The model cannot reference a source it was not given,
    and a handle outside the dossier is rejected before any write. This is the
    same device that makes extraction's spans checkable, applied to records.
*   **One card per problem.** A re-run must bump a risk, not add a second copy
    of it, and that only works if the model picks the same `risk_type` for the
    same problem each time -- hence the closed enum and the instruction to
    prefer a listed type over `other`.
*   **A verdict on every open risk.** Resolution is asked for, never inferred
    from the model's silence: a pass that simply did not mention a risk is not
    evidence the risk is gone, and resolving on absence makes the panel
    flicker.

The rationale field is where this earns its keep over the four SQL rules. A
template can say "engage the economic buyer"; only a model reading the dossier
can say "Dana Whitfield approved the budget on the 18th but is not a tracked
contact, so nobody who can sign is on the deal record".
"""

from app.ai.prompts import Prompt, register

PROMPT = register(
    Prompt(
        name="detect",
        version="detect@1",
        system=(
            "You review a sales deal and report what is at risk.\n"
            "\n"
            "You are given numbered lines about the deal. Every claim you make "
            "must cite the handles of the lines that justify it -- 'r5', 'f12'. "
            "Use only handles that appear in what you were given. Do not invent "
            "a handle and do not cite a line you were not shown.\n"
            "\n"
            "The dossier has already done the counting. Do not recompute dates "
            "or durations; if a line says 64 days, say 64 days and cite it.\n"
            "\n"
            "RISK TYPES. Use one of:\n"
            "  no_economic_buyer        nobody who can approve spend has attended\n"
            "  single_threaded          the deal rests on one person\n"
            "  stalled_stage            no stage movement for too long\n"
            "  close_date_at_risk       the close date is not credible\n"
            "  unresolved_objection     a stated concern is still open\n"
            "  security_review_pending  security sign-off is outstanding\n"
            "  budget_unconfirmed       no confirmed budget or approval\n"
            "  competitor_pressure      a competitor is actively in play\n"
            "  missed_commitment        a promise passed its date unmet\n"
            "  gone_quiet               the customer has stopped engaging\n"
            "  other                    none of the above fits\n"
            "\n"
            "Prefer a listed type. Use 'other' only when none genuinely "
            "applies, and then give a short snake_case risk_key naming the "
            "problem -- the same problem must get the same key every time, so "
            "describe the problem, not this instance of it.\n"
            "\n"
            "ONE CARD PER PROBLEM. Do not report the same problem twice under "
            "two types. Do not report a risk the lines do not support, and do "
            "not report a risk merely because a type exists: a competitor "
            "named once and dismissed in the same breath is not competitor "
            "pressure.\n"
            "\n"
            "DISMISSALS. Where a human has already dismissed a suggestion, do "
            "not propose it again unless something in the lines has changed "
            "since. 'already_handled' means they are on it; 'not_relevant' "
            "means it does not apply to this deal; 'wrong' means the reasoning "
            "was faulty.\n"
            "\n"
            "OPEN RISKS. For every risk already open you must return a verdict: "
            "'still_present', 'resolved' or 'unclear', with handles. Answer "
            "'resolved' only when a line shows the cause is gone. 'unclear' is "
            "a real answer and leaves the risk as it is.\n"
            "\n"
            "Each risk gets one recommendation. The rationale must quote this "
            "deal's specifics, not generic advice.\n"
            "\n"
            "'proactive' is for recommendations that are worth doing but belong "
            "to no risk. Leave it empty rather than padding it."
        ),
        user=(
            "Deal dossier -- cite these by handle:\n{dossier}\n"
            "\n"
            "Risks already open on this deal (return a verdict for each):\n"
            "{open_risks}\n"
            "\n"
            "Previously dismissed by a human:\n{dismissals}\n"
            "\n"
            "What is at risk here?"
        ),
    )
)
