"""Gate 1 -- does the cited evidence actually support the claim?

The prompt is short because the *context* is the mechanism. The validator is
shown the claim and the quoted spans and **nothing else**: no transcript, no
deal record, no extraction context. Given the source material it would
silently re-derive the claim and rubber-stamp it, and the gate would buy
nothing. A separate call with a deliberately starved context is the point
(docs/schema/README.md section 5).

Two things the wording works at:

*   **Four verdicts, not two.** ``partial`` and ``unsupported`` are the useful
    middle: a claim where one clause is evidenced and another invented is the
    commonest real failure, and collapsing it to "no" loses the information
    that most of it was fine.
*   **"Addresses" versus "supports".** ``unsupported`` means the span is real
    but about something else -- the citation was lazy. ``contradicted`` means
    the span says the opposite, which is a different and worse bug. Telling
    them apart is what makes the verdict distribution diagnostic rather than
    decorative.
"""

from app.ai.prompts import Prompt, register

PROMPT = register(
    Prompt(
        name="validate",
        version="validate@1",
        system=(
            "You check whether a quotation supports a claim.\n"
            "\n"
            "You are given a claim and the exact words quoted as evidence for "
            "it. You have nothing else, and you must not assume anything "
            "beyond the quotation. If the claim says something the quotation "
            "does not say, that is not support -- however plausible it sounds "
            "and however likely it is to be true.\n"
            "\n"
            "Break the claim into its separate assertions and check each "
            "against the quotation. Then answer with one verdict:\n"
            "\n"
            "  supported     every assertion in the claim is stated in the "
            "quotation.\n"
            "  partial       some assertions are stated, others are not. Use "
            "this when a claim adds a detail -- a figure, a date, a name, a "
            "condition -- that the quotation does not contain.\n"
            "  contradicted  the quotation states the opposite of the claim, "
            "or a fact incompatible with it.\n"
            "  unsupported   the quotation is about something else and does "
            "not address the claim either way.\n"
            "\n"
            "Pay particular attention to numbers, dates and names. A claim "
            "that names a figure or a date the quotation does not contain is "
            "`partial` at best, even when the rest of it is exact.\n"
            "\n"
            "Give one sentence of reasoning, naming the assertion that failed "
            "if one did. Do not restate the claim."
        ),
        user=(
            "Claim:\n{claim}\n"
            "\n"
            "Quoted evidence:\n{evidence}\n"
            "\n"
            "Which verdict, and why?"
        ),
    )
)
