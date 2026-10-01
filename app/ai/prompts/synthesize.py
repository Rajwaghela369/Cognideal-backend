"""The meeting summary -- composed from the validated facts, not the transcript.

This is the design decision the prompt exists to enforce. A summary generated
from the raw transcript is a plausible blob that can contradict the facts
sitting next to it on the same screen, and it inherits no grounding: nothing
traces anywhere. A summary composed from the surviving fact set inherits it for
free -- every sentence rests on a fact that rests on a span that Gate 0
verified and Gate 1 checked.

It is also about a tenth of the input: a fact list is ~1K tokens where the
transcript is ~10K.

So the prompt is given facts and forbidden from adding to them. "Say less" is
the instruction that matters, because a model handed eight bullet points will
reach for narrative connective tissue that no fact supports.
"""

from app.ai.prompts import Prompt, register

PROMPT = register(
    Prompt(
        name="synthesize",
        version="synthesize@1",
        system=(
            "You write a short summary of a sales call from a list of facts "
            "that were extracted from it and verified against the "
            "transcript.\n"
            "\n"
            "Use only the facts given. Do not add context, do not infer what "
            "was probably meant, and do not smooth the list into a narrative "
            "by inventing the connections between items. If the facts do not "
            "say why something happened, the summary does not say why.\n"
            "\n"
            "Three to five sentences. Lead with what changed or what was "
            "decided; a reader who attended the call should learn nothing new "
            "and a reader who missed it should learn what matters. Name people "
            "as the facts name them.\n"
            "\n"
            "Numbers and dates: write them exactly as the facts write them. If "
            "a fact says 'a hundred and fifty thousand', so does the summary. "
            "Converting it to a numeral invents precision the call did not "
            "contain.\n"
            "\n"
            "If the fact list is empty or says nothing of consequence, say so "
            "in one sentence rather than padding."
        ),
        user=(
            "Call: {meeting_type} on {occurred_at}\n"
            "Present: {attendees}\n"
            "\n"
            "Verified facts:\n{facts}\n"
            "\n"
            "Write the summary."
        ),
    )
)
