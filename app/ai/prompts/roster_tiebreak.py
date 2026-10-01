"""Which known contact is this transcript speaker, if any.

The narrowest model task in the project, and deliberately so. Trigram
similarity has already decided that *something* plausible exists; what it
cannot settle is which of two close candidates a partial name refers to --
"Priya" when the account has a Priya Raman and a Priya Shah.

Two rules the prompt has to carry, because both failure modes are silent:

*   **Answering "none" is a real answer**, not a failure to try. An unresolved
    attendee is the missing-stakeholder signal; a wrong link corrupts every
    attendance-based risk and nothing in the UI would reveal it.
*   **Choose by index, never by id.** The candidate list is numbered and the
    answer is a number, so the model cannot name a contact it was not given.
    Same reason the risk dossier uses handles.
"""

from app.ai.prompts import Prompt, register

PROMPT = register(
    Prompt(
        name="roster_tiebreak",
        version="roster_tiebreak@1",
        system=(
            "You match a speaker label from a sales call transcript to a known "
            "contact on the account.\n"
            "\n"
            "Answer with the number of the candidate, or null.\n"
            "\n"
            "Choose null unless you are confident. These are the only reasons to "
            "pick a candidate: the label is the same person's full name, a "
            "shortened or familiar form of it, or the same name with an obvious "
            "spelling or transcription slip. A shared first name alone is not "
            "enough when two candidates share it.\n"
            "\n"
            "Answering null is correct and useful whenever the speaker is "
            "somebody not in the list. Do not pick the closest candidate as a "
            "guess -- a wrong match is worse than no match, because it silently "
            "misattributes who attended the meeting."
        ),
        user=(
            "Speaker label from the transcript: {raw_name}\n"
            "\n"
            "Known contacts on this account:\n"
            "{candidates}\n"
            "\n"
            "Which numbered contact is this speaker? Answer null if none of them."
        ),
    )
)
