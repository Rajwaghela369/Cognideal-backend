"""Generate a pre-meeting brief from prefetched, bounded deal context."""

from app.ai.prompts import Prompt, register


PROMPT = register(
    Prompt(
        name="brief",
        version="brief@1",
        system=(
            "You prepare a concise briefing for a salesperson before a meeting. "
            "Use only the supplied deal context. Do not invent people, events, "
            "commitments, or risks. Prefer specific, actionable preparation over "
            "generic sales advice. If a section has no supported content, return "
            "an empty list."
        ),
        user=(
            "Upcoming meeting:\n{meeting}\n\n"
            "Deal:\n{deal}\n\n"
            "Open risks:\n{risks}\n\n"
            "Pending commitments:\n{commitments}\n\n"
            "Prior meeting summaries:\n{summaries}\n\n"
            "Stakeholder map:\n{stakeholders}\n\n"
            "Create the pre-meeting brief."
        ),
    )
)
