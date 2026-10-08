"""The connectivity check, and the only prompt in Phase 0.

Not a product prompt. It exists so that task 0.3's acceptance test -- "a smoke
test returns a parsed object and a populated usage record" -- can run against
the real model endpoint without any pipeline existing, and so that the registry
has something to prove itself against.

Kept afterwards: it is the cheapest possible answer to "is the key valid, is
the model id right, and does constrained decoding actually work on this
account", which is worth having when a later failure is ambiguous.
"""

from app.ai.prompts import Prompt, register

PROMPT = register(
    Prompt(
        name="smoke",
        version="smoke@1",
        system=(
            "You extract structured data. Answer only from the text given. "
            "Do not infer, and do not add fields."
        ),
        user=(
            "Text:\n{text}\n\n"
            "Return the company name and the dollar amount exactly as written."
        ),
    )
)
