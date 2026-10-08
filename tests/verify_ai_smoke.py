"""Task 0.3's live acceptance check: does `client.structured()` actually work?

Not in the pytest suite, because it needs a key and spends money (a fraction of
a cent). It is the cheapest possible answer to three questions that are
ambiguous when a later failure happens:

    1. is the key valid?
    2. is the model id right for this account?
    3. does OpenAI's strict json_schema structured output work here?

Run from the repo root with AI_ENABLED=true and OPENAI_API_KEY set:

    .venv/bin/python tests/verify_ai_smoke.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from pydantic import BaseModel, Field

from app.ai import client
from app.core.config import settings
from app.ai.prompts import get as get_prompt

TEXT = "Northwind Logistics signed for $180,000 after the security review."


class Extracted(BaseModel):
    """Every field required: strict mode permits no optional keys, so a
    "not found" answer is an empty string rather than an absent field."""

    company: str = Field(description="The company name, exactly as written")
    amount: str = Field(description="The dollar amount, exactly as written")


async def main() -> int:
    prompt = get_prompt("smoke")
    print("model   :", client.model_for(client.ROLE_PRIMARY))
    print("method  :", "json_schema", "| strict:", settings.structured_output_method in ("json_schema", "function_calling"))
    print("prompt  :", prompt.version)

    parsed, run = await client.structured(
        Extracted,
        prompt.messages(text=TEXT),
        task="extract",
        prompt_version=prompt.version,
        reasoning_effort="low",
    )

    print()
    print("parsed  :", parsed.model_dump())
    print("type    :", type(parsed).__name__)
    print("usage   : in=%s out=%s total=%s" % (run.input_tokens, run.output_tokens, run.total_tokens))
    print("latency : %d ms  attempts=%d  outcome=%s" % (run.latency_ms, run.attempts, run.outcome))

    ok = (
        isinstance(parsed, Extracted)
        and "Northwind" in parsed.company
        and "180,000" in parsed.amount
        and run.total_tokens
        and run.latency_ms > 0
    )
    print()
    print("PASS -- parsed object and populated usage record" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
