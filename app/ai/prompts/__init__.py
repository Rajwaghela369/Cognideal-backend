"""The prompt registry.

Every prompt is a module in this package exporting a single ``PROMPT``, and
every ``PROMPT`` carries a ``version``. That version is not bookkeeping: it is
written to ``claim_validations.validator_version`` and (after migration 0011)
``risks.detector_version``, columns that exist because after a prompt change
every historical row came from a different judge. Without the version the
coverage metric silently averages two populations -- and the rows already
written can never be told apart retroactively.

So the rule is: **a prompt change without a version bump is a bug.** Bump on
any edit that could change an output, including whitespace inside an example.

Prompts are kept out of the task modules so that the prompt and the code that
calls it can be reviewed, diffed and versioned separately -- a prompt edit
should be legible as a prompt edit.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Prompt:
    """One prompt, at one version.

    ``system`` and ``user`` are templates formatted with ``str.format``.
    Deliberately not Jinja or LangChain's ``ChatPromptTemplate``: the inputs
    here are assembled in Python (the dossier, the evidence spans, the fact
    set), so a template engine would add a second place where context is built.
    """

    name: str
    version: str
    system: str
    user: str

    def messages(self, **values: object) -> List[Tuple[str, str]]:
        """Render to the ``[(role, content)]`` shape the client takes.

        A missing key raises ``KeyError`` here rather than reaching the model
        as the literal ``{placeholder}``, which is the failure mode that
        produces a confidently wrong answer from a malformed prompt.
        """
        return [
            ("system", self.system.format(**values)),
            ("human", self.user.format(**values)),
        ]


_REGISTRY: Dict[str, Prompt] = {}


def register(prompt: Prompt) -> Prompt:
    """Add a prompt to the registry, rejecting duplicates and blank versions."""
    if not prompt.version:
        raise ValueError("prompt %r has no version" % prompt.name)
    existing = _REGISTRY.get(prompt.name)
    if existing is not None and existing is not prompt:
        raise ValueError(
            "prompt %r registered twice (%s and %s)"
            % (prompt.name, existing.version, prompt.version)
        )
    _REGISTRY[prompt.name] = prompt
    return prompt


def get(name: str) -> Prompt:
    """Look a prompt up by name.

    Raises rather than returning ``None``: a task that proceeds without its
    prompt would call the model with an empty system message.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            "no prompt named %r; registered: %s" % (name, sorted(_REGISTRY))
        ) from None


def all_prompts() -> Sequence[Prompt]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


# Importing a prompt module is what registers it, so every module has to be
# imported here. Listed explicitly rather than auto-discovered by walking the
# package: an import that only happens when something else happens to import it
# is a prompt that is missing from `all_prompts()` for reasons nobody can see.
from app.ai.prompts import detect as _detect  # noqa: E402,F401
from app.ai.prompts import extract as _extract  # noqa: E402,F401
# Task 11.10's candidate, under its own name so both versions score in one run.
from app.ai.prompts import extract_v3 as _extract_v3  # noqa: E402,F401
from app.ai.prompts import reconcile as _reconcile  # noqa: E402,F401
from app.ai.prompts import roster_tiebreak as _roster_tiebreak  # noqa: E402,F401
from app.ai.prompts import sentiment as _sentiment  # noqa: E402,F401
from app.ai.prompts import smoke as _smoke  # noqa: E402,F401
from app.ai.prompts import synthesize as _synthesize  # noqa: E402,F401
from app.ai.prompts import validate as _validate  # noqa: E402,F401
from app.ai.prompts import brief as _brief  # noqa: E402,F401
from app.ai.prompts import chat_title as _chat_title  # noqa: E402,F401
