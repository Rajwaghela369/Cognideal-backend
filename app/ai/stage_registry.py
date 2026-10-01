"""The thirteen stages, declared: their order, and which of them are critical.

A declaration, not a runner. ``graph.py`` holds the wiring and ``stages.py``
the work; this is the executable copy of ``docs/ai/README.md`` section 4, and a
test asserts the graph's nodes match it.

Worth keeping separate from the graph for one reason above all: the
**critical/degradable split** is a product rule, not a graph detail.
``analysis_status='failed'`` means stages 0-4 failed -- those five produce the
facts, and without them the meeting was not analysed. Everything after them
enriches, so a failed summary must not discard facts that already landed. A
node consults this to know which it is.

(Formerly ``pipeline.py``, which also held a sequential runner. The runner was
the right thing until stage 2 had real work to fan out; it then became a second
implementation of the same thirteen stages, with its own copy of each body, and
was removed in favour of the graph.)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    #: True for stages 0-4. See the module docstring.
    critical: bool


STAGES: List[Stage] = [
    Stage(0, "parse_transcript", True),        # Phase 2.1
    Stage(1, "roster", True),                  # Phase 2.2-2.4
    Stage(2, "extract_window", True),          # Phase 3.2
    Stage(3, "gate0", True),                   # Phase 3.3
    Stage(4, "drop_unevidenced", True),        # Phase 3.4
    Stage(5, "gate1_entailment", False),       # Phase 4.1
    Stage(6, "quarantine", False),             # Phase 4.3
    Stage(7, "reconcile_commitments", False),  # Phase 6.1
    Stage(8, "supersede_facts", False),        # Phase 6.2
    Stage(9, "synthesize_summary", False),     # Phase 5.2
    Stage(10, "sentiment", False),             # Phase 5.3
    Stage(11, "finalize", False),              # Phase 5.4
    Stage(12, "redetect_risks", False),        # Phase 7
]

BY_NAME: Dict[str, Stage] = {stage.name: stage for stage in STAGES}

#: The degradable stages in order, chained after Gate 1 by the graph.
LATE_STAGES: List[str] = [stage.name for stage in STAGES if not stage.critical]


class CriticalStageFailed(RuntimeError):
    """A stage 0-4 failure.

    Raised out of the graph so the worker records ``failed`` and does not
    retry: a handled failure in one of these will not succeed on a second
    attempt, and leaving the row ``queued`` is a retry storm.
    """

    def __init__(self, stage: Stage, cause: BaseException) -> None:
        super().__init__("stage %d (%s) failed: %s" % (stage.index, stage.name, cause))
        self.stage = stage
        self.cause = cause
