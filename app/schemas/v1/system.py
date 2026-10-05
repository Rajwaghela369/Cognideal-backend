"""The operational read: configuration, queue depth, and this process's budget.

Deliberately not a metrics system. `client.py` already logs every model call as
`ai.run task=... model=... outcome=... in=... out=... ms=...` (task 0.7), which
is the per-run record; Langfuse and an `ai_runs` table were both declined
(docs/ai/TASKS.md, Out of scope). What those logs cannot answer is "what is the
state right now" -- how much token budget is left, how many deals are waiting,
how far behind the sweep is. That is this.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.common import ORM


class AIConfig(BaseModel):
    """What the layer is configured to do. Echoed because the single most
    expensive misconfiguration in this project's history was a rate limit set
    31x too high, which looked healthy from every angle until a run died."""

    model_config = ORM

    enabled: bool
    model_primary: str
    model_cheap: str
    tokens_per_minute: int
    requests_per_minute: int
    max_concurrency: int
    debounce_seconds: int
    max_debounce_seconds: int
    sweep_hours: int
    tier2_suppressed: bool


class AIBudget(BaseModel):
    """The leaky bucket's current level, **in this process only**.

    The governor is a module-level singleton per interpreter, and the worker is
    a separate container (`docker-compose.yml`), so the API's bucket is not the
    one doing the work. Reported anyway because it is the one that would throttle
    a synchronous route, and labelled rather than quietly presented as global --
    a shared bucket would need Redis, which is a second datastore this project
    has not earned.
    """

    model_config = ORM

    process: str
    tokens_available: float
    tokens_capacity: int
    #: Seconds until the bucket refills to capacity at the configured rate. The
    #: number that explains a 77-second pause mid-pipeline.
    seconds_to_full: float


class AIQueues(BaseModel):
    """What the worker's three poll queries would find, counted now.

    Cross-process and therefore the trustworthy half of this payload: these are
    rows in Postgres, not in-memory state.
    """

    model_config = ORM

    queued_meetings: int
    oldest_queued_meeting_at: Optional[datetime] = None
    dirty_deals: int
    oldest_dirty_at: Optional[datetime] = None
    #: Deals whose `analysis_swept_at` is null or older than the sweep window.
    #: A number that only ever grows means the sweep is not keeping up, since it
    #: claims one deal per pass by design.
    sweep_backlog: int


class AIStatus(BaseModel):
    model_config = ORM

    config: AIConfig
    budget: AIBudget
    queues: AIQueues
