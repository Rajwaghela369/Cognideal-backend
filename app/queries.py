"""Reusable SQL expressions shared by more than one route.

Lives at the app root rather than under db/, because it is business logic --
"what counts as stalled", "what is the next action" -- not infrastructure.
Keeping it in db/ also inverted the layering: models/ imports db.base and
db.mixins, so db/ importing back from models/ made the two packages mutually
dependent. Now every arrow points one way: db -> models -> queries -> services
-> routes.

Kept out of the route modules so the definition of "next action" or "how deals
are ranked by risk" exists exactly once -- two routes quietly disagreeing about
either is the kind of drift nobody notices until the numbers differ between
two screens.
"""

from sqlalchemy import Integer, and_, case, false, func, select

from app.models import Commitment, Deal, DealStageHistory, Risk, Task
from app.models.enums import DealStage

CLOSED_STAGES = ("closed_won", "closed_lost")

# high -> medium -> low -> unset. Used for ordering, so lower sorts first.
RISK_PRIORITY = case(
    (Deal.risk_level == "high", 0),
    (Deal.risk_level == "medium", 1),
    (Deal.risk_level == "low", 2),
    else_=3,
)

# `deals` deliberately has no next_action column -- it is derived from the
# oldest open task so the two can never drift. Soonest due date wins; tasks
# with no due date fall to the back.
NEXT_ACTION = (
    select(Task.title)
    .where(Task.deal_id == Deal.id, Task.status == "open")
    .order_by(Task.due_date.asc().nulls_last(), Task.created_at.asc())
    .limit(1)
    .correlate(Deal)
    .scalar_subquery()
    .label("next_action")
)

NEXT_ACTION_DUE_DATE = (
    select(Task.due_date)
    .where(Task.deal_id == Deal.id, Task.status == "open")
    .order_by(Task.due_date.asc().nulls_last(), Task.created_at.asc())
    .limit(1)
    .correlate(Deal)
    .scalar_subquery()
    .label("next_action_due_date")
)


# When this deal last moved stage. Falls back to created_at: a deal whose
# opening history row predates that route, or that a seeder wrote directly, has
# still been sitting in its stage since it existed.
STAGE_CHANGED_AT = func.coalesce(
    select(func.max(DealStageHistory.changed_at))
    .where(DealStageHistory.deal_id == Deal.id)
    .correlate(Deal)
    .scalar_subquery(),
    Deal.created_at,
)

# How long the deal has sat where it is.
#
# Distinct from last_activity_at, and the distinction is the whole point:
# `stale_days` asks "has anyone talked to them?", this asks "is any of that
# talking moving the deal?". A deal with weekly check-ins and no stage movement
# for two months looks perfectly healthy to the first question and is exactly
# what RiskType.STALLED_STAGE is for.
_DAYS_IN_STAGE = func.floor(
    func.extract("epoch", func.now() - STAGE_CHANGED_AT) / 86400
).cast(Integer)

DAYS_IN_STAGE = _DAYS_IN_STAGE.label("days_in_stage")


# How long a deal may sit in each stage before that is worth flagging.
#
# Per-stage rather than one number, because stages do not take the same time. A
# flat 45-day rule flags every security review -- legal is simply slow, and that
# is not a risk -- while staying silent on a discovery deal that died three
# weeks ago. False positives are the expensive kind of wrong here: once the risk
# panel cries wolf people stop reading it, and then it is worse than nothing.
#
# PLACEHOLDERS. These should be the 75th percentile of actual dwell time per
# stage among deals that went on to close, taken from real data once the seeder
# exists. Guessing them now is fine; leaving them guessed is not.
STALL_THRESHOLD_DAYS = {
    DealStage.QUALIFICATION: 14,
    DealStage.DISCOVERY: 21,
    DealStage.EVALUATION: 30,
    DealStage.SECURITY_AND_LEGAL: 60,
    DealStage.NEGOTIATION: 21,
}

# This deal's threshold, as a value rather than a branch per comparison. Putting
# the CASE here instead of around the whole predicate means STAGE_CHANGED_AT's
# correlated subquery is emitted once rather than once per stage.
#
# Closed stages are deliberately absent from the mapping, so `else_` leaves them
# NULL: a won or lost deal has stopped moving because it is finished, which is
# not the same as being stuck.
_STALL_THRESHOLD = case(
    *[(Deal.stage == stage, days) for stage, days in STALL_THRESHOLD_DAYS.items()],
    else_=None,
)

# coalesce, not a bare comparison: `>=` against the NULL threshold of a closed
# deal is NULL, and NULL is excluded by both `WHERE IS_STALLED` and
# `WHERE NOT IS_STALLED` -- so a closed deal would vanish from *both* sides of
# the filter. false() makes "not stalled" mean what it says.
IS_STALLED = func.coalesce(_DAYS_IN_STAGE >= _STALL_THRESHOLD, false())


# urgent -> high -> medium -> low. Used for ordering, so lower sorts first --
# same shape as RISK_PRIORITY, and for the same reason: a priority enum sorts
# alphabetically in SQL, which would put "high" above "urgent".
PRIORITY_ORDER = case(
    (Task.priority == "urgent", 0),
    (Task.priority == "high", 1),
    (Task.priority == "medium", 2),
    (Task.priority == "low", 3),
    else_=4,
)

# Past its due date and still open. Defined once here because the task table,
# the (future) dashboard panel and any risk detector must agree on what
# "overdue" means -- three places inventing the same predicate is three places
# to disagree.
#
# A task with no due date is never overdue: undated is not late, it is
# unscheduled.
IS_OVERDUE = and_(
    Task.status == "open",
    Task.due_date.is_not(None),
    Task.due_date < func.current_date(),
)


# critical -> high -> medium -> low. Ordering, so lower sorts first. A severity
# enum sorts alphabetically in SQL, which would put "critical" above "high" by
# luck and "low" above "medium" wrongly.
SEVERITY_ORDER = case(
    (Risk.severity == "critical", 0),
    (Risk.severity == "high", 1),
    (Risk.severity == "medium", 2),
    (Risk.severity == "low", 3),
    else_=4,
)

# Past due and still pending. An overdue commitment from the customer side is
# what MISSED_COMMITMENT will detect once extraction exists, so the definition
# lives here rather than being re-expressed by the detector later.
#
# Undated is not late, it is unscheduled -- same rule as tasks.
COMMITMENT_OVERDUE = and_(
    Commitment.status == "pending",
    Commitment.due_date.is_not(None),
    Commitment.due_date < func.current_date(),
)

# How close a deal may be to its expected close date, in a stage that is not
# yet negotiation, before that is worth flagging.
#
# PLACEHOLDER, like STALL_THRESHOLD_DAYS. Should come from how long deals
# actually take to move from each stage into negotiation, once the seeder
# exists.
CLOSE_DATE_WARNING_DAYS = 21
