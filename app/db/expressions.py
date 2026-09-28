"""Reusable SQL expressions shared by more than one route.

Kept out of the route modules so the definition of "next action" or "how deals
are ranked by risk" exists exactly once -- two routes quietly disagreeing about
either is the kind of drift nobody notices until the numbers differ between
two screens.
"""

from sqlalchemy import case, select

from app.models import Deal, Task

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
