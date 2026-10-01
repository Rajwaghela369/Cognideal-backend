"""Everything reached through /deals.

One module per *resource*, not per route: /deals/{id}/stakeholders is a
different table with its own CRUD, and shares the prefix only because it is
reached through a deal. Grouping them in a package keeps the whole prefix in
one place while letting meetings.py, documents.py, tasks.py and the rest drop
in later without touching anything else.

This module does the assembling so app/api/router.py includes one router
rather than growing a line per sub-resource.
"""

from fastapi import APIRouter

from app.api.v1.routes.deals import (
    attendees,
    commitments,
    core,
    documents,
    facts,
    meetings,
    risks,
    stage_history,
    stakeholders,
)

router = APIRouter()

router.include_router(core.router)
router.include_router(stakeholders.router)
router.include_router(stage_history.router)
router.include_router(meetings.router)
router.include_router(attendees.router)
router.include_router(attendees.participants)
router.include_router(documents.router)
router.include_router(risks.router)
router.include_router(commitments.router)
router.include_router(facts.router)
