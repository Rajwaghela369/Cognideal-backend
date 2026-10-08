from fastapi import APIRouter

from app.api.v1.routes import accounts, chat, deals, documents, system, tasks

api_router = APIRouter()

# Accounts come first because everything hangs off them: a deal needs an
# account_id and a stakeholder needs a contact_id, and until this router
# existed neither could be created through the API at all.
api_router.include_router(accounts.router)
api_router.include_router(deals.router)
api_router.include_router(tasks.router)
# Top-level: reached from a preview link or a Layer C citation, neither of
# which holds a deal id.
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(chat.usage_router)
# Not about any one deal: configuration, token budget and queue depth.
api_router.include_router(system.router)
