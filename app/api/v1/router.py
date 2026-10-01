from fastapi import APIRouter

from app.api.v1.routes import deals, documents, tasks

api_router = APIRouter()

api_router.include_router(deals.router)
api_router.include_router(tasks.router)
# Top-level: reached from a preview link or a Layer C citation, neither of
# which holds a deal id.
api_router.include_router(documents.router)
