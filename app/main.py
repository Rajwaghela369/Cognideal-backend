from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1 import api_router
from app.core.config import settings
from app.db.session import SessionLocal


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    async def health(response: Response) -> dict:
        """Liveness plus database reachability, for orchestration.

        **Unversioned, and outside `/api`.** `docs/api/README.md` section 2 puts
        the wire surface under `v1` and leaves mechanics unversioned; a
        healthcheck is mechanics. A compose `healthcheck` or a load balancer
        pointed at `/api/v1/health` would break the day v2 lands, which is the
        opposite of what the probe is for.

        **Deliberately does not check the model provider.** `ai_enabled`, the token bucket and
        the queue depth belong to `GET /api/v1/system/ai-status`. If a provider
        outage made this endpoint fail, `restart: unless-stopped` would restart a
        perfectly healthy API in a loop -- and the REST layer genuinely does
        work without a model, which is the point of `_require_enabled`.

        Returns 503 rather than raising when the database is unreachable: a
        probe needs a status code, and a traceback through the exception
        handler is a 500 that says less.
        """
        try:
            async with SessionLocal() as db:
                await db.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 -- the probe reports, never raises
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "unhealthy",
                "database": "unreachable",
                "detail": "%s: %s" % (type(exc).__name__, exc),
            }
        return {"status": "ok", "database": "ok"}

    app.include_router(api_router, prefix=f"{settings.api_prefix}/v1")
    return app


app = create_app()
