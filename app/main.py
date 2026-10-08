import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1 import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import SessionLocal


configure_logging()
logger = logging.getLogger("cognideal.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown for the parts that are not request-scoped.

    The chat checkpointer's tables are created before the first chat turn, so
    no user's first message waits on DDL -- and a client that gives up on that
    message cannot cancel the migration half-way. Failure is logged, not
    raised: the REST layer works without AI, and ``checkpointer.saver()``
    retries setup on the next chat turn.

    With ``embedded_worker`` the analysis worker runs here as a background
    task, for hosts that cannot run it as its own service.
    """
    if settings.ai_enabled:
        from app.ai import checkpointer

        try:
            await checkpointer.saver()
        except Exception:  # noqa: BLE001 -- retried lazily by the first chat turn
            logger.exception("checkpointer.startup_setup_failed")

    worker_task = None
    if settings.embedded_worker:
        # Imported here: `worker` lives at the repo root, and an API that does
        # not embed it should not load the analysis graph at all.
        import worker

        worker_task = asyncio.create_task(
            worker.run_forever(embedded=True), name="embedded-worker"
        )
        logger.info("worker.embedded_started")

    yield

    if worker_task is not None:
        import worker

        worker.stop()
        try:
            # Let the current unit of work finish if it can; Render allows
            # about 30 seconds after SIGTERM. A run cut short is rolled back
            # and stays queued, so the next start picks it up.
            await asyncio.wait_for(worker_task, timeout=20)
        except asyncio.TimeoutError:
            worker_task.cancel()
            logger.warning("worker.embedded_cancelled -- in-flight run rolled back, stays queued")
        except Exception:  # noqa: BLE001 -- shutting down regardless
            logger.exception("worker.embedded_stop_failed")
    if settings.ai_enabled:
        from app.ai import checkpointer

        await checkpointer.close()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

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
