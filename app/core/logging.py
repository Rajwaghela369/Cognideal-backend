"""Console logging for the API and the worker -- one format, one level.

Uvicorn configures only its own loggers, so without this every ``cognideal.*``
INFO line -- stage timings, chat turns, tool calls, ``ai.run`` accounting --
is dropped and only WARNING and above reach the console, via Python's
last-resort handler. Called once at process start by ``app.main`` and
``worker``.
"""

import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"

#: Third-party loggers that are noisy at INFO. httpx logs one line per HTTP
#: request -- every OpenAI call -- which `ai.run` already records with tokens
#: and latency.
_QUIET = ("httpx", "httpcore", "openai", "psycopg.pool", "langsmith")


def configure_logging() -> None:
    level_name = (settings.log_level or ("DEBUG" if settings.debug else "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    app_logger = logging.getLogger("cognideal")
    app_logger.setLevel(level)
    # Idempotent: uvicorn's --reload re-imports the app, and a second handler
    # would print every line twice.
    if not any(getattr(h, "_cognideal", False) for h in app_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._cognideal = True  # type: ignore[attr-defined]
        app_logger.addHandler(handler)
    # Ours prints them; the root (if anything configured it) must not as well.
    app_logger.propagate = False

    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
