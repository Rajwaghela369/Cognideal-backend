from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


def _rewrite_postgres_url(url: str, scheme: str, renames: dict, drop: tuple = ()) -> str:
    """Swap the scheme and translate query parameters between driver dialects."""
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key in drop:
            continue
        query.append((renames.get(key, key), value))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def asyncpg_url(url: str) -> str:
    """Any Postgres URL -> the form SQLAlchemy's asyncpg dialect accepts.

    Providers hand out libpq URLs (``postgresql://...?sslmode=require&
    channel_binding=require`` is Neon's). SQLAlchemy passes every query
    parameter to ``asyncpg.connect()`` as a keyword, and asyncpg has no
    ``sslmode`` or ``channel_binding`` keyword -- it raises on both. asyncpg's
    own name for the SSL mode is ``ssl``, with the same values; it does not
    support channel binding, so that one is dropped rather than renamed.
    """
    return _rewrite_postgres_url(
        url, "postgresql+asyncpg", {"sslmode": "ssl"}, drop=("channel_binding",)
    )


def psycopg_url(url: str) -> str:
    """The same database as a libpq URL, for psycopg (the chat checkpointer).

    The inverse of :func:`asyncpg_url`: libpq rejects both the ``+asyncpg``
    dialect suffix and asyncpg's ``ssl`` parameter.
    """
    return _rewrite_postgres_url(url, "postgresql", {"ssl": "sslmode"})


class Settings(BaseSettings):
    """Application settings, overridable via environment or backend/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CogniDeal API"
    api_prefix: str = "/api"
    debug: bool = True

    # Origins allowed to call the API from a browser. The Vite dev server
    # proxies /api, so this mainly matters for direct cross-origin calls.
    cors_origins: Annotated[List[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Database ---
    # Accepts the URL exactly as the provider gives it (Neon's libpq form
    # included); `_normalize_database_url` turns it into the asyncpg form the
    # app uses, and `psycopg_database_url` derives the checkpointer's.
    database_url: str = "postgresql+asyncpg://dealpilot:change-me@localhost:5433/dealpilot"

    # --- Embeddings ---
    # document_chunks.embedding is declared as vector(embedding_dim), so this
    # value is baked into the column type at migration time. Changing models
    # later means a new column and a full re-embed -- it is not a config-only
    # switch. See docs/schema/README.md section 7.
    #
    # Neither value drives any behaviour today, and the model name is worse
    # than unused.
    # Nothing in this codebase computes an embedding, so every
    # `document_chunks.embedding` is NULL and the HNSW index over it is empty;
    # the only retrieval here is lexical (`content.ilike(...)` in
    # `app/ai/tools`) plus trigram similarity for supersession candidates.
    # `text-embedding-3-small` is an OpenAI model, and OpenAI is now the
    # provider, so a backfill is reachable -- but nothing has been written to
    # compute one.
    # Nothing reads `embedding_model`; `embedding_dim` is read only by the
    # column declaration in `app/models/document.py`.
    #
    # They stay anyway: the column type depends on `embedding_dim`, and a
    # dimension with no model name beside it would be a number nobody can
    # explain. Treat both as a record of the choice, not as live config.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Object storage ---
    # Uploaded documents live in S3-compatible object storage (Neon's, in
    # production). A document row exists only once its bytes are stored, so
    # this is not optional infrastructure.
    #
    # The endpoint may be a bare host (`host[:port]`, with `s3_secure` choosing
    # the scheme) or a full URL such as the provider's
    # `https://<id>.storage...neon.tech`, in which case the scheme wins.
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = "dealpilot"
    s3_secret_key: str = "change-me"
    s3_bucket: str = "dealpilot-documents"
    s3_secure: bool = False
    # The host a presigned preview URL is signed for. Presigned URLs embed the
    # host they were signed against, and the browser is what opens them -- so
    # this must be reachable from the browser. Unset means "same as
    # s3_endpoint", which is right whenever the backend and the browser reach
    # storage by the same public host.
    s3_public_endpoint: Optional[str] = None
    # Supplied rather than discovered. The client otherwise makes a live
    # `?location=` request to resolve the bucket's region before signing.
    # Presigning should be pure computation; naming the region keeps it so.
    s3_region: str = "us-east-1"

    # --- Ingest ---
    # Chunks are immutable and citations point into them, so what matters is
    # that these stay stable, not that they are optimal. Nothing retrieves by
    # similarity yet, so there is no retrieval quality to tune against.
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 150
    max_upload_bytes: int = 25 * 1024 * 1024

    # --- AI layer ---
    # Off by default so the app boots with no key present: enabling AI must
    # never be a prerequisite for the 48 endpoints that do not use it.
    ai_enabled: bool = False
    openai_api_key: Optional[str] = None

    # Named `ai_model_*` rather than `model_*` because Pydantic reserves the
    # `model_` prefix and warns on every field that uses it.
    #
    # Roles, not sizes -- see docs/ai/README.md section 7. Primary does the
    # extraction, validation and detection that claims rest on; cheap does the
    # small classifications (titles, name tiebreaks). The challenger is for the
    # eval harness only and never appears in the pipeline.
    ai_model_primary: str = "gpt-4o"
    ai_model_cheap: str = "gpt-4o-mini"
    ai_model_challenger: str = "gpt-4o"

    # `json_schema` is OpenAI's Structured Outputs: with `strict` it is
    # constrained decoding, so schema adherence is guaranteed at the token
    # level. Configurable as an escape hatch (`function_calling`) should a
    # model without Structured Outputs ever be configured.
    structured_output_method: str = "json_schema"

    ai_request_timeout_seconds: float = 120.0
    # One retry, then fail. A stage that cannot succeed twice should surface as
    # `failed` with its error, not consume the rate limit retrying.
    ai_max_retries: int = 1

    # Per-stage output ceilings. Deliberately not one number: an extraction
    # window returns many facts, a validator returns a verdict and a sentence.
    #
    # **Each must leave room for its own input inside the TPM ceiling below.**
    # These were first sized against the published 250K TPM figure and left
    # unchanged when that was corrected to 8,000 -- at which point a cap of
    # 8192 was 102% of the entire per-minute budget, so a single request could
    # never fit in a fresh minute. Observed output: extraction 3.1-4.3K,
    # detection ~3K, a verdict ~200.
    ai_max_tokens_extract: int = 6144
    ai_max_tokens_validate: int = 1024
    ai_max_tokens_detect: int = 6144
    ai_max_tokens_synthesize: int = 2048
    ai_max_tokens_chat: int = 4096
    ai_max_tokens_default: int = 2048

    # --- Extraction ---
    # Chunks per extraction window. Small on purpose: instruction-following on
    # open models degrades with input length long before the context window
    # runs out, and Groq is fast enough that more small calls beat fewer large
    # ones in wall-clock. Windows do not overlap -- the chunks inside them
    # already do, so coverage is continuous without extracting the same
    # exchange twice.
    extract_window_chunks: int = 4

    # --- Roster and name resolution ---
    # Who we are. There is no users table (no auth in the MVP), so nothing else
    # can tell the roster parser that this speaker is us and the rest are the
    # customer. Inferring it from context -- whoever makes commitments -- would
    # misfile the first customer who promises something.
    ae_display_name: str = "Maya Chen"

    # Auto-link a transcript speaker to a contact only at or above this
    # trigram similarity, and only when the runner-up is at least
    # roster_link_margin behind. A wrong contact_id silently corrupts every
    # attendance-based risk, and an unresolved attendee is itself the
    # missing-stakeholder signal -- so NULL is the safe failure.
    roster_link_threshold: float = 0.75
    roster_link_margin: float = 0.15
    # Below the link threshold but above this, the name is a *candidate* and
    # goes to the model tiebreak. Below it, no model is called at all: a score
    # this low means nobody we know, which is an answer rather than a doubt.
    roster_candidate_threshold: float = 0.40

    # --- Worker ---
    # How long to idle when no poll query found work. Low enough that a queued
    # meeting starts promptly, high enough that an idle worker is not a
    # busy-loop against Postgres.
    worker_poll_seconds: float = 2.0
    analysis_debounce_seconds: int = 60
    analysis_max_debounce_seconds: int = 600
    analysis_sweep_hours: int = 24
    gone_quiet_days: int = 21
    analysis_tier2_suppressed: bool = False
    # How long a chat turn may sit in `status='streaming'` before the worker
    # calls it abandoned (task 11.6). Must exceed the longest plausible single
    # turn: the row is created before inference starts, so too low a value reaps
    # a stream that is still being written. A ReAct turn with several tool calls
    # and a throttled bucket can legitimately run minutes.
    chat_stream_timeout_seconds: int = 900

    # How many trailing messages the chat agent sees. The LangGraph checkpointer
    # stores the whole thread; this caps only what is sent to the model, via the
    # `pre_model_hook` in `ai/checkpointer.py` -- so raising it later makes
    # history the agent already has visible again, rather than needing a
    # backfill.
    #
    # Counted in messages, not turns, and tool traffic counts: one question that
    # takes three tool calls is roughly eight messages. 14 is about two such
    # exchanges. The real spend ceiling is `ai_max_tokens_chat` plus RunBudget;
    # this exists to stop unbounded growth in a long session.
    chat_history_messages: int = 14
    # Separate from the SQLAlchemy pool: the checkpointer speaks psycopg, not
    # asyncpg, so it cannot share one. Small on purpose -- it serves chat turns
    # only, and each turn holds a connection briefly.
    chat_checkpoint_pool_size: int = 4

    # --- Rate-limit governor ---
    # TIER-DEPENDENT, and getting it wrong defeats the governor entirely. OpenAI
    # sets limits per model and per usage tier (platform.openai.com ->
    # Settings -> Limits); the defaults below are gpt-4o at Tier 1, the
    # tightest of the two models configured. Set too high, the bucket never
    # throttles and the first symptom is a 429 storm part-way through a
    # pipeline run -- precisely what the governor exists to prevent.
    #
    # The authoritative number is on every response as `x-ratelimit-limit-tokens`.
    # Reading it from there would remove this setting, and is the obvious
    # improvement.
    ai_max_concurrency: int = 4
    ai_tokens_per_minute: int = 30_000
    ai_requests_per_minute: int = 500
    # A single pipeline run may not spend more than this, whatever it thinks it
    # needs. Bounds the cost of a prompt bug to one run.
    ai_token_ceiling_per_run: int = 200_000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        # Allow CORS_ORIGINS="http://a.com,http://b.com" in the environment.
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        return asyncpg_url(value)

    @property
    def psycopg_database_url(self) -> str:
        return psycopg_url(self.database_url)


settings = Settings()
