from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class Settings(BaseSettings):
    """Application settings, overridable via environment or backend/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DealPilot API"
    api_prefix: str = "/api"
    debug: bool = True

    # Origins allowed to call the API from a browser. The Vite dev server
    # proxies /api, so this mainly matters for direct cross-origin calls.
    cors_origins: Annotated[List[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Database ---
    database_url: str = "postgresql+asyncpg://dealpilot:change-me@localhost:5433/dealpilot"

    # --- Embeddings ---
    # document_chunks.embedding is declared as vector(embedding_dim), so this
    # value is baked into the column type at migration time. Changing models
    # later means a new column and a full re-embed -- it is not a config-only
    # switch. See docs/schema/README.md section 7.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Object storage ---
    # Uploaded documents live in MinIO. A document row exists only once its
    # bytes are stored, so this is not optional infrastructure.
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "dealpilot"
    minio_secret_key: str = "change-me"
    minio_bucket: str = "dealpilot-documents"
    minio_secure: bool = False
    # The host a presigned preview URL is signed for. Presigned URLs embed the
    # host they were signed against, and the browser is what opens them -- so
    # signing with the compose service name produces links that resolve only
    # inside the network and 404 for the user.
    minio_public_endpoint: str = "localhost:9000"
    # Supplied rather than discovered. The client otherwise makes a live
    # `?location=` request to resolve the bucket's region before signing --
    # which, for the public endpoint, means the backend trying to reach
    # localhost:9000 inside its own container. Presigning should be pure
    # computation; naming the region keeps it that way.
    minio_region: str = "us-east-1"

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
    groq_api_key: Optional[str] = None

    # Named `groq_model_*` rather than `model_*` because Pydantic reserves the
    # `model_` prefix and warns on every field that uses it.
    #
    # Roles, not sizes -- see docs/ai/README.md section 7. The challenger is
    # for the eval harness only: it is Preview tier and several times the price
    # of the primary, so it must never appear in the pipeline.
    groq_model_primary: str = "openai/gpt-oss-120b"
    groq_model_cheap: str = "openai/gpt-oss-20b"
    groq_model_challenger: str = "qwen/qwen3.8-27b"

    # `json_schema` is Groq's Structured Output API: constrained decoding, so
    # schema adherence is guaranteed at the token level. Configurable because
    # older langchain-groq releases -- which is what pip resolves on Python 3.9
    # -- predate it and accept only `json_mode` or `function_calling`.
    structured_output_method: str = "json_schema"

    ai_request_timeout_seconds: float = 120.0
    # One retry, then fail. A stage that cannot succeed twice should surface as
    # `failed` with its error, not consume the rate limit retrying.
    ai_max_retries: int = 1

    # Per-stage output ceilings. Deliberately not one number: an extraction
    # window returns many facts, a validator returns a verdict and a sentence.
    ai_max_tokens_extract: int = 8192
    ai_max_tokens_validate: int = 1024
    ai_max_tokens_detect: int = 8192
    ai_max_tokens_synthesize: int = 4096
    ai_max_tokens_chat: int = 8192
    ai_max_tokens_default: int = 4096

    # --- Worker ---
    # How long to idle when no poll query found work. Low enough that a queued
    # meeting starts promptly, high enough that an idle worker is not a
    # busy-loop against Postgres.
    worker_poll_seconds: float = 2.0

    # --- Rate-limit governor ---
    # Groq's Developer plan allows 250K TPM / 1K RPM per model. The parallel
    # extraction fan-out is what reaches that first, so the ceiling is enforced
    # client-side rather than discovered as a 429 storm.
    groq_max_concurrency: int = 4
    groq_tokens_per_minute: int = 250_000
    groq_requests_per_minute: int = 1000
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


settings = Settings()
