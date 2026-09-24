from typing import List

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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        # Allow CORS_ORIGINS="http://a.com,http://b.com" in the environment.
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
