"""Provider config pasted as-is must reach each driver in a form it accepts.

Neon hands out a libpq URL and an S3 endpoint URL. asyncpg, psycopg and the S3
client each want something slightly different, and each fails at connect time
-- in production, not here -- when handed the wrong shape.
"""

from app.core.config import Settings, asyncpg_url, psycopg_url
from app.services import storage

NEON = (
    "postgresql://neondb_owner:pw@ep-x-pooler.c-5.eu-central-1.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)


def test_neon_url_becomes_asyncpg_form():
    """asyncpg raises on `sslmode` and `channel_binding` keywords."""
    url = asyncpg_url(NEON)
    assert url.startswith("postgresql+asyncpg://neondb_owner:pw@ep-x-pooler.")
    assert url.endswith("/neondb?ssl=require")


def test_psycopg_form_requires_ssl_too():
    """The checkpointer must not silently fall back to `prefer`."""
    url = psycopg_url(asyncpg_url(NEON))
    assert url.startswith("postgresql://")
    assert url.endswith("/neondb?sslmode=require")


def test_settings_normalize_the_pasted_url():
    settings = Settings(database_url=NEON)
    assert settings.database_url == asyncpg_url(NEON)
    assert settings.psycopg_database_url.endswith("?sslmode=require")


def test_a_url_without_query_is_left_alone():
    url = "postgresql+asyncpg://u:p@localhost:5433/db"
    assert asyncpg_url(url) == url
    assert psycopg_url(url) == "postgresql://u:p@localhost:5433/db"


def test_s3_endpoint_url_sets_host_and_tls():
    host, secure = storage._split_endpoint("https://br-x.storage.c-5.eu-central-1.aws.neon.tech")
    assert host == "br-x.storage.c-5.eu-central-1.aws.neon.tech"
    assert secure is True


def test_bare_s3_host_uses_the_secure_setting(monkeypatch):
    monkeypatch.setattr(storage.settings, "s3_secure", False)
    assert storage._split_endpoint("localhost:9000") == ("localhost:9000", False)
