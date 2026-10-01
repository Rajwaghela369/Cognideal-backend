"""Shared fixtures.

The engine is disposed after every test on purpose. ``app.db.session.engine``
is a module-level singleton whose pool binds each asyncpg connection to the
event loop that first used it, and pytest-asyncio gives each test its own loop
-- so a pooled connection reused by the next test raises "attached to a
different loop". Disposing between tests costs a reconnect and removes the
entire class of failure. (It is the same trap that made `TestClient` unusable in
``verify_phase0_http.py``.)
"""

import uuid

import pytest
import pytest_asyncio

from app.core.config import settings
from app.db.session import SessionLocal, engine
from app.models import Account, Deal
from app.models.enums import DealStage


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test may call a provider. Enforced, not assumed.

    Without this the suite's behaviour depends on whatever ``AI_ENABLED`` the
    developer happens to have in ``backend/.env`` -- and when it is on, pytest
    silently starts spending money and needing the network. One test already
    relied on the flag being off and passed for the wrong reason until the key
    was configured.

    Tests that need a model stub ``client.structured`` directly, which bypasses
    this gate because it never reaches ``chat_model``. The live checks live in
    ``verify_ai_*.py`` and are run by hand.
    """
    monkeypatch.setattr(settings, "ai_enabled", False)


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def db():
    async with SessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def deal_id(db):
    """A throwaway account and deal, removed afterwards.

    Named with a uuid fragment so a crashed run leaves findable rows rather
    than colliding with the next one.
    """
    account = Account(name="TEST %s" % uuid.uuid4().hex[:8])
    db.add(account)
    await db.flush()
    deal = Deal(account_id=account.id, name="SecureFlow", stage=DealStage.DISCOVERY)
    db.add(deal)
    await db.flush()
    account_id, created = account.id, deal.id
    await db.commit()

    yield created

    async with SessionLocal() as cleanup:
        obsolete = await cleanup.get(Account, account_id)
        if obsolete is not None:
            await cleanup.delete(obsolete)
            await cleanup.commit()
