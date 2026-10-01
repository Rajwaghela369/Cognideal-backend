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

from app.db.session import SessionLocal, engine
from app.models import Account, Deal
from app.models.enums import DealStage


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
