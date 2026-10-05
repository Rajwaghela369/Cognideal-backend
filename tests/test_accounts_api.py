"""The accounts and contacts routes, over HTTP.

The first route-level tests in the project -- everything else asserts against
services. They go through the real app so the things only the HTTP layer can get
wrong are covered: status codes, `extra="forbid"`, path-vs-body scoping, and the
409 that a bare IntegrityError would have surfaced as a 500.

These two collections are worth that attention because they are the entry point.
Until they existed a fresh install could not create a deal at all: `DealCreate`
requires an `account_id` and `DealStakeholderCreate` an existing `contact_id`,
and only the transcript-resolution path could mint either.
"""

import uuid

import pytest
from sqlalchemy import select

from app.models import Account, Contact, Deal, DealContact, MeetingAttendee
from app.models.enums import DealStage

API = "/api/v1"


@pytest.fixture
def account_body():
    return {"name": "Northwind Probe %s" % uuid.uuid4().hex[:8]}


async def _make_account(client, **overrides):
    body = {"name": "Acct %s" % uuid.uuid4().hex[:8]}
    body.update(overrides)
    res = await client.post(f"{API}/accounts", json=body)
    assert res.status_code == 201, res.text
    return res.json()


async def _make_contact(client, account_id, **overrides):
    body = {"first_name": "Priya", "last_name": "Raman"}
    body.update(overrides)
    res = await client.post(f"{API}/accounts/{account_id}/contacts", json=body)
    assert res.status_code == 201, res.text
    return res.json()


# --------------------------------------------------------------------------
# Accounts
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_read_an_account(client, db):
    created = await _make_account(client, industry="Logistics", hq_region="EMEA")
    assert created["industry"] == "Logistics"

    res = await client.get(f"{API}/accounts/{created['id']}")
    assert res.status_code == 200
    assert res.json()["name"] == created["name"]

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(created["id"])))
    await db.commit()


@pytest.mark.asyncio
async def test_an_unknown_field_is_422_not_a_silent_drop(client):
    """`extra="forbid"` is load-bearing -- see schemas/common.py."""
    res = await client.post(
        f"{API}/accounts", json={"name": "Typo Co", "industri": "Logistics"}
    )
    assert res.status_code == 422
    assert "industri" in res.text


@pytest.mark.asyncio
async def test_a_blank_name_is_refused(client):
    res = await client.post(f"{API}/accounts", json={"name": ""})
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_unknown_account_is_404(client):
    res = await client.get(f"{API}/accounts/{uuid.uuid4()}")
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_list_reports_deal_and_contact_counts(client, db):
    """Correlated subqueries, not joins: two one-to-manys would multiply."""
    account = await _make_account(client)
    account_id = uuid.UUID(account["id"])

    await _make_contact(client, account["id"], first_name="A", last_name="One")
    await _make_contact(client, account["id"], first_name="B", last_name="Two")
    db.add(Deal(account_id=account_id, name="D1", stage=DealStage.DISCOVERY))
    db.add(Deal(account_id=account_id, name="D2", stage=DealStage.DISCOVERY))
    await db.commit()

    res = await client.get(f"{API}/accounts", params={"q": account["name"]})
    assert res.status_code == 200
    row = res.json()["items"][0]
    # 2 and 2, not 4 and 4.
    assert (row["deal_count"], row["contact_count"]) == (2, 2)

    await db.execute(Account.__table__.delete().where(Account.id == account_id))
    await db.commit()


@pytest.mark.asyncio
async def test_patch_updates_only_what_was_sent(client, db):
    account = await _make_account(client, industry="Logistics")
    res = await client.patch(
        f"{API}/accounts/{account['id']}", json={"hq_region": "APAC"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["hq_region"] == "APAC"
    assert body["industry"] == "Logistics", "unsent field was overwritten"

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(account["id"])))
    await db.commit()


@pytest.mark.asyncio
async def test_deleting_an_account_with_deals_is_refused(client, db):
    """Cascading here would silently destroy evidence -- see the route."""
    account = await _make_account(client)
    account_id = uuid.UUID(account["id"])
    db.add(Deal(account_id=account_id, name="Keeps it alive", stage=DealStage.DISCOVERY))
    await db.commit()

    res = await client.delete(f"{API}/accounts/{account['id']}")
    assert res.status_code == 409
    assert "deal" in res.text.lower()
    assert await db.get(Account, account_id) is not None

    await db.execute(Account.__table__.delete().where(Account.id == account_id))
    await db.commit()


@pytest.mark.asyncio
async def test_an_account_with_no_deals_deletes(client, db):
    account = await _make_account(client)
    await _make_contact(client, account["id"])

    res = await client.delete(f"{API}/accounts/{account['id']}")
    assert res.status_code == 204
    assert await db.get(Account, uuid.UUID(account["id"])) is None


# --------------------------------------------------------------------------
# Contacts
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_a_contact_without_an_email(client, db):
    """The transcript path creates contacts from a spoken name alone."""
    account = await _make_account(client)
    contact = await _make_contact(client, account["id"], first_name="Tom", last_name="Alvarez")
    assert contact["email"] is None
    assert contact["account_id"] == account["id"]

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(account["id"])))
    await db.commit()


@pytest.mark.asyncio
async def test_a_duplicate_email_is_409_naming_the_existing_contact(client, db):
    """UNIQUE(account_id, email) left to the database would be a 500."""
    account = await _make_account(client)
    first = await _make_contact(client, account["id"], email="dana@northwind.example")

    res = await client.post(
        f"{API}/accounts/{account['id']}/contacts",
        json={"first_name": "Dana", "last_name": "Dupe", "email": "dana@northwind.example"},
    )
    assert res.status_code == 409
    # The useful part: which contact to link to instead.
    assert first["id"] in res.text

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(account["id"])))
    await db.commit()


@pytest.mark.asyncio
async def test_two_contacts_with_no_email_both_land(client, db):
    """Postgres permits many NULLs under a unique constraint, deliberately."""
    account = await _make_account(client)
    await _make_contact(client, account["id"], first_name="A", last_name="One")
    await _make_contact(client, account["id"], first_name="B", last_name="Two")

    res = await client.get(f"{API}/accounts/{account['id']}/contacts")
    assert len(res.json()) == 2

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(account["id"])))
    await db.commit()


@pytest.mark.asyncio
async def test_the_same_email_on_a_different_account_is_fine(client, db):
    """The constraint is per-account: one person can work at two companies."""
    first = await _make_account(client)
    second = await _make_account(client)
    await _make_contact(client, first["id"], email="shared@example.com")
    await _make_contact(client, second["id"], email="shared@example.com")

    for account in (first, second):
        await db.execute(
            Account.__table__.delete().where(Account.id == uuid.UUID(account["id"]))
        )
    await db.commit()


@pytest.mark.asyncio
async def test_resending_a_contacts_own_email_is_not_a_conflict(client, db):
    """`excluding` in assert_email_free -- a PATCH must not collide with itself."""
    account = await _make_account(client)
    contact = await _make_contact(client, account["id"], email="priya@example.com")

    res = await client.patch(
        f"{API}/accounts/{account['id']}/contacts/{contact['id']}",
        json={"email": "priya@example.com", "title": "Director of IT Security"},
    )
    assert res.status_code == 200
    assert res.json()["title"] == "Director of IT Security"

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(account["id"])))
    await db.commit()


@pytest.mark.asyncio
async def test_a_contact_cannot_be_reached_through_another_accounts_url(client, db):
    """Both ids are matched, so one account's URL cannot edit another's row."""
    owner = await _make_account(client)
    stranger = await _make_account(client)
    contact = await _make_contact(client, owner["id"])

    res = await client.get(f"{API}/accounts/{stranger['id']}/contacts/{contact['id']}")
    assert res.status_code == 404

    res = await client.patch(
        f"{API}/accounts/{stranger['id']}/contacts/{contact['id']}",
        json={"title": "Hijacked"},
    )
    assert res.status_code == 404

    for account in (owner, stranger):
        await db.execute(
            Account.__table__.delete().where(Account.id == uuid.UUID(account["id"]))
        )
    await db.commit()


@pytest.mark.asyncio
async def test_contacts_on_an_unknown_account_are_404(client):
    res = await client.get(f"{API}/accounts/{uuid.uuid4()}/contacts")
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_deleting_a_contact_unresolves_attendees_rather_than_erasing_them(
    client, db, deal_id
):
    """`meeting_attendees.contact_id` is nullable by design.

    The row records that a name spoke in a transcript, which stays true after
    the person is untracked. Deleting the attendee row instead would erase
    evidence that a meeting had a participant at all.
    """
    from app.models import Meeting

    deal = await db.get(Deal, deal_id)
    contact = Contact(account_id=deal.account_id, first_name="Tom", last_name="Alvarez")
    db.add(contact)
    await db.flush()

    meeting = Meeting(deal_id=deal_id, title="Discovery")
    db.add(meeting)
    await db.flush()
    attendee = MeetingAttendee(
        meeting_id=meeting.id, raw_name="Tom Alvarez", contact_id=contact.id
    )
    db.add(DealContact(deal_id=deal_id, contact_id=contact.id))
    db.add(attendee)
    await db.commit()
    attendee_id, contact_id = attendee.id, contact.id

    res = await client.delete(
        f"{API}/accounts/{deal.account_id}/contacts/{contact_id}"
    )
    assert res.status_code == 204

    db.expire_all()
    surviving = await db.get(MeetingAttendee, attendee_id)
    assert surviving is not None, "the attendee row was destroyed"
    assert surviving.contact_id is None, "the link should be NULL, not dangling"
    assert surviving.raw_name == "Tom Alvarez"

    # The stakeholder link does go, because they are no longer a stakeholder.
    link = await db.scalar(
        select(DealContact).where(DealContact.contact_id == contact_id)
    )
    assert link is None


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_reports_the_database(client):
    """Unversioned and outside /api -- a probe must survive a version bump."""
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "database": "ok"}


# --------------------------------------------------------------------------
# Pagination
# --------------------------------------------------------------------------
#
# These exist because the suite had 188 green tests while `?limit=2` was a 422
# on every paginated endpoint. Every list test passed `params={"q": ...}` and
# none ever passed a window, so the one parameter a table cannot work without
# was the one parameter nothing exercised. The envelope made it worse by
# reporting `"limit": 50` -- the unused dependency's default -- so a response
# looked like a correctly paginated page one.


@pytest.mark.asyncio
async def test_limit_and_offset_are_accepted_and_applied(client, db):
    """The regression. A 422 here means `limit` left the filter model."""
    names = []
    for index in range(3):
        account = await _make_account(client, name="Paged %s %d" % (uuid.uuid4().hex[:6], index))
        names.append(account)

    res = await client.get(f"{API}/accounts", params={"limit": 2})
    assert res.status_code == 200, res.text
    body = res.json()
    # Applied, not merely accepted -- and echoed back as what was applied.
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["total"] >= 3, "total counts before the window, not after"

    for account in names:
        await db.execute(
            Account.__table__.delete().where(Account.id == uuid.UUID(account["id"]))
        )
    await db.commit()


@pytest.mark.asyncio
async def test_offset_walks_to_the_next_page(client, db):
    """Page two must not repeat page one."""
    created = [
        await _make_account(client, name="Walk %s %d" % (uuid.uuid4().hex[:6], i))
        for i in range(3)
    ]

    first = await client.get(f"{API}/accounts", params={"limit": 1, "offset": 0})
    second = await client.get(f"{API}/accounts", params={"limit": 1, "offset": 1})
    assert first.status_code == second.status_code == 200
    assert first.json()["items"][0]["id"] != second.json()["items"][0]["id"]
    assert second.json()["offset"] == 1

    for account in created:
        await db.execute(
            Account.__table__.delete().where(Account.id == uuid.UUID(account["id"]))
        )
    await db.commit()


@pytest.mark.asyncio
async def test_a_window_and_a_filter_work_together(client, db):
    """The combination is what a real table sends, and it 422'd before."""
    tag = uuid.uuid4().hex[:8]
    created = [
        await _make_account(client, name="Combo %s %d" % (tag, i)) for i in range(3)
    ]

    res = await client.get(f"{API}/accounts", params={"q": tag, "limit": 2})
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3, "the filter must apply to the count as well"

    for account in created:
        await db.execute(
            Account.__table__.delete().where(Account.id == uuid.UUID(account["id"]))
        )
    await db.commit()


@pytest.mark.asyncio
async def test_the_bounds_on_limit_are_still_enforced(client):
    """Moving the field into the model must not drop its validation."""
    assert (await client.get(f"{API}/accounts", params={"limit": 0})).status_code == 422
    assert (await client.get(f"{API}/accounts", params={"limit": 201})).status_code == 422
    assert (await client.get(f"{API}/accounts", params={"offset": -1})).status_code == 422


@pytest.mark.asyncio
async def test_a_misspelled_filter_is_still_refused(client):
    """`extra="forbid"` is why this fix was not "just drop the config".

    Without it `?nmae=x` returns the whole unfiltered collection with a 200,
    which looks exactly like a working filter that matched everything.
    """
    res = await client.get(f"{API}/accounts", params={"nmae": "typo"})
    assert res.status_code == 422
    assert "nmae" in res.text


@pytest.mark.asyncio
async def test_an_unpaginated_endpoint_still_refuses_a_window(client, db):
    """Contacts returns a bare list, so it must not pretend to paginate.

    Accepting and ignoring `?limit=` is the same lie as rejecting a window it
    could honour -- which is why `ListQuery` is inherited per endpoint rather
    than applied to every filter model.
    """
    account = await _make_account(client)
    res = await client.get(f"{API}/accounts/{account['id']}/contacts", params={"limit": 1})
    assert res.status_code == 422

    await db.execute(Account.__table__.delete().where(Account.id == uuid.UUID(account["id"])))
    await db.commit()
