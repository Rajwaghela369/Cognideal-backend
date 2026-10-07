# CogniDeal — API

The HTTP layer over the schema in `docs/schema/README.md`. Single user, no auth.

---

## 1. Layout

Type-based, not domain-based. Chosen deliberately: the domain-driven layout
(one package per domain, each with its own `router.py`/`schemas.py`/`models.py`)
is what the FastAPI community recommends for large apps, but this codebase has
two domains and a shared schema whose tables reference each other constantly.
Revisit if the number of domains grows past a handful.

```
app/
├── db/                     infrastructure — depends on nothing
│   ├── base.py             Base + naming convention
│   ├── mixins.py           UUIDPrimaryKeyMixin, TimestampMixin
│   └── session.py          engine, SessionLocal, get_db()
├── models/                 SQLAlchemy — the tables. Alembic reads this.
├── queries.py              reusable SQL over models
├── services/               writes that are multi-statement or have a rule
│   ├── deal.py             stage transitions, primary demotion
│   ├── task.py             status ↔ completed_at, fact release
│   ├── meeting.py          status ↔ timestamps, attendee resolution
│   ├── ingest.py           extract, chunk, and the upload ordering
│   ├── storage.py          object storage (MinIO)
│   ├── detect.py           the four deterministic risk rules
│   ├── risk.py             risk status, accept/dismiss
│   └── claims.py           evidence attach, and the orphan problem
├── schemas/                Pydantic — the HTTP contract
│   ├── common.py           Page[T], ORM, WRITE — shared across versions
│   └── v1/
│       ├── account.py      AccountRef
│       ├── task.py
│       ├── document.py
│       └── deal/           core · stakeholder · stage_history · meeting · risk · commitment
└── api/
    ├── deps.py             pagination, get_deal_or_404 — not version-specific
    └── v1/
        ├── router.py       mounts the top-level resources
        └── routes/
            ├── deals/      core · stakeholders · stage_history · meetings ·
            │               attendees · documents · risks · commitments
            ├── tasks.py
            └── documents.py
```

**Every arrow points one way**: `db → models → queries → services → routes`.
`queries.py` sits at the app root rather than in `db/` for exactly this reason —
`models/` imports `db.base` and `db.mixins`, so a query module inside `db/`
importing back from `models/` made the two packages mutually dependent.

**`models/` vs `schemas/`.** `models/` is SQLAlchemy and defines tables;
`schemas/` is Pydantic and defines JSON shapes. The word "schema" here is the
JSON Schema sense — the shape and validation of data — not the Postgres sense,
where a schema is a namespace like `public`. This is the standard FastAPI
convention.

**A resource gets a package, not a file, once it has sub-resources.**
`routes/deals/` holds three modules because `/deals` has three resources under
it; `routes/tasks.py` is a flat module because tasks have none. `schemas/`
mirrors the same shape.

---

## 2. Versioning

Everything mounts under `/api/v1`. `app/api/v1/` holds the routers; `deps.py`
stays outside it because pagination and session handling are not
version-specific.

`schemas/v1/` is versioned alongside the routes, because these models *are*
the contract rather than a convenience for building it: `extra="forbid"` on
`DealCreate` is a promise about what v1 accepts, and a v2 that accepts one more
field is a genuinely different model that has to coexist with this one.

A future `schemas/v2/` does **not** copy the package. It imports whatever is
unchanged and redefines only what moved:

```python
from app.schemas.v1.deal.core import DealCounts, DealListItem   # unchanged
class DealDetail(BaseModel): ...                                # changed
```

### What is not versioned, and why

| layer | versioned | |
|---|---|---|
| `api/v1/routes/` | yes | the URL surface |
| `schemas/v1/` | yes | the wire shapes |
| `api/deps.py` | no | pagination mechanics, not a contract |
| `schemas/common.py` | no | `Page[T]` envelope and the two model configs |
| `services/` | **no** | domain invariants |
| `queries.py` | no | `IS_OVERDUE` must mean one thing everywhere |
| `models/` | no | there is one database |

Everything above the wire is versioned; everything below it is not, and
`services/` sits below.

Versioning `services/` would be the one change that undoes the reason the layer
exists. `apply_stage_change` appends to `deal_stage_history` and maintains
`closed_at`; `demote_primary_stakeholder` must run in an *earlier statement*
than the insert, because a partial unique index cannot be `DEFERRABLE`. Those
rules hold no matter which URL wrote the row — there is one `deals` table. A
`services/v2/` whose copy forgot to clear `closed_at` would make the same deal
read as closed or open depending on which API version the client called: one
database, two truths.

If v2 ever needs genuinely different behaviour, the answer is a parameter or a
sibling function in the same module — visible, and in one place to fix — not a
parallel tree.

---

## 3. Conventions

**Requests and responses are different models.** `DealCreate` is not
`DealDetail`. A single model cannot express "the client may set `name` but not
`risk_level`", and that distinction is the whole point of the column-ownership
rules below.

**`extra="forbid"` on every request and query model** (`schemas/common.py::WRITE`).
FastAPI silently drops unrecognised fields, so without it a client POSTing
`risk_level` gets a 201 and no risk level, and `?stalled_dayz=30` returns the
whole unfiltered pipeline with a 200 — which looks exactly like a working
filter. With it, both are a 422 naming the field.

**Query parameters are models too**, bound with `Annotated[DealFilters, Query()]`.
The `Query()` marker is what keeps them query parameters — a bare Pydantic model
in a handler signature is a request *body*, and a GET with a body is not
reliably handled by clients or proxies.

**Validate once.** Handlers return raw SQLAlchemy `Row` objects and let
`response_model` validate them on the way out; FastAPI calls
`validate_python(..., from_attributes=True)`. Building the Pydantic models in
the handler *and* declaring `response_model` validates every row twice.

**`PATCH`, never `PUT`**, and read the body with `model_dump(exclude_unset=True)`
— never `exclude_none`. Pydantic tracks which keys the client actually sent, so
`{"due_date": null}` (clear it) and an omitted `due_date` (leave it alone) are
distinguishable only through `exclude_unset`.

**Every list endpoint returns `Page[T]`** — `items`, `total`, `limit`, `offset`,
where `total` is the count *before* the window, so a table can render page
numbers without a second request. Sub-resource lists that are inherently short
(stakeholders, stage history) return a plain list instead.

**A shared definition lives in `queries.py`, never in two route modules.**
`NEXT_ACTION`, `IS_OVERDUE`, `IS_STALLED`, `DAYS_IN_STAGE`, `RISK_PRIORITY`.
Two screens quietly disagreeing about what "overdue" means is the kind of drift
nobody notices until the numbers differ.

**Filters are built once per statement.** `paginate()` runs the statement twice,
counted and windowed; filters applied in two places eventually disagree and the
table renders "137 results" over a page drawn from a different set.

**Sort keys are a whitelist**, with the vocabulary in `schemas/` (it validates
the request) and the name → expression mapping in `routes/` (it names columns),
plus an import-time assert that the two agree. `getattr(Model, name)` would turn
a query string into arbitrary column access.

---

## 4. Column ownership

Who may write what. The request models enforce this — a column owned by the
server or the analyzer simply has no field.

| owner | columns | notes |
|---|---|---|
| **server** | `id`, `created_at`, `updated_at` | `gen_random_uuid()`, `func.now()` — the database's clock, not the app's |
| **user** | `name`, `value`, `currency`, `expected_close_date`, `win_probability`, `title`, `description`, `due_date`, `priority`, `notes`, `is_primary` | plain fields |
| **user, via a service** | `stage`, `status` | not column writes: they also maintain `closed_at` / `completed_at` and append to `deal_stage_history` |
| **workflow** | `closed_at`, `completed_at`, `last_activity_at` | derived; never in a request body |
| **analyzer** | `risk_level`, `origin`, `confidence` | read-only over the API |
| **human approval only** | `buying_role`, `influence`, `sentiment` on `deal_contacts` | Layer A is what a salesperson owns. The analyzer writes `extracted_facts` with `status='pending'`; a human accepting one is what creates the link. That is why `deal_contacts` carries no `origin`/`confidence` while every Layer C table does. |

**Derived per request, never stored**: `next_action`, `next_action_due_date`,
`days_in_stage`, `is_overdue`, `account_name`, `deal_name`, and every `counts`
field. A stored copy is a copy that goes stale — the same rule that keeps
`deals` free of a `next_action` column.

---

## 5. Routes

### Deals — `routes/deals/core.py`

```
GET    /api/v1/deals              Page[DealListItem]
POST   /api/v1/deals              201 DealDetail
GET    /api/v1/deals/{deal_id}    DealDetail
PATCH  /api/v1/deals/{deal_id}    DealDetail
DELETE /api/v1/deals/{deal_id}    204
```

Filters: `account_id`, `contact_id`, `stage`*, `risk_level`*, `open`, `q`,
`value_min`/`value_max`, `close_before`/`close_after`, `stale_days`,
`stalled_days`, `stalled`, `sort`, `limit`/`offset`. (`*` = repeatable.)

`stale_days` and `stalled` answer different questions. `stale_days` reads
`last_activity_at` — *has anyone talked to them?* `stalled` compares the newest
`deal_stage_history` row against the per-stage thresholds in
`queries.py::STALL_THRESHOLD_DAYS` — *is any of that talking moving the deal?*
A deal with weekly check-ins and no stage movement for two months is invisible
to the first and is exactly what the second finds.

Thresholds are per-stage, not one number: a security review at 40 days is
normal where a discovery deal at 40 days is not. A flat rule floods the risk
panel with false positives, and a panel that cries wolf stops being read.
**The current values are placeholders** — they should be the 75th percentile of
real dwell time per stage among deals that closed, once the seeder exists.

`stage` in `PATCH` also appends to `deal_stage_history` and sets or clears
`closed_at`, in one transaction. `stage_note` rides along and is rejected
without a `stage`.

### Accounts and contacts — `routes/accounts.py`

```
POST   /accounts                                 create
GET    /accounts                  ?q=            list, with deal_count/contact_count
GET    /accounts/{id}                            detail
PATCH  /accounts/{id}                            partial update
DELETE /accounts/{id}                            409 if it has deals

POST   /accounts/{id}/contacts                   create; 409 on a repeat email
GET    /accounts/{id}/contacts    ?q=            unpaginated picker
GET    /accounts/{id}/contacts/{contact_id}
PATCH  /accounts/{id}/contacts/{contact_id}
DELETE /accounts/{id}/contacts/{contact_id}      unresolves attendees, drops stakeholder links
```

The entry point to the whole API: `DealCreate` requires an `account_id` and
`DealStakeholderCreate` an existing `contact_id`, and before this collection
existed only the transcript-resolution path could mint either — so a stakeholder
had to have spoken on a recorded call before they could be tracked.

Contacts nest under accounts because `contacts.account_id` is the foreign key,
not `deal_id`: one person can appear in several deals with one company, and
everything deal-specific lives on `deal_contacts`. A delete sets
`meeting_attendees.contact_id` back to NULL rather than removing the row — the
transcript still had a speaker of that name, which Phase 2 treats as a signal.

`DELETE /accounts/{id}` refuses while deals exist. The foreign keys would
cascade through meetings, documents, facts and risks, and `claim_evidence`
has no foreign key on `claim_id`, so its links would survive as orphans.

---

### Stakeholders — `routes/deals/stakeholders.py`

```
GET    /api/v1/deals/{deal_id}/stakeholders
POST   /api/v1/deals/{deal_id}/stakeholders                 201 · 409 if the pair exists
PATCH  /api/v1/deals/{deal_id}/stakeholders/{contact_id}
DELETE /api/v1/deals/{deal_id}/stakeholders/{contact_id}    204
```

Addressed by the `(deal_id, contact_id)` pair; the surrogate `id` never leaves
the database. `DELETE` removes the link, never the contact.

Two invariants the database cannot express alone:

- **The contact must belong to the deal's account.** No foreign key can say
  this, so the service layer does — 422.
- **At most one `is_primary` per deal.** Enforced by a partial unique index, so
  a write setting the flag must demote the incumbent in an *earlier statement*:
  partial uniqueness can only be an index, never a constraint, so it can never
  be `DEFERRABLE`.

### Stage history — `routes/deals/stage_history.py`

```
GET /api/v1/deals/{deal_id}/stage-history
```

Read-only, and that is the point. The only writer is `apply_stage_change`,
reached through `PATCH /deals/{id}`. An endpoint that appended here
independently would let the history say a deal reached `negotiation` while
`deals.stage` still said `discovery` — the drift the schema avoids by having no
`status` column. No `PATCH`/`DELETE` either: the table has `created_at` and no
`updated_at` because a transition is a thing that happened.

Oldest first, unlike an activity feed — it reads as a progression, and
`days_in_stage` (a `LEAD()` window over `changed_at`) only makes sense forwards.

### Tasks — `routes/tasks.py`

```
GET    /api/v1/tasks              Page[TaskListItem]
POST   /api/v1/tasks              201 TaskDetail
GET    /api/v1/tasks/{task_id}    TaskDetail
PATCH  /api/v1/tasks/{task_id}    TaskDetail
DELETE /api/v1/tasks/{task_id}    204
```

Top-level, not nested under `/deals`, even though `tasks.deal_id` is NOT NULL:
the tasks page is one table across the whole pipeline, which a nested route
cannot serve, and `?deal_id=` covers the per-deal case. One collection instead
of two that drift.

Filters: `deal_id`, `account_id`, `status`*, `priority`*, `origin`, `open`,
`overdue`, `due_before`/`due_after`, `has_due_date`, `q`, `sort`.

Default sort is deliberately the same ordering as `queries.py::NEXT_ACTION`, so
the top row of `?deal_id=X&open=true` is always that deal's next action. If the
two diverge, the deal page and the task table disagree about what is next.

`status` in `PATCH` maintains `completed_at`: set on `done`, cleared on
reopening, left NULL for `cancelled` — cancelled is not completed.

`DELETE` is for a task that should not have existed; `status='cancelled'` is
for "we decided not to do this" and keeps the record. Deleting a task promoted
from an extracted fact also releases that fact back to `pending`, because
`extracted_facts.promoted_to_id` has no foreign key and would otherwise point
at a row that no longer exists.

---

### Meetings — `routes/deals/meetings.py`

```
GET    /api/v1/deals/{deal_id}/meetings
POST   /api/v1/deals/{deal_id}/meetings
GET    /api/v1/deals/{deal_id}/meetings/{meeting_id}
PATCH  /api/v1/deals/{deal_id}/meetings/{meeting_id}
DELETE /api/v1/deals/{deal_id}/meetings/{meeting_id}
GET    /api/v1/deals/{deal_id}/meetings/{meeting_id}/analysis
POST   /api/v1/deals/{deal_id}/meetings/{meeting_id}/analysis
```

Fully nested, unlike tasks: meetings are read from the deal detail page and
nowhere else, so there is no cross-deal collection. The cost is that every
handler must confirm the meeting belongs to the deal named in the path, which
`get_meeting_or_404` does once — otherwise deal A's URL would serve deal B's
meeting.

`status` in `PATCH` maintains the timestamps: `→ completed` stamps `ended_at` if
none was supplied, `→ scheduled` clears both (a meeting moved back did not
happen), `→ cancelled` leaves them (a meeting that started and was abandoned
really did start).

**The analysis endpoints are a contract, not a feature yet.** `GET` is real —
it reads the columns. `POST` sets `analysis_status='queued'` and returns 202, but
**nothing consumes that queue**, so a meeting stays queued until the Meeting
Analyzer exists. They are here so the screen can be built against its final
shape; the route module says so in a block comment.

`analysis_status` is never in a `PATCH` body — the analyzer owns it.
`transcript_document_id` is absent too: it belongs to the upload flow, not to a
client PATCHing a foreign key.

### Attendees and participants — `routes/deals/attendees.py`

```
GET    …/meetings/{meeting_id}/attendees
POST   …/meetings/{meeting_id}/attendees
PATCH  …/meetings/{meeting_id}/attendees/{attendee_id}
DELETE …/meetings/{meeting_id}/attendees/{attendee_id}
POST   …/meetings/{meeting_id}/attendees/{attendee_id}/resolve
GET    /api/v1/deals/{deal_id}/participants
```

`raw_name` is required and `contact_id` is the optional *resolution* of it —
never the reverse. A transcript speaker is frequently not in `contacts` yet, and
those are exactly the people worth surfacing, so `?resolved=false` is the
missing-stakeholder signal.

**`/participants` is the roll-up across every meeting on the deal**, and it is
the screen that makes the signal actionable. It shows two things no other route
can:

```
Priya Raman      champion        stakeholder ✓   4 meetings
Dana (procurement)   —           not tracked ✗   1 meeting    ← resolve me
Tom Reeves       economic_buyer  stakeholder ✓   0 meetings   ← never shows up
```

Someone influencing the deal who is not tracked, and a stakeholder who has never
turned up. The second requires its own `SELECT`: a stakeholder who never attended
has no `meeting_attendees` row at all, so no join from the attendee side reaches
them.

**`resolve` turns a transcript name into tracked data in one transaction** —
create the contact, point the attendee at it, optionally add the stakeholder row.
Three writes for one human action, so it cannot half-fail. `account_id` is never
in the body: it is derived from attendee → meeting → deal → account, which makes
a cross-account contact structurally impossible rather than something the service
must check. A repeat email returns **409 naming the existing contact** ("link
instead of creating") rather than a 500 from `UNIQUE(account_id, email)`.

This is the same shape as `extracted_facts` promotion: a raw signal, a human
approving it, Layer A rows created as a result. The attendee row *is* the staging
record, so no fact is involved.

### Documents — `routes/deals/documents.py`, `routes/documents.py`

```
POST   /api/v1/deals/{deal_id}/documents       multipart; 201, or 200 on a content_hash match
GET    /api/v1/deals/{deal_id}/documents
GET    /api/v1/documents/{document_id}
GET    /api/v1/documents/{document_id}/preview 302 → presigned object-storage URL
DELETE /api/v1/documents/{document_id}
GET    /api/v1/chunks/{chunk_id}
```

**Upload is synchronous and all-or-nothing.** There is no `ingest_status` to poll
because a row exists only when everything succeeded; anything that fails leaves
no row. That is affordable because embeddings are not computed yet — when search
lands, the slow step moves to a batch backfill rather than into this request.

Write ordering is the design, not an implementation detail — see
`services/ingest.py` and the `documents` section of the schema doc. **Postgres is
the source of truth; object storage may hold orphans, never the reverse.**

**Re-upload is a designed behaviour, not an error.** `UNIQUE(content_hash)` makes
it idempotent, so the second upload of the same file returns **200 with the
existing document** rather than a 500 from an `IntegrityError`.

**Preview redirects to a presigned URL** rather than streaming: proxying bytes
would make every preview an application request. The URL is signed against
`MINIO_PUBLIC_ENDPOINT`, the host the *browser* will use, and the client is
supplied with `MINIO_REGION` so signing needs no network call — without it the
client makes a live `?location=` request and the backend tries to reach the
browser's hostname from inside its own container.

**`/chunks/{chunk_id}` is the citation path, not search.** A Layer C claim stores
a `chunk_id` and char offsets; the UI resolves it here to show the quote, and
Gate 0 reads the same row to re-check the snippet still occurs verbatim. Top-level
because the caller holds only the chunk id.

**No writes on chunks, ever.** They are derived and immutable; re-ingest is
delete-and-re-upload.

### Risks and recommendations — `routes/deals/risks.py`

```
GET    /api/v1/deals/{deal_id}/risks                     each risk + nested recommendation
GET    /api/v1/deals/{deal_id}/risks/{risk_id}           evidence expanded
PATCH  /api/v1/deals/{deal_id}/risks/{risk_id}           mitigating | resolved | dismissed
GET    /api/v1/deals/{deal_id}/risks/{risk_id}/evidence
POST   /api/v1/deals/{deal_id}/analysis                  run the deterministic detector

GET    /api/v1/deals/{deal_id}/recommendations
GET    /api/v1/deals/{deal_id}/recommendations/{rec_id}
GET    /api/v1/deals/{deal_id}/recommendations/{rec_id}/evidence
POST   /api/v1/deals/{deal_id}/recommendations/{rec_id}/accept
POST   /api/v1/deals/{deal_id}/recommendations/{rec_id}/dismiss
```

**The recommendation is nested inside the risk**, not a parallel list. The user
does not think "show me risks" then "show me recommendations" — they think "what
is wrong and what do I do about it", and those are one card. A recommendation
without its risk is advice with no reason; a risk without its recommendation is a
complaint with no fix.

Only `status` is writable on a risk. The claim itself belongs to the detector —
editing what it asserted would destroy the record of what it asserted. If a risk
is wrong, dismiss it.

**`accept` does not silently create a task.** The body is the prefilled form as
the user edited it, and `due_date` is **required** even though `tasks.due_date` is
nullable: accepting means committing to *when*, and undated committed work is how
a task list becomes noise. One transaction creates the task with `origin='ai'`,
sets `created_task_id`, `status` and `decided_at`.

**`dismiss` requires a reason.** Remembered rather than deleted, so the detector
stops re-suggesting it and so "why do suggestions get refused" is answerable.

The `/recommendations` list is secondary to the panel. It earns its place for the
*proactive* recommendations that have no `source_risk_id` to nest under, and for
evaluation.

`evidence_count` and the evidence join are explicit correlated subqueries keyed
on `(claim_type, claim_id)` — `claim_evidence.claim_id` carries no foreign key, so
SQLAlchemy cannot express it as a relationship.

### Commitments — `routes/deals/commitments.py`

```
GET    /api/v1/deals/{deal_id}/commitments
POST   /api/v1/deals/{deal_id}/commitments
PATCH  /api/v1/deals/{deal_id}/commitments/{commitment_id}
DELETE /api/v1/deals/{deal_id}/commitments/{commitment_id}
```

Distinct from `tasks`, and the distinction is `owner_side`. A task is work *you*
committed to; a commitment is a promise tracked, and the valuable half is
`owner_side='customer'` — "they will send the SOC 2 report" is not something you
can do, but it going unmet is what kills the deal.

**A customer commitment needs an owner** (`owner_contact_id` or `owner_name`) —
a promise with nobody attached cannot be chased, and `MISSED_COMMITMENT` would
have no one to point at. Our own side may be unowned: "we will send the
questionnaire" is the team.

`origin='user'`: anything created here was typed by a person. `source_fact_id`,
`confidence` and `origin` are absent from the request models — the fact-promotion
path owns them.

`DELETE` is for a commitment logged in error; `status='waived'` is the soft
version and keeps the record.

## 6. Status codes

| | |
|---|---|
| `200` | read, or a successful `PATCH` |
| `201` | created, with a `Location` header |
| `204` | deleted, empty body |
| `404` | unknown id in the path |
| `202` | accepted, not done — `POST …/meetings/{id}/analysis` only |
| `302` | `…/documents/{id}/preview` → a presigned object-storage URL |
| `409` | a state or uniqueness conflict: duplicate stakeholder pair, duplicate attendee, already-resolved attendee, an email already used on the account, a recommendation already accepted or dismissed |
| `413` | upload above `max_upload_bytes` — upload is synchronous, there is no queue to absorb it |
| `415` | a file whose text cannot be extracted |
| `422` | validation: unknown field, bad enum, inverted range, unknown sort key, an id in the body that does not resolve, a cross-account contact, `stage_note` without `stage`, `ended_at` before `started_at`, a customer commitment with no owner |
| `502` | object storage unreachable during upload — **nothing was saved** |

A referenced-but-missing id in a *body* is a 422, not a 404 — the path resolved
fine; the payload is what is wrong. Left to the database it would surface as a
foreign-key `IntegrityError` and reach the client as a 500.

---

## 7. Known gaps

- ~~**No worker.**~~ **Closed.** `backend/worker.py` runs the graph and carries
  four pollers (analysis claim, dirty deals, nightly sweep, abandoned-stream
  reaper). Writes invalidate through `services/analysis.py:record_change`, and
  `GET /deals/{id}/analysis` reports the pending state. *But the worker
  container has never been started and has no `langchain` installed — see the
  root README's Known gaps.*
- **Placeholder thresholds** in `queries.py`: `STALL_THRESHOLD_DAYS`
  (14/21/30/60/21) and `CLOSE_DATE_WARNING_DAYS` (21). They should be derived
  from real dwell-time data once a seeder exists.
- **No `origin` column on `meetings` or `deal_contacts`.** Both `summary`/
  `sentiment` and `buying_role`/`influence` can now be written by a human, and
  will be writable by an analyzer — with no way to tell which did. Every Layer C
  table carries `origin`; these two Layer A tables do not.
- **`commitments` has no `met_at`.** Unlike deals, tasks, meetings and risks
  there is no timestamp maintained alongside `status`; `updated_at` is the only
  record of when it was marked.
- **`confidence` is not comparable across sources.** Deterministic risks are
  written with `1.0` because a SQL rule is not a guess, which means the column
  means something different for them than it will for LLM output.
- ~~**Nothing writes `deals.last_activity_at`.**~~ **Closed.**
  `services/activity.py:touch_deal` advances it, and `gone_quiet` reads it.
- ~~**No test suite.**~~ **Closed.** 173 tests, and
  `tests/test_model_registry.py` exists. Caveat worth keeping: until
  `tests/test_accounts_api.py` every test asserted against *services*, so the
  other 60 endpoints still have no route-level coverage — status codes,
  `extra="forbid"` and path-vs-body scoping are only exercised for accounts and
  contacts. Schema task 8.4 remains open.
- **Missing index** `(deal_id, due_date) WHERE status = 'open'` on `tasks`,
  which would serve both `NEXT_ACTION` and the per-deal task list.
- **`release_source_fact` is untested** — no `extracted_facts` rows exist yet.
- **Bypassing `services/claims.py` still creates orphans.** A raw
  `DELETE FROM risks` in psql leaves `claim_evidence` rows pointing at nothing,
  because `claim_id` has no foreign key. The API paths handle it; direct SQL and
  the future seeder must call `delete_claim_links` themselves.

### Paid off

- ~~`claim_evidence` orphaned on deal or document delete.~~ `services/claims.py`
  now clears the polymorphic links in the same transaction, and both delete
  handlers call it. Verified: deleting a deal with 4 risks and 4 recommendations
  took all 8 links and 8 evidence rows with it, leaving none behind.
