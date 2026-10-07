# CogniDeal — Data Model

The schema for the MVP: a single-user, evidence-grounded sales copilot backed by
Postgres + pgvector. No auth, no organizations, no CRM integration.

This document is the reference the SQLAlchemy models and Alembic migrations are
built from. `TASKS.md` in this directory tracks the implementation.

---

## 1. Design principles

**Three layers, one direction of flow.**

```
  Layer A  Deal domain      facts a salesperson owns   ──┐
  Layer B  Knowledge        unstructured input + vectors ─┼─> Layer C  AI output
  Layer C  Assertions       what the model claims, and  ──┘            cites both
                            what backs each claim
```

Layer A is the source of truth. Layer B is everything the customer said or sent.
Layer C is everything the model asserts — and **nothing in Layer C is allowed to
exist without a link back to Layer A or Layer B**. That rule is the entire
product thesis expressed as a constraint.

**Conventions**

| Concern         | Rule                                                                    |
| --------------- | ------------------------------------------------------------------------ |
| Primary keys    | `uuid`, `server_default=gen_random_uuid()` (native in PG13+, no `pgcrypto` needed) |
| Instants        | `timestamptz` — never `timestamp`, never `date`                        |
| Calendar dates  | `date` only where a wall-clock day is genuinely meant (`due_date`, `expected_close_date`) |
| Money           | `numeric(14,2)` + `currency char(3)`                                    |
| Free text       | `text`, never `varchar(n)` unless a real limit exists                  |
| Email           | `varchar(320)`                                                          |
| Timestamps      | every table carries `created_at`; mutable tables also carry `updated_at` |

**Enum strategy**

| Approach                          | Used for                                                            | Why |
| ---------------------------------- | -------------------------------------------------------------------- | --- |
| Native Postgres enum               | `deal_stage`, `severity`, `risk_level`, `verdict`                   | Sets that will not churn |
| `text` + `CHECK` constraint        | `risk_type`, `fact_type`, `action_type`, `activity_type`             | Sets that *will* churn as prompts are tuned — altering a native enum in Alembic is more friction than it's worth for values that change weekly during agent development |

---

## 2. Layer A — Deal domain

Human-owned records. These are what the future synthetic seeder writes.

### `accounts`

| Column          | Type / reference |
| ---------------- | ------------------ |
| `id`            | uuid, PK          |
| `name`          |                   |
| `industry`      |                   |
| `website`       |                   |
| `employee_band` |                   |
| `hq_region`     |                   |
| `created_at`    |                   |
| `updated_at`    |                   |

### `contacts`

| Column         | Type / reference     |
| --------------- | ---------------------- |
| `id`           | uuid, PK              |
| `account_id`   | → `accounts`          |
| `first_name`   |                       |
| `last_name`    |                       |
| `email`        |                       |
| `title`        |                       |
| `phone`        |                       |
| `created_at`   |                       |
| `updated_at`   |                       |

Contacts belong to the **account**, not to a deal. A stakeholder can appear in
several deals with the same company, and the selling metadata is per-deal.

### `deals`

| Column                 | Type / reference           |
| ----------------------- | ---------------------------- |
| `id`                   | uuid, PK                    |
| `account_id`           | → `accounts`                |
| `name`                 |                             |
| `value`                | `numeric(14,2)`             |
| `currency`             |                             |
| `stage`                | `deal_stage` enum           |
| `win_probability`      | `smallint`                  |
| `risk_level`           | `risk_level` enum           |
| `expected_close_date`  | `date`                      |
| `closed_at`            |                             |
| `last_activity_at`     |                             |
| `created_at`           |                             |
| `updated_at`           |                             |

- **No `status` column.** `stage` carries the whole lifecycle; the existing
  `deal_stage` enum already includes `closed_won` / `closed_lost`. Open pipeline
  is `stage NOT IN ('closed_won','closed_lost')`. Two columns encoding the same
  lifecycle always drift apart.
- **No `next_action` / `next_action_due_date`.** Derived from the oldest open row
  in `tasks`. A denormalized copy of "the next action" is a copy that goes stale.

### `deal_contacts`

| Column         | Type / reference                                                  |
| --------------- | -------------------------------------------------------------------- |
| `id`           | uuid, PK                                                             |
| `deal_id`      | → `deals`, `UNIQUE(deal_id, contact_id)`                             |
| `contact_id`   | → `contacts`                                                         |
| `buying_role`  | `∈ (champion, economic_buyer, technical, blocker, influencer, unknown)` |
| `influence`    |                                                                      |
| `sentiment`    |                                                                      |
| `is_primary`   | `bool`                                                              |
| `notes`        |                                                                      |
| `created_at`   |                                                                      |
| `updated_at`   |                                                                      |

This table is what makes *"no economic buyer has ever attended a meeting"* a SQL
query instead of a model's guess. Missing-stakeholder detection lives or dies here.

**`(deal_id, contact_id)` is still the identity.** It is what the API addresses a
row by — `/api/v1/deals/{deal_id}/stakeholders/{contact_id}` — and the UNIQUE constraint
enforces it exactly as the old composite primary key did. The surrogate `id` is
internal and never appears in a request or a response.

It exists because `extracted_facts.promoted_to_id` is a single `uuid`. That column
is how a fact a human accepted records the Layer A row it became, and
`deal_contacts` was the one promotion target it could not name: a pair is not a
uuid. A `stakeholder` fact could be accepted but never record where it went. Every
other promotion target already had a surrogate key. Added in
`0004_deal_contacts_id`.

**`is_primary` means *the* main contact on this deal** — whose name the pipeline
row shows, who the follow-up goes to. Distinct from `buying_role='champion'` (who
sells for you internally) and from `economic_buyer` (who signs); a deal can have a
champion who is not your day-to-day contact. The word doing the work is *the*:

```sql
CREATE UNIQUE INDEX uq_deal_contacts_deal_id_primary ON deal_contacts (deal_id)
  WHERE is_primary;
```

At most one per deal, zero allowed. Without it two rows can both carry the flag
and `… WHERE is_primary LIMIT 1` returns whichever the planner picks — an answer
free to change between two identical requests. The path there is the ordinary UI
flow, not an edge case: flag Priya, later flag Marcus, nobody unflags Priya.

The cost is an ordering rule on every writer. Partial uniqueness can only ever be
an *index*, never a constraint — a constraint cannot carry a `WHERE` — so it can
never be `DEFERRABLE`, and it is checked at the end of each statement. Demoting
the incumbent must therefore happen in an **earlier statement**, not merely the
same transaction. Reverse the order and every primary swap fails. See
`services/deal.py::demote_primary_stakeholder`.

**Writes are human-only.** Layer A is what a salesperson owns, so the analyzer
never writes `buying_role` or `sentiment` directly — it writes `extracted_facts`
with `status='pending'`, and a human accepting one is what creates or updates the
link. That is why this table carries no `origin` / `confidence` columns while
every Layer C table does: there is nothing here the model wrote.

### `deal_stage_history`

| Column        | Type / reference |
| -------------- | ------------------ |
| `id`          | uuid, PK          |
| `deal_id`     | → `deals`         |
| `from_stage`  |                   |
| `to_stage`    |                   |
| `changed_at`  |                   |
| `note`        |                   |

Feeds stalled-deal risk detection. Cheap to maintain, impossible to reconstruct
later if skipped.

**Append-only, and the API is read-only.** `GET /api/v1/deals/{deal_id}/stage-history` is the
whole surface. The only writer is `services/deal.py::apply_stage_change`, reached
through `PATCH /api/v1/deals/{deal_id}` with a `stage`, which writes the history row,
`deals.stage` and `closed_at` in one transaction. An endpoint that appended here
independently would let the history say a deal reached `negotiation` while
`deals.stage` still said `discovery` — the same drift the missing `status` column
exists to prevent. No `PATCH` or `DELETE`: the table carries `created_at` and no
`updated_at` because a transition is a thing that happened, and a fumbled stage
change leaves two honest rows rather than one edited one.

Every read is "one deal's transitions, in order" — the timeline, and the `LEAD()`
that derives time-in-stage from it — so the index is composite and `deal_id`-leading
(`ix_deal_stage_history_deal_id_changed_at`, `0005_stage_history_idx`), which serves
the plain lookup too.

**`stalled_days` is not `stale_days`.** `deals.last_activity_at` answers "has anyone
talked to them"; the newest row here answers "is any of that talking moving the
deal". A deal with weekly check-ins and no stage movement for two months looks
healthy to the first question and is exactly what `RiskType.STALLED_STAGE` is for.
Both are filters on `GET /api/v1/deals`; the derivation lives once, in
`queries.py::STAGE_CHANGED_AT`.

### `meetings`

| Column                    | Type / reference                     |
| -------------------------- | --------------------------------------- |
| `id`                      | uuid, PK                               |
| `deal_id`                 | → `deals`                              |
| `title`                   |                                        |
| `meeting_type`            |                                        |
| `status`                  |                                        |
| `scheduled_at`            |                                        |
| `started_at`              |                                        |
| `ended_at`                |                                        |
| `transcript_document_id`  | → `documents` (nullable)               |
| `summary`                 | `text`                                 |
| `sentiment`               |                                        |
| `analysis_status`         | `∈ (not_started, queued, running, complete, failed)` |
| `analyzed_at`             |                                        |
| `created_at`              |                                        |
| `updated_at`              |                                        |

`analysis_status` is what the Meeting Analyzer screen polls.

### `meeting_attendees`

| Column          | Type / reference          |
| ---------------- | ---------------------------- |
| `id`            | uuid, PK                    |
| `meeting_id`    | → `meetings`                |
| `contact_id`    | → `contacts` (nullable)     |
| `raw_name`      | `text`, **not null**        |
| `is_internal`   | `bool`                      |
| `attended`      | `bool`                      |

`raw_name` is required; `contact_id` is the optional **resolution** of that name
to a known person, not the other way round. Two reasons:

- Transcript speakers frequently are not in `contacts` yet, and those are
  precisely the people worth surfacing — a name that maps to no known contact is
  the raw signal for "missing stakeholder."
- Attendance is a historical fact. Deleting a contact sets `contact_id` to NULL
  via `ON DELETE SET NULL`, and a row whose only identity was that link would be
  left anonymous. An earlier CHECK requiring one or the other made such contacts
  **undeletable entirely**. Snapshotting the name lets the row degrade to
  "unresolved attendee" instead of breaking. Fixed in `0003_attendee_name`.

### `tasks`

| Column            | Type / reference                          |
| ------------------ | -------------------------------------------- |
| `id`              | uuid, PK                                    |
| `deal_id`         | → `deals`                                   |
| `title`           |                                             |
| `description`     |                                             |
| `due_date`        | `date`                                      |
| `status`          | `∈ (open, done, cancelled)`                 |
| `priority`        |                                             |
| `origin`          | `∈ (user, ai)`                              |
| `source_fact_id`  | → `extracted_facts` (nullable)              |
| `completed_at`    |                                             |
| `created_at`      |                                             |
| `updated_at`      |                                             |

Powers the dashboard's overdue-actions panel and the deal's next action.

### `activities` — dropped in `0007_drop_activities`

The timeline is **derived**, not stored. Every one of the table's seven
`activity_type` values already had a primary home:

| `activity_type` | where that event actually lives |
| ---------------- | -------------------------------- |
| `meeting` | `meetings` |
| `stage_change` | `deal_stage_history` |
| `task_completed` | `tasks.completed_at` |
| `document_upload` | `documents.uploaded_at` |
| `email` | `documents`, `source_type='email'` |
| `note` | `documents`, `source_type='note'` |
| `call` | `meetings`, `meeting_type='check_in'`, no transcript |

Not one was a primary record, which makes `activities` the same pattern this
document rejects twice above — no `next_action` column, no `status` column —
for the same reason: **a stored copy drifts from what it copies.**

The cost of keeping it was a write inside every service that touches a deal:
`apply_stage_change`, task status, meeting create and status, document upload,
attendee resolve. Any one of those forgotten produces a timeline that is
silently incomplete, which is worse than no timeline because it looks complete.

The deal timeline is a `UNION ALL` over the four source tables, discriminated
by the same vocabulary:

```sql
SELECT 'meeting'        AS kind, title,    scheduled_at AS occurred_at FROM meetings            WHERE deal_id = :d
UNION ALL
SELECT 'stage_change',        to_stage, changed_at                     FROM deal_stage_history  WHERE deal_id = :d
UNION ALL
SELECT 'task_completed',      title,    completed_at                   FROM tasks               WHERE deal_id = :d AND status = 'done'
UNION ALL
SELECT 'document_upload',     title,    uploaded_at                    FROM documents           WHERE deal_id = :d
ORDER BY occurred_at DESC
LIMIT :n;
```

Already served by `ix_meetings_deal_id_scheduled_at`,
`ix_deal_stage_history_deal_id_changed_at` and `ix_tasks_open_due_date`. It
cannot drift, because there is nothing to keep in sync.

**The `ActivityType` enum is kept.** It stops being a column and becomes the
discriminator the `UNION` emits per row.

Reversible if the trade ever flips: the table was append-only, so
`0007`'s `downgrade()` recreates it and the same query backfills it.

---

## 3. Layer B — Knowledge and ingest

### `documents`

| Column                | Type / reference                                                  |
| ---------------------- | -------------------------------------------------------------------- |
| `id`                  | uuid, PK                                                             |
| `deal_id`             | → `deals` (nullable)                                                 |
| `account_id`          | → `accounts` (nullable)                                              |
| `source_type`         | `∈ (meeting_transcript, email, proposal, contract, note)`           |
| `title`               |                                                                      |
| `original_filename`   |                                                                      |
| `storage_uri`         | **not null** — object key in MinIO, `documents/{content_hash}`        |
| `mime_type`           |                                                                      |
| `byte_size`           |                                                                      |
| `content_hash`        | **not null** — sha256; the idempotency key *and* the storage key      |
| `occurred_at`         |                                                                      |
| `uploaded_at`         |                                                                      |
| `created_at`          |                                                                      |
| `updated_at`          |                                                                      |

**One table for every unstructured input.** Separate tables per type would fork
retrieval five ways for no benefit — a RAG query wants "everything relevant to
this deal," not five unions.

**A row exists only when the document is fully ingested** — text extracted,
chunks written, bytes in object storage. `ingest_status` and `ingest_error` were
dropped in `0008_document_all_or_nothing` because there is no half-ingested
state to report: anything that fails leaves no row, so every document the API
returns is usable.

That pushes the guarantee into the write ordering, since MinIO and Postgres
cannot share a transaction:

```
extract text → split chunks → INSERT (flush, NOT committed)
                            → PUT to object storage
                            → COMMIT
```

The Postgres transaction stays open across the upload. A failed upload rolls the
rows back with no compensation code to get wrong; a crash before the commit
leaves an orphaned object — invisible, harmless, reapable. Committing first would
leave a visible document whose bytes were never written and whose preview 404s.
Deletes follow the same rule in reverse. **Postgres is the source of truth;
object storage may hold orphans, never the reverse.**

**`raw_text` is gone too**, and was always redundant: the chunks ordered by
`chunk_index` *are* the text. The original file is served from object storage
for preview, so keeping both meant two copies of one thing that could disagree
after a re-ingest.

- `occurred_at` ≠ `uploaded_at`. Recency ranking must use **when the conversation
  happened**, not when the file was dragged in. A transcript from June uploaded
  today is still a transcript from June.
- `content_hash` makes re-upload idempotent and keeps the future seeder re-runnable.

### `document_chunks`

| Column          | Type / reference                        |
| ---------------- | ------------------------------------------ |
| `id`            | uuid, PK                                  |
| `document_id`   | → `documents`, `ON DELETE CASCADE`        |
| `chunk_index`   | `int`                                     |
| `content`       | `text`                                    |
| `token_count`   | `int`                                     |
| `embedding`     | `vector(1536)`                            |
| `metadata`      | `jsonb`                                   |
| `created_at`    |                                           |

`UNIQUE(document_id, chunk_index)`

`metadata` carries `{speaker, page, char_start, char_end}` — this is what turns a
retrieved chunk into a **clickable citation** rather than an unattributed blob.

**Chunks are immutable.** If a document is re-ingested, write a new set of chunks
and retain the old ones. Re-chunking in place silently rots every citation that
already points into that document, and Gate 0 (§5) starts failing claims that
were perfectly good. In practice re-ingest is delete-and-re-upload, so nothing
in the API can re-cut an existing document.

**`embedding` is NULL for now.** Nothing retrieves by similarity in the MVP, and
computing embeddings at upload would put an external API call inside a request
that is synchronous by design (see `documents` above). The chunks and their
offsets do not change, so embeddings can be backfilled in a batch job when
search lands — no citation rots. Until then `document_chunks` exists to serve
**citations**: a chunk id plus char offsets, so a Layer C claim can point at an
exact span. That is a different reason to chunk than RAG, and it means the HNSW
index sits unused.

---

## 4. Layer C — Assertions and evidence

### `evidence`

| Column          | Type / reference                       |
| ---------------- | ----------------------------------------- |
| `id`            | uuid, PK                                 |
| `deal_id`       | → `deals`                                |
| `source_kind`   | `∈ (document, record, derived)`          |
| `document_id`   | → `documents` (nullable)                 |
| `chunk_id`      | → `document_chunks` (nullable)           |
| `record_ref`    | `jsonb`                                  |
| `snippet`       | `text`                                   |
| `char_start`    | `int`                                    |
| `char_end`      | `int`                                    |
| `speaker`       |                                          |
| `occurred_at`   |                                          |
| `created_at`    |                                          |

A piece of evidence is a **locatable span**, not a description. Two flavors:

- `source_kind='document'` → `chunk_id` + `char_start`/`char_end` point at an
  exact stretch of a transcript, email or contract.
- `source_kind='record'` → `record_ref = {"table":"deals","id":"…","field":"expected_close_date"}`
  points at a field in our own database.

The `record` flavor is not an afterthought. **Most risk detection reasons over
structured state, not over quotes** — "close date is 21 days out", "stage has not
moved in 58 days", "no economic buyer has attended a meeting". If evidence could
only point at document chunks, every one of those risks would render uncited and
look like a hunch, and the evidence-coverage metric would lie.

### `claim_evidence`

| Column         | Type / reference                                                        |
| --------------- | --------------------------------------------------------------------------- |
| `id`           | uuid, PK                                                                    |
| `claim_type`   | `∈ (fact, commitment, risk, recommendation, chat_message)`                 |
| `claim_id`     | `uuid` (no FK — polymorphic, see §4.1)                                     |
| `evidence_id`  | → `evidence`, `ON DELETE CASCADE`                                          |
| `relevance`    | `numeric(3,2)`                                                             |
| `created_at`   |                                                                             |

`UNIQUE(claim_type, claim_id, evidence_id)`

**This is the spine of the product.** See §4.1.

### `claim_validations`

| Column               | Type / reference                                              |
| --------------------- | ------------------------------------------------------------------ |
| `id`                 | uuid, PK                                                           |
| `claim_type`         |                                                                     |
| `claim_id`           | `uuid`                                                             |
| `verdict`            | `∈ (supported, partial, contradicted, unsupported)`                |
| `method`             | `∈ (deterministic, llm, human)`                                    |
| `rationale`          | `text`                                                             |
| `model`              |                                                                     |
| `validator_version`  |                                                                     |
| `checked_at`         |                                                                     |

Append-only: one row per validation run. See §5.

### `extracted_facts`

| Column             | Type / reference                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| `id`               | uuid, PK                                                                                         |
| `deal_id`          | → `deals`                                                                                        |
| `meeting_id`       | nullable                                                                                         |
| `document_id`      | nullable                                                                                         |
| `fact_type`        | `∈ (requirement, objection, stakeholder, commitment, deadline, budget, competitor, decision_criteria)` |
| `content`          | `text`                                                                                           |
| `payload`          | `jsonb`                                                                                          |
| `confidence`       | `numeric(3,2)`                                                                                   |
| `status`           | `∈ (pending, accepted, rejected, superseded)`                                                    |
| `promoted_to_type` |                                                                                                   |
| `promoted_to_id`   |                                                                                                   |
| `extracted_at`     |                                                                                                   |
| `reviewed_at`      |                                                                                                   |

### `commitments`

| Column             | Type / reference                       |
| -------------------- | ------------------------------------------ |
| `id`               | uuid, PK                                   |
| `deal_id`          | → `deals`                                  |
| `source_fact_id`   | nullable                                   |
| `description`      |                                            |
| `owner_side`       | `∈ (us, customer)`                         |
| `owner_contact_id` | nullable                                   |
| `owner_name`       |                                            |
| `due_date`         | `date`                                     |
| `status`           | `∈ (pending, met, missed, waived)`         |
| `origin`           |                                            |
| `confidence`       |                                            |
| `created_at`       |                                            |
| `updated_at`       |                                            |

### `risks`

| Column               | Type / reference |
| ---------------------- | ------------------ |
| `id`                 | uuid, PK          |
| `deal_id`            | → `deals`         |
| `risk_type`          |                   |
| `title`              |                   |
| `description`        |                   |
| `severity`           |                   |
| `status`             |                   |
| `origin`             |                   |
| `confidence`         |                   |
| `first_detected_at`  |                   |
| `last_seen_at`       |                   |
| `resolved_at`        |                   |
| `created_at`         |                   |
| `updated_at`         |                   |

```sql
CREATE UNIQUE INDEX uq_risks_open_type ON risks (deal_id, risk_type)
  WHERE status = 'open';
```

Re-running the analyzer must **bump `last_seen_at`, not insert a fifth copy** of
"single-threaded". Without this partial index the risk panel fills with duplicates
within a week.

### `recommendations`

| Column              | Type / reference                                              |
| --------------------- | ---------------------------------------------------------------- |
| `id`                | uuid, PK                                                        |
| `deal_id`           | → `deals`                                                       |
| `source_risk_id`    | → `risks` (nullable, `ON DELETE SET NULL`)                      |
| `title`             |                                                                 |
| `description`       |                                                                 |
| `rationale`         |                                                                 |
| `action_type`       |                                                                 |
| `priority`          |                                                                 |
| `confidence`        |                                                                 |
| `status`            | `∈ (suggested, accepted, dismissed, completed)`                 |
| `origin`            |                                                                 |
| `generated_at`      |                                                                 |
| `decided_at`        |                                                                 |
| `created_task_id`   | → `tasks` (nullable)                                            |
| `dismissal_reason`  | `∈ (already_handled, not_relevant, wrong, bad_timing, other)`   |
| `dismissal_note`    | `text`                                                          |
| `created_at`        |                                                                 |
| `updated_at`        |                                                                 |

```sql
CREATE UNIQUE INDEX uq_recommendations_deal_id_source_risk_id_suggested
  ON recommendations (deal_id, source_risk_id)
  WHERE status = 'suggested' AND source_risk_id IS NOT NULL;
```

**The proposal layer, distinct from `tasks`.** A recommendation is *suggested*
work; a task is *committed* work. `next_action` only ever comes from tasks —
from things a person agreed to — so the model can propose but never silently put
work on someone's list:

```
risk  →  recommendation  →  [human accepts]  →  task  →  next_action
```

`source_risk_id` was added in `0009_recommendation_provenance`. Before it the two
tables had no link at all — only shared `evidence` rows, which is a poor thing to
assemble a UI from, and the risk panel shows one card per risk with its suggested
action inline. Nullable because a *proactive* recommendation has no risk behind
it; `SET NULL` rather than `CASCADE` because an accepted recommendation and its
task are real work that should outlive the risk that prompted them.

**Dismissal is recorded, not deleted.** Without it the detector re-suggests the
same thing on every run. `dismissal_reason` is countable — "40% are `wrong`" says
the detector needs work, "40% are `already_handled`" says it is right but late —
while `dismissal_note` carries the detail a count cannot.

The partial unique index is keyed on the **risk**, not `action_type`: two risks
legitimately share an action type (`no_economic_buyer` and `single_threaded` both
map to `engage_stakeholder`), and keying on the action silently dropped the
second one's advice. Scoped to `suggested` so a dismissal does not block a fresh
suggestion later if circumstances change.

**`status='completed'` is never written.** It is derivable from
`created_task_id`'s task being done — the same rule that keeps `next_action` off
the `deals` table.

### The deterministic detector

Four of the ten `risk_type` values need no model at all — they are joins over
tables that already exist, so a cited risk panel works before any agent does:

| rule | evidence |
| ---- | -------- |
| `no_economic_buyer` — nobody who can say yes has *attended* | `record` → the listed-but-absent `deal_contacts` row, or `derived` when none is listed |
| `single_threaded` — one external attendee across all meetings | `derived` |
| `stalled_stage` — past the per-stage threshold in `queries.py` | `record` → newest `deal_stage_history.changed_at` |
| `close_date_at_risk` — close date near, stage not yet negotiation | `record` → `deals.expected_close_date` |

Each risk also produces one recommendation from a `RiskType → ActionType`
mapping; the two enums line up one-for-one by design. Detection **upserts** —
`uq_risks_deal_id_risk_type_open` exists so a re-run bumps `last_seen_at` rather
than adding a fifth "single-threaded" — and a risk that no longer detects is
auto-*resolved* rather than deleted, scoped to these four types so an
LLM-detected risk is not resolved by a pass that cannot see it.

### `meeting_briefs`

| Column                   | Type / reference |
| -------------------------- | ------------------ |
| `id`                     | uuid, PK          |
| `meeting_id`             | → `meetings`      |
| `objectives`             | `jsonb`           |
| `context_summary`        | `text`            |
| `key_risks`               | `jsonb`           |
| `recommended_questions`  | `jsonb`           |
| `model`                  |                   |
| `generated_at`           |                   |

Persisted so Meeting Prep survives a page reload instead of costing a
regeneration every time.

### `chat_sessions`

| Column           | Type / reference          |
| ------------------ | ---------------------------- |
| `id`             | uuid, PK                     |
| `scope`          | `∈ (global, deal)`           |
| `deal_id`        | nullable                     |
| `title`          |                              |
| `last_message_at`|                              |
| `created_at`     |                              |
| `updated_at`     |                              |

### `chat_messages`

| Column          | Type / reference                     |
| ----------------- | --------------------------------------- |
| `id`            | uuid, PK                               |
| `session_id`    | → `chat_sessions`                       |
| `role`          |                                        |
| `content`       | `text`                                 |
| `status`        | `∈ (streaming, complete, error)`        |
| `model`         |                                        |
| `token_usage`   | `jsonb`                                |
| `latency_ms`    |                                        |
| `created_at`    |                                        |

Citations reuse `claim_evidence` with `claim_type='chat_message'` — no separate
citations table. `status='streaming'` gives SSE a row to write into, so a refresh
mid-answer does not lose the turn.

**Every AI-written row carries `origin`, `confidence` and `status`.** Without
`origin` you can never answer "how many of these eight commitments did the model
write?", which is the first question anyone asks when they stop trusting it.

---

## 4.1 Why `claim_evidence` exists

Five tables hold things the AI *asserts* — `extracted_facts`, `commitments`,
`risks`, `recommendations`, `chat_messages`. Call each one a **claim**. Every
claim must point at the source material backing it.

A plain foreign key cannot express this, for two reasons:

1. **One claim usually rests on several pieces of evidence.** "This deal will
   slip" is supported by a quote, *plus* the close date, *plus* how long the
   stage has been stuck. A single `evidence_id` column holds one of those three.
2. **One piece of evidence supports several claims.** The same sentence in a
   transcript can yield a requirement, a commitment, and feed a risk.

Many-to-many on both sides → a join table.

```
extracted_facts ─┐
commitments     ─┤
risks           ─┼──(claim_type, claim_id)──> claim_evidence <──evidence_id── evidence
recommendations ─┤                                                              │
chat_messages   ─┘                                              ┌───────────────┴───────────────┐
                                                       source_kind='document'      source_kind='record'
                                                       chunk + char offsets        record_ref jsonb
                                                       (a quote)                   (a field in our own DB)
```

`claim_id` is a bare `uuid` with **no foreign key**, because the row it points at
lives in one of five tables; `claim_type` says which. This is a *polymorphic
association*.

**Queries.** Citations for one risk:

```sql
SELECT e.snippet, e.speaker, e.occurred_at, ce.relevance,
       e.document_id, e.chunk_id, e.record_ref
FROM claim_evidence ce
JOIN evidence e ON e.id = ce.evidence_id
WHERE ce.claim_type = 'risk' AND ce.claim_id = :risk_id
ORDER BY ce.relevance DESC;
```

The reverse lookup — *"what else came out of this sentence?"*:

```sql
SELECT claim_type, claim_id FROM claim_evidence WHERE evidence_id = :evidence_id;
```

`relevance` is how strongly *that* piece supports *that* claim. It is what lets
the UI sort citations so the strongest quote shows first and marginal ones
collapse behind "show 2 more".

**The cost being accepted.** Because `claim_id` has no FK, Postgres will not stop
you orphaning rows: delete a risk and its `claim_evidence` rows survive, pointing
at nothing. Mitigation for the MVP: delete links in the same service-layer
transaction that deletes the claim, and prefer soft-delete for dismissed risks
(which you want anyway). The `evidence_id` side *is* a real FK, so dropping a
document cascades cleanly.

**The alternative, honestly.** Five separate join tables (`risk_evidence`,
`fact_evidence`, …) each with two real foreign keys gives full referential
integrity and no orphans. The cost is five tables, five relationships, and an
Evidence Validator that must `UNION` five ways every time a claim type is added.
For an MVP where claim types are still moving, one polymorphic table wins.
Splitting it into five later is a mechanical migration.

---

## 5. How AI assertions get validated

`claim_evidence` does not make claims **true**. It makes them **checkable**, and
it separates two questions that are usually collapsed into one:

- **Grounding** — does the cited source exist, and does it actually say this?
  *Mechanically verifiable.*
- **Correctness** — is the assertion right about the world?
  *Only partly verifiable; ultimately a human call.*

Nearly every model failure in this product is a grounding failure. Four gates:

### Gate 0 — Span integrity (deterministic, no LLM)

Runs on every claim write, for each link in `claim_evidence`:

- `source_kind='document'`: does `chunk_id` exist, and does `evidence.snippet`
  occur **verbatim** in `document_chunks.content` at `[char_start, char_end]`?
- `source_kind='record'`: does `record_ref` resolve to a live row, and does the
  field still hold the value recorded in `snippet`?

Catches fabricated citations — the most common and most damaging failure — at
near-zero cost. Never ask a model to do this.

**The literal rule** belongs here too: *if the claim text contains a date or a
monetary amount, that literal must appear in a cited span or in a resolved record
value.* Models are worst at exactly these, and it is a string match.

Result is written to `claim_evidence.verification_status ∈ (unverified, verified,
span_missing, value_drifted, stale)`.

### Gate 1 — Entailment (the Evidence Validator agent)

Given the claim and **only** the cited spans:

| Verdict         | Meaning                                        | Handling |
| ----------------- | ------------------------------------------------- | ---------- |
| `supported`      | every atomic assertion is entailed              | surface normally |
| `partial`        | some entailed, others not                       | surface with a caution badge |
| `contradicted`   | evidence says the opposite                      | auto-quarantine, never render |
| `unsupported`    | evidence is real but does not address the claim | hide, log for eval |

**The critical constraint: the validator sees the evidence spans and nothing
else.** No transcript, no deal record, no extraction context. Given the source
material it will silently re-derive the claim and rubber-stamp it, and the gate
buys nothing. A separate call with a deliberately starved context is the point.

### Gate 2 — Staleness and contradiction

Compare `evidence.occurred_at` against the claim's age and the deal's
`last_activity_at`. Flag claims whose newest evidence predates the last customer
interaction. When new evidence contradicts a live claim, **do not overwrite** —
mark the old claim `superseded`, link the new evidence, keep both.

### Gate 3 — Human adjudication

Where *correctness* is actually resolved. Gates 0–2 decide what a person is
allowed to see; a person decides what is true. The fact-promotion step **is** this
gate: accepting an `extracted_facts` row into a `commitment` or `task` is the
human saying "yes, this is real", recorded via `promoted_to_type` /
`promoted_to_id`.

### `confidence` is not `verdict`

`confidence` is the **generator's self-report** — a weak, poorly-calibrated
signal. `verdict` is an **independent check**. Never render confidence as if it
were validation, and never let a high confidence score bypass a gate. Log the
pairs: high confidence + `contradicted` is the most valuable eval case there is.

`validator_version` on `claim_validations` matters more than it looks — after a
validator prompt change, every historical verdict came from a different judge.
Without the column the coverage metric silently mixes two populations.

### Write-time rules that do most of the work

1. **The extractor returns claims and their spans in one structured output.**
   Never generate a claim and then go looking for evidence — post-hoc retrieval
   always finds something plausible, which is how confident nonsense is
   manufactured.
2. **A claim with zero surviving links never reaches the UI.** Enforce in the
   service layer on insert. "No evidence" is a rejection, not a warning.
3. **Evidence and chunks are immutable** (see §3).

---

## 6. Worked scenario

Today is **2026-09-24**.

| Field                  | Value |
| ------------------------ | ------- |
| Deal                   | SecureFlow Enterprise — Northwind Logistics |
| `value`                | $180,000 |
| `stage`                | `discovery`, unchanged since 2026-07-28 (**58 days**) |
| `expected_close_date`  | 2026-10-15 (**21 days out**) |
| `deal_contacts`        | Priya Raman (Dir. IT Security, champion), Marcus Webb (Platform Lead, technical) |
| AE                     | Maya Chen — ~14 open deals, has not opened this one in nine days |

### 6.1 The transcript lands

Maya uploads Tuesday's call. → `documents` row, `source_type='meeting_transcript'`,
`occurred_at=2026-09-22T15:00Z`, `uploaded_at=2026-09-24T09:12Z`,
`ingest_status='pending'`, linked from `meetings.transcript_document_id`.

The worker parses, chunks and embeds. `ingest_status='ready'`, 34 chunks. The
relevant stretch is `chunk_index=11`:

> **Priya Raman:** Look, I'm sold on the product. But I can't start procurement
> until your SOC 2 Type II is in hand. Our auditors flagged three vendors last
> year and I'm not going through that again.
>
> **Tom Alvarez (SecureFlow SE):** Completely fair. I'll get you the SOC 2 Type II
> and the latest pen test summary by Friday.
>
> **Priya Raman:** Perfect. Once I've reviewed it, I can loop Dave in — he'd need
> to sign off on anything over 150.
>
> **Maya Chen:** Dave is…?
>
> **Priya Raman:** Dave Okonkwo, our CFO. And honestly we'd like something in
> place before our fiscal year close, so there's some urgency on my side too.

### 6.2 Evidence rows

| id | kind | points at | snippet |
| --- | --- | --- | --- |
| `e1` | document | chunk 11, chars 48–121 | *"I can't start procurement until your SOC 2 Type II is in hand"* — Priya Raman |
| `e2` | document | chunk 11, chars 203–282 | *"I'll get you the SOC 2 Type II and the latest pen test summary by Friday"* — Tom Alvarez |
| `e3` | document | chunk 11, chars 297–412 | *"Once I've reviewed it, I can loop Dave in — he'd need to sign off on anything over 150… Dave Okonkwo, our CFO"* — Priya Raman |
| `e4` | document | chunk 11, chars 455–521 | *"we'd like something in place before our fiscal year close"* — Priya Raman |

### 6.3 Six claims hit the gates

**C1 — requirement:** *"Northwind requires SOC 2 Type II delivery before
procurement can begin."* → cites `e1`
Gate 0 passes, span verbatim. Gate 1: `supported`. **Renders.**

**C2 — commitment (our side):** *"SecureFlow to deliver SOC 2 Type II and pen test
summary by Friday 2026-09-25."* → cites `e2`
Gate 0: span verbatim. Literal rule fires on the date — "2026-09-25" is not in the
span, but the resolver maps "by Friday" against `documents.occurred_at` (Tue Sep
22) → Sep 25, and records the resolution in `rationale`. Gate 1: `supported`.
**Renders.**

**C3 — stakeholder:** *"Dave Okonkwo, CFO, must approve purchases above
$180,000."* → cites `e3`
Gate 0: literal "150" is present, "180,000" is not → flagged, corrected to
$150,000. Gate 1 on the corrected claim: `supported`. **Renders.**

**C4 — commitment:** *"Priya confirmed Northwind will sign by October 15."* →
cites `e1`
Gate 0: **the span is real and quoted correctly.** A citation-only system ships
this. The literal rule fires: "October 15" appears in no cited span and no cited
record. Gate 1, seeing only `e1`, returns:

> `unsupported` — *"The cited span establishes a prerequisite for beginning
> procurement. It contains no signing commitment and no date."*

Written with `status='rejected'`. **Never renders.**

Where did "October 15" come from? `deals.expected_close_date`, sitting in the
extractor's context window. **The model laundered a CRM field into a customer
statement.** This is the most dangerous failure in the product — the hallucination
that would send Maya into a forecast call confident — and it is catchable *only*
because the validator was starved of everything except `e1`.

**C5 — commitment:** *"Priya will introduce SecureFlow to Dave Okonkwo once the
SOC 2 report is delivered."* → cites `e3`
Gate 0 passes. Gate 1:

> `partial` — *"Span supports that Priya can involve Dave. The trigger is her
> review of the report, not its delivery, and 'I can loop Dave in' expresses
> willingness, not a commitment."*

**Renders with a caution badge**, not auto-promoted. The gap between "delivered"
and "reviewed" is the gap between a Friday intro and a two-week wait.

**C6 — fact:** *"Northwind's fiscal year ends October 31."* → cites a span it
returned as *"our fiscal year ends October 31"*
Gate 0 fails instantly: that string does not occur in chunk 11.
`verification_status='span_missing'`, rejected **before any LLM is invoked**. The
model paraphrased a vague statement into a specific date and quoted its own
paraphrase back as the source. Caught by string comparison, at zero cost.

### 6.4 Promotion

Three clean items, one flagged, two invisible. Every rendered line opens the
transcript at its highlighted span.

Maya accepts C1 and C2 → a `commitments` row (`owner_side='us'`, due 2026-09-25)
and a `tasks` row, with `promoted_to_id` stamped back on each fact. They accept C3
→ Dave Okonkwo is created in `contacts` and attached via `deal_contacts` with
`buying_role='economic_buyer'`. They reword C5 to "after Priya reviews the report"
and accept it. **Three minutes, no typing.**

### 6.5 The risk engine cites the database

New evidence rows, no document involved:

| id | kind | `record_ref` | snippet |
| --- | --- | --- | --- |
| `r_e1` | record | `{"table":"deals","field":"expected_close_date"}` | "expected_close_date = 2026-10-15" |
| `r_e2` | record | `{"table":"deal_stage_history","id":"h9"}` | "stage = discovery since 2026-07-28 (58 days)" |
| `r_e3` | record | `{"table":"deal_contacts","field":"buying_role"}` | "no contact with buying_role='economic_buyer' has attended a meeting" |

**Risk R1** — `close_date_at_risk`, severity `high`:

> *"Close date is 21 days out, but procurement cannot begin until the SOC 2 report
> is delivered and reviewed, the deal has been in discovery for 58 days, and the
> economic buyer has never been in a meeting."*

Linked to **`e1`, `r_e1`, `r_e2`, `r_e3`** — one quote and three facts from Maya's
own data. Verdict `supported`.

`e1` now carries two links: it backs both C1 and R1. Stored once, cited twice —
the many-to-many doing its job.

Three of the four citations are structured. Had evidence been restricted to
document chunks, this risk would render with a single quote and read as a hunch.
Instead it reads as arithmetic, which is why Maya acts on it.

### 6.6 Recommendation

> **Get Dave Okonkwo into a meeting this week.** — priority `high`
> *Rationale:* procurement is gated on a document we owe Friday, and the only
> person who can approve $180k has never spoken to us, with 21 days on the clock.

Cites R1's evidence set plus the Friday commitment. Maya accepts → a `tasks` row,
linked via `created_task_id`.

### 6.7 What happens next

**Sep 25 passes**, nobody sent the report. The commitment flips to `missed` and
surfaces in overdue actions — still carrying `e2`, so Maya can see it was *Tom*
who promised it, on the call, in his own words. Not an accusation; a retrievable
fact.

**Oct 6, next call.** Priya: *"We've pushed the security review to next quarter."*
New evidence `e9`. The validator re-runs C1's cluster and returns `contradicted`
against the Oct 15 close date. Nothing is overwritten: R1's `last_seen_at` is
bumped and severity raised to `critical`, the old claim is marked `superseded`
with `e9` linked, both retained.

Six weeks later, when Maya's manager asks why this slipped, the answer is a query,
not a memory: *the SOC 2 was promised Sep 22, never delivered, procurement never
opened, and the CFO was identified on Sep 22 but never met.*

### 6.8 What generalizes

Of six model outputs, **two were wrong and both were caught** — one by a string
comparison costing nothing, one by a deliberately blindfolded second model. A
third was *subtly* wrong and got flagged rather than silently promoted.

None of that comes from the model being good. It comes from three structural
choices: **claims arrive with spans attached**, **the validator sees only those
spans**, and **evidence can point at our own database as readily as at a
transcript**.

### 6.9 The ceiling

Evidence chains to the source, not to reality. A claim well-grounded in a source
that was itself wrong — a customer misstating their own budget, a transcript
mishearing a name — passes every gate. That is the right ceiling for this
product. The honest promise to an AE is *"every statement here traces to something
someone actually said or something actually in your data"* — not *"every statement
here is true."*

---

## 7. pgvector

| Concern      | Detail |
| -------------- | -------- |
| **Image**    | `postgres:16` ships no vector extension. Use `pgvector/pgvector:pg16`, which is the stock Postgres image plus the extension binaries. The data volume is compatible; no dump/restore needed. |
| **Extension** | `CREATE EXTENSION IF NOT EXISTS vector;` in the first migration, before any table using the type. |
| **Python**   | `pgvector` package → `from pgvector.sqlalchemy import Vector`. Alembic autogenerate emits `Vector(1536)` **without** adding the import to the generated migration — add `import pgvector.sqlalchemy` to any migration touching `document_chunks`, or the upgrade fails with `NameError`. |
| **Dimension** | Pinned in settings as `EMBEDDING_DIM` (1536 for `text-embedding-3-small`). The column type bakes the number in, so changing models later means a new column and a re-embed — assert the configured dimension against the column at startup so a mismatch fails loudly rather than at query time. |
| **Query shape** | Always filter by `deal_id` *before* the vector search for deal-scoped chat; a join to `documents` with a `WHERE deal_id = :id` ahead of the `ORDER BY embedding <=> :q LIMIT k` keeps one deal's chat from retrieving another's transcript. |

**Index**

```sql
CREATE INDEX ix_document_chunks_embedding ON document_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

HNSW requires pgvector ≥ 0.5.0. Cosine distance (`<=>`) with normalized
embeddings. At MVP volumes the index is not yet load-bearing — a sequential scan
over a few thousand chunks is fine — but creating it now means the query operator
never changes later.

---

## 8. Migration order

Existing head: `b3f2c4a1d9e7`. `accounts` and `deals` already exist, so M1
**alters** them rather than creating them.

| # | Contents | Notes |
| --- | ---------- | ------- |
| M1 | `CREATE EXTENSION vector`; alter `accounts`, `deals`; create `contacts`, `deal_contacts`, `deal_stage_history` | `deals.close_date` → `expected_close_date`; add `currency`, `win_probability`, `closed_at`; drop `next_action`, `next_action_due_date` |
| M2 | `meetings`, `meeting_attendees`, `tasks`, `activities` | `meetings.transcript_document_id` FK added in M3 |
| M3 | `documents`, `document_chunks` + HNSW index | needs `import pgvector.sqlalchemy` |
| M4 | `evidence`, `claim_evidence`, `claim_validations` | polymorphic indexes on `(claim_type, claim_id)` |
| M5 | `extracted_facts`, `commitments`, `risks`, `recommendations` | partial unique index on open risks |
| M6 | `chat_sessions`, `chat_messages`, `meeting_briefs` | |

## 9. Indexes beyond the foreign keys

| Table | Index |
| ------- | ------- |
| `deals`             | `(stage, expected_close_date)` |
| `tasks`             | `(status, due_date) WHERE status = 'open'` |
| `documents`         | `(deal_id, occurred_at DESC)` |
| `documents`         | `(content_hash)` |
| `document_chunks`   | `USING hnsw (embedding vector_cosine_ops)` |
| `evidence`          | `(deal_id)` |
| `claim_evidence`    | `(claim_type, claim_id)` |
| `claim_evidence`    | `(evidence_id)` |
| `claim_validations` | `(claim_type, claim_id, checked_at DESC)` |
| `risks`             | `(deal_id, risk_type) UNIQUE WHERE status = 'open'` |

## 10. Deferred

| Item | Reason |
| ------ | -------- |
| CRM suggestion / approval tables | Not needed for MVP |
| `ai_runs`                        | Langfuse covers tracing |
| Document versioning              | Not needed for MVP |
| Org and user tables              | No auth / multi-tenant in MVP |
| Data seeding                     | The seeder writes Layer A plus `documents.raw_text` only, never Layer C — seeding fabricated risks and commitments directly would destroy the ability to tell whether extraction actually works |
