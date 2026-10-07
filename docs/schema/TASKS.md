# Schema Implementation — Task List

Scope: **schema layer only.** Models, migrations, pgvector setup. No seed data,
no API routes, no agents. Design reference: `README.md` in this directory.

**Status: Phases 0–7 complete, Phase 8 partly.** The old migration chain was
deleted and the database rebuilt from empty, so the planned M1–M6 split
collapsed into two revisions — there was nothing left to migrate *from*:

| Revision | Contents |
|---|---|
| `0001_pgvector` | `CREATE EXTENSION vector` -- kept separate because it is a database-level concern that must land before any `vector` column, and it is the step that fails loudly on the wrong Postgres image |
| `0002_schema` | All 21 tables, 26 enum types, every index and CHECK |

The phase headings below are kept as the record of what each covers.

---

## Phase 0 — Infrastructure

- [x] **0.1 Swap the Postgres image to pgvector.**
      `docker-compose.yml`: `postgres:16` → `pgvector/pgvector:pg16`.
      Same upstream Postgres 16 plus the extension binaries — the existing
      `dealpilot_postgres_data` volume is compatible, no dump/restore.
      *Done when:* `docker compose up -d postgres` is healthy and
      `SELECT * FROM pg_available_extensions WHERE name='vector';` returns a row.

- [x] **0.2 Add the Python dependencies.**
      `backend/requirements.txt`: `pgvector==0.3.6`.
      *Done when:* `from pgvector.sqlalchemy import Vector` imports in the venv.

- [x] **0.3 Add embedding settings.**
      `app/core/config.py`: `embedding_dim: int = 1536`,
      `embedding_model: str = "text-embedding-3-small"`. Mirror into
      `backend/.env.example`.
      *Done when:* both read from the environment.

- [x] **0.4 (Cleanup) Drop the dead JWT settings.**
      `app/core/config.py` and `backend/.env.example` still carry
      `jwt_secret_key`, `refresh_token_expire_days`, `cookie_secure` — leftovers
      from the `users` tables that migration `b3f2c4a1d9e7` already dropped. The
      MVP has no auth. Independent of everything below.

---

## Phase 1 — Model plumbing

Do this **before** writing any new model — it changes how every subsequent
migration is generated.

- [x] **1.1 Add a metadata naming convention.**
      `app/db/base.py`: give `Base.metadata` a `naming_convention` for
      `ix/uq/ck/fk/pk`. Without it Alembic emits unnamed constraints that cannot
      be dropped by name in a downgrade.
      *Note:* the constraints already created by the three existing migrations
      keep their auto-generated names; the convention applies to new ones only.
      *Done when:* a scratch autogenerate shows named constraints.

- [x] **1.2 Add timestamp mixins.**
      `app/db/mixins.py` — `TimestampMixin` (`created_at`, `updated_at`) and
      `UUIDPrimaryKeyMixin` (`id` with `server_default=text("gen_random_uuid()")`).
      `gen_random_uuid()` is native in PG13+; no `pgcrypto` needed.
      *Note:* the existing `Account`/`Deal` models set `default=uuid.uuid4`
      Python-side. Move them to the server default for consistency.

- [x] **1.3 Centralize the enums.**
      `app/models/enums.py`. Native PG enums for the stable sets
      (`deal_stage`, `risk_level`, `severity`, `verdict`, `source_kind`,
      `claim_type`); `str`-valued Python enums backed by `text` + `CHECK` for the
      churny sets (`risk_type`, `fact_type`, `action_type`, `activity_type`).
      Rationale in README §1.

- [x] **1.4 Turn on `compare_type` in Alembic.**
      `alembic/env.py`: pass `compare_type=True` (and `compare_server_default=True`)
      to both `context.configure` calls, so a changed column type is actually
      detected by autogenerate.

- [x] **1.5 Guard the model registry.**
      `alembic/env.py` does `from app.models import *`, which respects
      `app/models/__init__.py`'s `__all__`. **A model missing from `__all__` is
      invisible to autogenerate and its table silently never gets created.**
      Every task below ends with an `__init__.py` update. Consider a test that
      asserts every `Base.metadata.tables` key has a matching export.

---

## Phase 2 — M1: Deal domain core

- [x] **2.1 Create the extension.** Hand-written migration step:
      `op.execute("CREATE EXTENSION IF NOT EXISTS vector")`. First, before any
      table needing it.

- [x] **2.2 Alter `accounts`.** Add `employee_band`, `hq_region`.

- [x] **2.3 Alter `deals`.**
      - rename `close_date` → `expected_close_date`
      - add `currency char(3)` default `'USD'`, `win_probability smallint`, `closed_at`
      - drop `next_action` (derived from `tasks`; see README §2)
      - keep `stage`, drop nothing else — there is no `status` column to add

- [x] **2.4 `contacts`** — `account_id` FK, not `deal_id`.

- [x] **2.5 `deal_contacts`** — composite PK `(deal_id, contact_id)`,
      `buying_role`, `influence`, `sentiment`, `is_primary`.

- [x] **2.6 `deal_stage_history`** — surrogate `id`, `from_stage`/`to_stage`.

- [x] **2.7 Index** `deals (stage, expected_close_date)`.

- [x] **2.8 Generate + review the migration.** Autogenerate will propose a
      drop-and-add for the `close_date` rename — **replace it with
      `op.alter_column(..., new_column_name=...)`** by hand, or the column's data
      is discarded.

---

## Phase 3 — M2: Activity domain

- [x] **3.1 `meetings`** — leave `transcript_document_id` out for now; the
      `documents` table does not exist until M3. Add the column and its FK in M3.
- [x] **3.2 `meeting_attendees`** — `contact_id` **nullable**, `raw_name` text.
- [x] **3.3 `tasks`** — `status`, `priority`, `origin`; `source_fact_id` left
      nullable with no FK until M5.
- [x] **3.4 `activities`** — append-only, no `updated_at`.
- [x] **3.5 Index** `tasks (status, due_date) WHERE status = 'open'`
      (`postgresql_where=`).

---

## Phase 4 — M3: Documents and vectors

- [x] **4.1 `documents`** — one table for all unstructured input.
      `occurred_at` and `uploaded_at` are **separate columns**, both required.
      Unique index on `content_hash`.
- [x] **4.2 `document_chunks`** — `embedding Vector(settings.embedding_dim)`,
      `metadata jsonb`, `UNIQUE(document_id, chunk_index)`,
      `ON DELETE CASCADE` from documents.
      *Careful:* `metadata` is reserved on SQLAlchemy declarative classes — map
      the attribute as `chunk_metadata` with `name="metadata"`.
- [x] **4.3 Back-fill `meetings.transcript_document_id`** + its FK.
- [x] **4.4 HNSW index.**
      ```sql
      CREATE INDEX ix_document_chunks_embedding ON document_chunks
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
      ```
- [x] **4.5 Add `import pgvector.sqlalchemy` to the generated migration** —
      autogenerate emits `Vector(1536)` without the import and the upgrade dies
      with `NameError`. This bites every time; check it explicitly.
- [x] **4.6 Startup assertion** — compare `settings.embedding_dim` against the
      actual column dimension so a model swap fails loudly at boot rather than
      silently at query time.

---

## Phase 5 — M4: Evidence spine

- [x] **5.1 `evidence`** — `source_kind`, nullable `document_id`/`chunk_id`,
      `record_ref jsonb`, `snippet`, `char_start`/`char_end`, `speaker`,
      `occurred_at`. A `CHECK` that `source_kind='document'` implies `chunk_id IS
      NOT NULL`, and `source_kind='record'` implies `record_ref IS NOT NULL`.
- [x] **5.2 `claim_evidence`** — `(claim_type, claim_id)` polymorphic pair with
      **no FK on `claim_id`** (deliberate; README §4.1),
      `UNIQUE(claim_type, claim_id, evidence_id)`, `verification_status`,
      `verified_at`, `relevance`.
- [x] **5.3 `claim_validations`** — append-only, `verdict`, `method`,
      `rationale`, `model`, `validator_version`, `checked_at`.
- [x] **5.4 Indexes** `claim_evidence (claim_type, claim_id)`,
      `claim_evidence (evidence_id)`,
      `claim_validations (claim_type, claim_id, checked_at DESC)`.
- [x] **5.5 Document the orphan contract.** `claim_id` has no FK, so deletes must
      be handled in the service layer. Write it as a docstring on the model now,
      while the reason is fresh, not as tribal knowledge later.

---

## Phase 6 — M5: AI assertions

- [x] **6.1 `extracted_facts`** — `fact_type`, `content`, `payload jsonb`,
      `confidence`, `status`, `promoted_to_type`/`promoted_to_id`.
- [x] **6.2 `commitments`** — `owner_side`, nullable `owner_contact_id`,
      `owner_name`, `due_date`, `status`, `origin`, `confidence`.
- [x] **6.3 `risks`** — plus the partial unique index:
      ```sql
      CREATE UNIQUE INDEX uq_risks_open_type ON risks (deal_id, risk_type)
        WHERE status = 'open';
      ```
- [x] **6.4 `recommendations`** — `created_task_id` FK → `tasks`.
- [x] **6.5 Add the `tasks.source_fact_id` FK** deferred from M2.
- [x] **6.6 Confirm `origin`, `confidence`, `status` are on all four tables.**

---

## Phase 7 — M6: Chat and briefs

- [x] **7.1 `chat_sessions`** — `scope`, nullable `deal_id`, `last_message_at`.
      `CHECK (scope = 'deal') = (deal_id IS NOT NULL)`.
- [x] **7.2 `chat_messages`** — `role`, `content`, `status` (incl. `streaming`),
      `token_usage jsonb`, `latency_ms`.
- [x] **7.3 `meeting_briefs`** — one per meeting, jsonb payloads.
- [x] **7.4 No separate citations table** — chat citations use `claim_evidence`
      with `claim_type='chat_message'`.

---

## Phase 8 — Verification

- [x] **8.1 Round-trip every migration.** `alembic upgrade head` then
      `alembic downgrade base` then `upgrade head` again on a scratch database.
      Downgrades are where hand-edited migrations break.
- [x] **8.2 Autogenerate must come back empty.** `alembic revision --autogenerate`
      against a fully migrated database should produce a no-op revision. Anything
      it proposes is model/migration drift — fix it now, not after data exists.
- [x] **8.3 pgvector smoke test.** Insert a chunk with a random 1536-vector, run
      `ORDER BY embedding <=> :q LIMIT 5`, confirm `EXPLAIN` uses the HNSW index
      once past a few hundred rows.
- [ ] **8.4 Constraint tests** (pytest, no fixtures/seed — construct inline):
      - a `risks` duplicate with `status='open'` is rejected; with
        `status='resolved'` it is accepted
      - `evidence` with `source_kind='document'` and null `chunk_id` is rejected
      - a duplicate `claim_evidence` triple is rejected
      - deleting a `documents` row cascades to its chunks
- [ ] **8.5 Model-registry test.** Every table in `Base.metadata.tables` has a
      matching export in `app/models/__init__.py` (guards the Phase 1.5 trap).
- [ ] **8.6 Check the ERD in pgAdmin** — already wired up at `localhost:5051`.

---

## Out of scope

Seed/synthetic data · ingest worker · LangGraph agents · CRM suggestion
tables · `ai_runs` · Langfuse wiring.

**API routes are no longer out of scope.** 48 endpoints are built and mounted
under `/api/v1`, covering Layer A (deals, stakeholders, stage history, tasks,
meetings, attendees), Layer B (documents, chunk fetch) and the built half of
Layer C (risks, recommendations, commitments, and the deterministic detector).
A `services/` layer holds the multi-statement writes. See `docs/api/README.md`.

Migrations `0004`–`0009` came out of that work; each is documented in its own
docstring and summarised in the root `README.md`.

Still out of scope: chat, the extraction agent, the Evidence Validator
(Gates 1–2), `meeting_briefs`, and the seeder.

When seeding does start, it writes **Layer A + `documents.raw_text` only** —
never `extracted_facts`, `commitments`, `risks` or `recommendations`. Those must
come from the pipeline, or there is no way to tell whether extraction works.

---

## Found during implementation

Two traps that were not in the original plan, both now fixed in the code:

**Alembic leaves native enum types behind on downgrade.** `op.drop_table()`
drops the table but not the `CREATE TYPE` that the table's enum columns
implicitly created. After `downgrade base`, all 26 types survived, and the next
`upgrade head` died with `type "claim_type" already exists`. Fixed with an
explicit `ENUM_TYPES` list and a `DROP TYPE IF EXISTS` loop at the end of
`0002_schema`'s `downgrade()`. **Any future migration that adds a native enum
must add the matching drop** — autogenerate will not do it.

**The host virtualenv is Python 3.9, so the codebase must stay 3.9-compatible.**
`deal-pilot-env` is 3.9.6 while the container is 3.12. `Mapped[str | None]`
(PEP 604) is 3.10+ syntax that SQLAlchemy evaluates at class-creation time, so
every model raised `TypeError` on import in the venv. `from __future__ import
annotations` does **not** rescue it — SQLAlchemy still fails with
`MappedAnnotationError`. All 92 unions were rewritten to `Optional[...]`, which
runs identically on both interpreters.

**Keep new code 3.9-safe** (`Optional[X]`, `List[X]`, not `X | None`) for as
long as the venv is 3.9, or install Python 3.10+ and rebuild the venv to match
the container — the host has only system 3.9.6, no pyenv/uv/homebrew Python.
Matching the container is the better long-term fix; a version split between dev
and runtime invites divergence that only shows up in one of the two.

**`backend/.env` pointed at port 5433 but compose publishes 5432**
(root `.env` sets `POSTGRES_HOST_PORT=5432`). Only host-venv runs hit this,
since compose overrides `DATABASE_URL` for the container. Corrected to 5432.

## Remaining

Tasks 8.4–8.6. The constraints, cascades, round trip, drift check and HNSW
index usage were all verified by hand against the live database and pass; what
is missing is the **pytest harness** to keep them verified — which needs
`pytest`, `pytest-asyncio` and a throwaway test database, and is arguably the
first task of the API layer rather than the last of the schema layer.
