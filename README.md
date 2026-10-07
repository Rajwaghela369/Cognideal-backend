# CogniDeal Backend

The API and background worker for CogniDeal, an evidence-grounded sales
copilot. Nothing the system asserts about a deal may exist without a link back
to the record or transcript span that backs it.

The frontend lives in its own repository, **cognideal-frontend**.

| | |
| --- | --- |
| API | FastAPI, SQLAlchemy 2 (async), Alembic |
| Database | Postgres 16 with pgvector and pg_trgm (Neon in production) |
| Object storage | Any S3-compatible store (Neon in production) |
| AI | LangChain / LangGraph on OpenAI (gpt-4o, gpt-4o-mini) |

## Layout

```
app/
├── main.py           app factory, CORS, /health, mounts /api/v1
├── core/config.py    every setting, read from the environment
├── db/               engine, session, base, mixins
├── models/           SQLAlchemy tables
├── queries.py        reusable SQL
├── services/         multi-statement writes with ordering rules
├── schemas/          Pydantic -- the HTTP contract
├── api/v1/routes/    route handlers
└── ai/               pipeline, gates, chat agent
alembic/              migrations
seed/                 demo data loader (python -m seed)
worker.py             background worker (python -m worker)
tests/
docs/                 schema, api and ai references
```

Dependencies point one way: `db -> models -> queries -> services -> routes`.

## Local setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in DATABASE_URL and S3_* values
.venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --reload --port 8000
```

In a second terminal, the worker (consumes queued analyses):

```bash
.venv/bin/python -m worker
```

Swagger UI is at http://localhost:8000/docs.

## Configuration

All settings are environment variables; `.env.example` lists them. The ones
that must be set outside local development:

| Variable | Value |
| --- | --- |
| `DATABASE_URL` | Neon connection string, pasted unchanged (pooled host) |
| `S3_ENDPOINT` | Neon storage endpoint, `https://...` |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | Neon storage credentials |
| `S3_REGION` | e.g. `eu-central-1` |
| `S3_BUCKET` | bucket created in the Neon console |
| `CORS_ORIGINS` | the frontend's URL |
| `DEBUG` | `false` |
| `AI_ENABLED` / `OPENAI_API_KEY` | to turn the AI layer on |
| `AI_TOKENS_PER_MINUTE` | your OpenAI tier's gpt-4o limit (default 30000) |

`DATABASE_URL` is converted for both Postgres drivers the app uses (asyncpg
for the API, psycopg for chat memory), so Neon's `sslmode` and
`channel_binding` parameters are fine as-is.

## Demo data

```bash
.venv/bin/python -m seed             # accounts, deals, meetings, tasks, transcripts
.venv/bin/python -m seed --analyze   # also queue AI analysis
```

Re-runnable: rows are matched by natural key and only added when missing.

## Deploying on Render

| Service | Type | Build | Start |
| --- | --- | --- | --- |
| API | Web Service (Python) | `pip install -r requirements.txt` | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Worker | Background Worker (Python) | `pip install -r requirements.txt` | `python -m worker` |

- Pre-deploy command on the API: `alembic upgrade head`.
- Health check path: `/health`.
- Put the variables above in one Environment Group shared by both services.
- Set `PYTHON_VERSION` to the version the code is tested on.

## Tests

```bash
.venv/bin/pytest
```

## Migrations

```bash
.venv/bin/alembic upgrade head
.venv/bin/alembic downgrade -1
```

Full history and the traps to avoid: [`docs/schema/README.md`](docs/schema/README.md)
and [`docs/schema/TASKS.md`](docs/schema/TASKS.md). HTTP conventions:
[`docs/api/README.md`](docs/api/README.md). Model layer:
[`docs/ai/README.md`](docs/ai/README.md).
