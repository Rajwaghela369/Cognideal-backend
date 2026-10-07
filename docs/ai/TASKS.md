# AI Layer Implementation — Task List

Scope: **the model layer only.** Design reference: `README.md` in this
directory. Schema reference: `docs/schema/README.md`. Existing migration head:
`0009`.

**Status: Phases 0-10 complete.** All thirteen pipeline stages are implemented
— `stage_registry.STAGES` has no `_todo` left — the worker runs the graph, and
the suite is provably offline.

Migrations: head at `0016`. `0010_pg_trgm`, `0011_meeting_analysis_origin`,
`0012_detector_provenance`, `0013_open_risk_taxonomy`,
`0014_analysis_triggers`, `0015_commitment_correction`, all round-tripped.
`0016_fact_content_trgm` adds the GIN trigram index supersession selects
against -- see *Found during implementation*.

**Phase 11: 11.1–11.9 complete, 11.10 and 11.11 open.** Every capability item
has shipped; what remains is the two measurement tasks, and they are open for a
reason that is itself the finding — **the fixture corpus cannot resolve the
effects they were written to measure.** Five runs of one unchanged prompt span
f1 0.51–0.58, so a one-run prompt or model comparison ranks noise. Both now
have a working harness (`--repeat N`), a stated cost (~25 min and ~40 min at
8,000 TPM), and a recorded reason for not having been run. See each task and
*Found during implementation*.

Task 0.9 was declined, not deferred, and re-confirmed 2026-10-05. Task 1.5 is
not separately open: 11.11 **is** its outstanding half, and closing 11.11
closes it.

Corrections to an earlier version of this header: 7.4's shadow run **was** run
and recorded (recall 1.00, Jaccard 0.67 over two runs), and 7.5's writes
shipped with it. Task 11.9 enlarged that sample to n=10 and the figure is
**0.882 mean pairwise Jaccard with zero verdict flicker over twenty runs**;
7.4's 0.67 was a two-run artefact. Getting there uncovered a defect that was
failing 3 detect runs in 10, now fixed.

The payoff from Phase 2 is visible: with `meeting_attendees` populated, the
deterministic detector now returns three real cited risks on the fixture deal
(`no_economic_buyer`, `stalled_stage`, `close_date_at_risk`) and correctly
stays silent on `single_threaded`.

Files added by Phase 0: `app/ai/{client,governor,pipeline}.py`,
`app/ai/prompts/{__init__,smoke}.py`, `app/services/activity.py`,
`backend/worker.py`, `backend/tests/verify_phase0{,_http}.py`, and a `worker`
compose service.

**32 acceptance checks pass** — 15 in `verify_phase0.py` (no server needed) and
17 in `verify_phase0_http.py` (route -> service -> worker end to end, including
the hard-kill crash test and the `stale_days` filter in both directions).

Dependency order — each phase depends on the one above it, except Phase 8 and
Phase 9 which depend only on Phases 0–5:

```
0  Foundations ......... worker, client, provenance, versioning
1  Eval harness ........ fixtures + labels + metrics        <- no model calls yet
2  Transcript + roster .. meeting_attendees                 <- unblocks the 4 SQL risks
3  Extraction + Gate 0 .. extracted_facts, evidence
4  Gate 1 ............... claim_validations
5  Meeting synthesis .... meetings.summary / .sentiment      (needs 0010)
6  Reconciliation ....... commitments, supersession
7  Risk detection ....... risks, recommendations             (needs 0011, 0012)
8  Meeting briefs ....... meeting_briefs
9  Chat ................. chat_sessions, chat_messages
10 Triggers ............. event invalidation + the nightly sweep    needs 0013
11 Closing the loop .... last_activity_at, proposals, read surfaces (needs 0015)
```

Phase 10 is the exception to the order above: **10.1–10.4 can land as soon as
Phase 3 is done** (tier 0 needs Gate 0 to exist, tier 1 needs only the existing
`detect.py`), while 10.5 needs Phase 7. Doing tiers 0 and 1 early is worth it —
tier 0 is what makes the schema's self-invalidation promise real, and nothing
performs it today.

---

## Phase 0 — Foundations

No model calls. Everything here is testable without an API key.

- [x] **0.1 Add the dependencies.** `backend/requirements.txt`: `langchain`,
      `langchain-groq`, `langgraph`. Pin them. No checkpointer package — see
      `README.md` §7.
      *Done when:* `from langchain_groq import ChatGroq` and
      `from langgraph.graph import StateGraph` both import in the venv.

- [x] **0.2 Add AI settings.** `app/core/config.py`, mirrored into
      `backend/.env.example`: `groq_api_key`, `model_primary =
      "openai/gpt-oss-120b"`, `model_cheap = "openai/gpt-oss-20b"`,
      `model_challenger = "qwen/qwen3.8-27b"` (evals only),
      `ai_enabled: bool = False`, per-stage `max_tokens` and a token ceiling.
      *Done when:* the app starts with `ai_enabled=False` and no key present.

- [x] **0.3 One LLM client module.** `app/ai/client.py` — the only place a
      model is named. Builds a configured `ChatGroq` runnable per role, wires
      `.with_structured_output()` with `strict: true`, and captures usage
      (`input_tokens`, `output_tokens`, `latency_ms`). Every task calls through
      it; no task constructs a client.
      *Done when:* a smoke test returns a parsed Pydantic object and a
      populated usage record, and grepping for a model id finds only this file
      and `config.py`.
      **Built and verified offline** — imports with no provider installed,
      `AIDisabled` when `ai_enabled=False`, `RunBudget` ceiling enforced,
      retry classification correct, and the grep check passes.
      **Live call verified** — `tests/verify_ai_smoke.py`:
      `openai/gpt-oss-120b` returned a parsed Pydantic object with a populated
      usage record (226 in / 57 out, 443 ms, one attempt). `json_schema`
      structured output works on this account **even without `strict`**, so
      best-effort mode is adequate in practice — the Pydantic validation is
      still the only guarantee on this interpreter.

- [x] **0.4 Prompt registry with versions.** `app/ai/prompts/` — one module per
      task, each exporting `PROMPT` and `VERSION`. The version string is what
      lands in `claim_validations.validator_version` and (after 0011)
      `risks.detector_version`. A prompt change without a version bump is a bug.
      *Done when:* every prompt is importable by name and carries a version.

- [x] **0.5 The worker.** `backend/worker.py` — poll
      `meetings WHERE analysis_status='queued' ... FOR UPDATE SKIP LOCKED
      LIMIT 1`, run the pipeline, move the status. Write the loop so it can
      carry more than one poll query: Phase 10 adds dirty deals and the nightly
      sweep to the same process, and there is to be no scheduler. Add it to
      `docker-compose.yml` as a separate service sharing the backend image.
      `failed` means stages 0–4 failed; a later stage failure still yields
      `complete` with the stage error logged.
      *Done when:* `POST .../meetings/{id}/analysis` reaches `complete` with a
      no-op pipeline, and killing the worker mid-run leaves the row `queued`.

- [x] **0.6 Write `deals.last_activity_at`.** `services/ingest.py` and
      `services/meeting.py` touch it in the same transaction as the write.
      Blocking for Gate 2 and for the existing `stale_days` filter, which
      currently reads a column nothing sets.
      *Done when:* uploading a document and completing a meeting both advance
      it, and `stale_days` returns the expected deals.

- [x] **0.7 Tracing.** Langfuse (per `docs/schema/README.md` §10, which is why
      there is no `ai_runs` table) or structured logging of
      `(task, model, version, usage, latency, outcome)` from `client.py`.
      *Done when:* one analysis run is inspectable end to end.

- [x] **0.8 Rate-limit governor.** The Developer plan gives 250K TPM / 1K RPM,
      and the parallel extraction fan-out is what hits it first. `client.py`
      needs a global concurrency semaphore plus a token bucket estimated from
      request size, and 429 handling with backoff — not a bare retry.
      *Done when:* firing 50 extraction calls at once completes without a 429
      reaching the caller, and the concurrency ceiling is configurable.

- [x] **0.9 Resolve the Python version.** ~~Rebuild the venv on 3.12.~~
      **Declined — staying on 3.9, deps left unpinned.** Recorded because it
      has a standing consequence, not because it is still planned.
      Re-confirmed 2026-10-05: staying on the current version is the
      preference, so this is settled rather than pending. Checked off to stop
      it reading as outstanding work; the consequences below still apply.

      `langchain`, `langchain-core`, `langchain-groq` and `langgraph` all
      require `>=3.10`, so from one unpinned `requirements.txt` pip resolves
      **LangChain 0.x on the 3.9 host venv and 1.x in the 3.12 container**.
      The container is therefore the runtime of record for anything importing
      LangChain; the host venv stays useful for Alembic, the existing API, and
      every part of the AI layer that does not import a provider — which is why
      `governor.py` has no LangChain import and `client.py` imports lazily.

      Two places this leaks: `structured_output_method` is a setting because
      `json_schema` postdates the 0.x line, and `_is_retryable` matches on text
      rather than exception classes for the same reason.

      *Revisit when:* a stage behaves differently on host and container, or
      `pip install` on 3.9 stops resolving any compatible version at all.

---

## Phase 1 — Eval harness

Before the extractor, not after. This is the largest debt in the project and
everything downstream is unmeasurable without it.

- [x] **1.1 Three fixture transcripts.** `backend/tests/fixtures/transcripts/`.
      Realistic speaker-labelled text: a discovery call, a security review, a
      negotiation check-in. One should contain a *contradiction* of an earlier
      fact and one an *unfulfilled commitment*, so Phase 6 has something to
      find.
      *Done when:* each ingests through the real upload path and produces chunks.
      **Done** — one deal, three meetings, July/August/September, 8.5KB total →
      4 + 4 + 3 chunks. `manifest.json` records the eight planted cases and the
      expected rosters; `load.py` is idempotent (it deletes the account first,
      because `documents.content_hash` is globally unique and a re-run would
      otherwise 200 and attach chunks to the previous deal).

- [x] **1.2 Hand-labelled expectations.** A JSON file per transcript: expected
      facts with `fact_type`, content, and the exact `char_start`/`char_end` of
      the span that backs each one.
      *Done when:* the labels are machine-readable and offsets resolve verbatim
      against the ingested chunks.
      **Done** — 26 facts across all eight `fact_type` values; 26/26 resolve,
      0 straddling a chunk boundary. Snippets are authored by hand and offsets
      are *derived* by `resolve_labels.py`: `split_into_chunks` decides the
      boundaries, so a hand-written offset is a guess about an implementation
      detail, and Gate 0 compares against a chunk rather than the file.

- [x] **1.3 Metric functions.** `backend/tests/eval/metrics.py` — Gate 0 pass
      rate, fact precision/recall against labels, verdict distribution,
      `confidence × verdict` matrix, risk-key Jaccard across two runs.
      *Done when:* each runs against a recorded output file with no API call.
      **Done** — `tests/eval/metrics.py` plus 13 tests scoring a hand-built
      recording with exact expected numbers, so a scoring bug cannot look like a
      model regression. The matching rule is **(fact_type, span overlap)**,
      never text equality: the model's phrasing varies legitimately between
      runs, the span it cites should not.
      Also absorbed the Phase 0 database checks — `test_worker.py` and
      `test_activity.py`. **28 tests pass**; the subprocess crash test and the
      HTTP round trip stay in `verify_phase0_http.py`.

- [x] **1.4 The missing model-registry test.** `models/__init__.py` already
      references `tests/test_model_registry.py`, which does not exist.
      *Done when:* every table in `Base.metadata.tables` has a mapped class.
      **Done** — and extended to the failure the module docstring actually
      warns about: a mapped class missing from `__all__` is invisible to
      `alembic` autogenerate, so its table silently never gets created.

- [ ] **1.5 Challenger harness.** The metric functions take a model id, so the
      same fixtures can be scored on `openai/gpt-oss-120b`,
      `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`. The Qwen model is Preview
      tier and 5-6x the price, so it belongs here and nowhere else
      (`README.md` §7).
      *Done when:* one command scores all three and prints a comparison.
      **Partly done** — `tests/eval/run_extraction_eval.py`. `--replay DIR`
      scores recorded runs for any number of models today, with no key and no
      network. `--live --model all` is written but refuses to run: it imports
      the real extractor and says so if absent, rather than inventing a
      throwaway prompt — an eval scored against a prompt nothing else uses
      measures nothing. **Outstanding:** the live three-way comparison, which
      needs tasks 3.1/3.2 and a key (wired in 3.6).

---

## Phase 2 — Transcript parse and roster

Unblocks `no_economic_buyer` and `single_threaded`, which today read a table
nothing populates from transcripts.

- [x] **2.1 Speaker-aware parse.** `services/ingest.py` — for
      `source_type='meeting_transcript'`, populate
      `document_chunks.metadata` with `{speaker, char_start, char_end}` and
      prefer splitting on speaker turns. Deterministic; no model.
      *Done when:* every chunk of a fixture transcript carries a speaker and
      offsets that resolve verbatim.
      **Done** — and chunks start on a turn boundary as well as ending on one,
      which the first attempt got wrong: the overlap pulled each start back
      mid-turn so every chunk after the first opened unattributed. Prose is
      untouched (a test pins the old offsets). Re-cutting moved every fixture
      offset, so `load.py` + `resolve_labels.py` were re-run and the recorded
      eval run regenerated.

- [x] **2.2 Roster extraction.** Distinct speaker labels → `meeting_attendees`
      rows with `raw_name`, `attended=true`, `is_internal` inferred from the
      label or left false.
      *Done when:* a fixture transcript yields one row per distinct speaker and
      a re-run inserts nothing new.
      **Done** — 4 / 4 / 5 attendees, matching `manifest.json` exactly, and 13
      rows after two full runs. Idempotence has to be enforced in the service:
      `uq_meeting_attendees_meeting_id_contact_id` is partial
      (`WHERE contact_id IS NOT NULL`) so the database does not stop an
      unresolved name being inserted twice.

- [x] **2.3 Entity resolution, deterministic first.** `pg_trgm` similarity of
      `raw_name` against `contacts`. Link only above a high threshold; leave
      `contact_id` NULL otherwise. **Never guess** — a wrong link silently
      corrupts every attendance-based risk, and an unresolved attendee is the
      missing-stakeholder signal.
      *Done when:* exact and near-exact names link, ambiguous ones stay NULL,
      and the migration enabling `pg_trgm` is in place.
      **Done** — `0010_pg_trgm` with a `gin_trgm_ops` index on the concatenated
      name, round-tripped. Observed similarities: exact 1.00, typo
      ("Priya Ramen") 0.60, bare first name 0.50, unrelated 0.00. Thresholds in
      settings: link at ≥0.75 *and* ≥0.15 clear of the runner-up, candidate for
      the tiebreak at ≥0.40, nothing below that reaches a model. On the fixture
      corpus: Priya and Marcus link, Maya/Tom/Dana stay NULL, **zero model
      calls**.

- [x] **2.4 LLM tiebreak on the residue only.** `openai/gpt-oss-20b`, given the
      unresolved label and the deal's candidate contacts, returns a
      `contact_id` or null. Never invents a contact.
      *Done when:* the ambiguous fixture case resolves, and a name with no
      plausible candidate returns null.
      **Built, not yet run live.** `app/ai/tiebreak.py` + the
      `roster_tiebreak@1` prompt. The model answers with a **candidate number,
      not a contact id** — the dossier's handle pattern: an out-of-range answer
      is detectable and treated as "no match", where a fabricated uuid would
      not be. Unit-tested with a stubbed client for the link, the out-of-range
      and the AI-disabled paths.
      **Live call verified** — `tests/verify_ai_tiebreak.py` builds the two
      cases the corpus lacks and cleans up after itself:
      a transcription slip ("Priya Ramen", similarity 0.60) is deferred and
      correctly linked to Priya Raman; **two contacts sharing a first name
      ("Priya" → Priya Shah 0.55, Priya Raman 0.50) is deferred and the model
      declines to guess, leaving `contact_id` NULL** — which is the behaviour
      the prompt asks for and the one that matters, since a wrong link
      corrupts every attendance-based risk invisibly. "Tom Alvarez" never
      reaches a model at all.

---

## Phase 3 — Extraction and Gate 0

- [x] **3.1 `payload` schemas per `fact_type`.** `app/ai/schemas.py` — one
      Pydantic model per value of `FactType` (budget → `{amount, currency,
      basis}`, deadline → `{date, what}`, ...). These are both the structured
      output schema and what makes the literal rule checkable: you know which
      field holds the money. Strict mode needs every field `required` and
      `additionalProperties: false`, so nullable fields are `["string","null"]`
      unions rather than omitted keys (`README.md` §7).
      *Done when:* each validates a hand-written example, rejects a bad one, and
      its JSON Schema is accepted by Groq with `strict: true`.
      **Done** — `app/ai/schemas.py`, in two layers. The **wire** schema is
      flat, because strict mode forbids the obvious design twice over: a
      free-form `Dict[str, str]` renders as `additionalProperties: {schema}`
      where strict needs `false`, and a Pydantic field with `= None` is omitted
      from `required`. So all nine payload fields are `Optional[...]` with **no
      default** — present-but-null, never missing. The **per-type** models are
      the semantic contract, checked in Python after parsing, and are what make
      a `budget` fact promotable later. `strict_schema_problems()` asserts
      conformance locally, since this interpreter's client cannot send `strict`
      at all.

- [x] **3.2 The extractor.** `app/ai/extract.py` — one structured
      `openai/gpt-oss-120b` call per chunk window, run in parallel. Returns facts
      **with their spans in the same output** — `chunk_id` plus the verbatim
      `snippet`. The model never returns character offsets: Python locates the
      quote in the chunk by substring search and computes
      `char_start`/`char_end`, and a quote it cannot find dies at Gate 0. No
      tools, no retrieval.
      *Done when:* a fixture transcript yields facts whose snippets are literal
      substrings of the cited chunks, and a paraphrased snippet is rejected.
      **Done** — `app/ai/extract.py` + the `extract@1` prompt. Windows are
      even-sized rather than greedy (a greedy split leaves a one-chunk
      remainder, the worst possible input for a task that needs surrounding
      context) and non-overlapping, since the chunks inside them already
      overlap. `render_window` trims that overlap using the stored offsets, so
      the model is not shown the same exchange twice and invited to extract it
      twice.
      `locate()` tries exact, then **case-insensitive**, and stores the
      *document's* characters either way — a model quoting mid-sentence writes
      "we'd" where the transcript has "We'd", and that is the same quote.
      Nothing looser: a fuzzy match would store a span that then fails Gate 0
      later, further from the cause.

- [x] **3.3 Gate 0.** `services/claims.py` — for each `claim_evidence` link:
      document spans matched verbatim at their offsets; `record_ref` resolved to
      a live row whose field still equals `snippet`; and the **literal rule** —
      any date or monetary amount in the claim text must appear in a cited span
      or resolved value. Writes `verification_status`. Runs *inside the insert*.
      *Done when:* a deliberately corrupted snippet yields `span_missing`, a
      mutated record yields `value_drifted`, and a fabricated date fails.
      **Done** — `app/services/gate0.py`, in `services/` rather than `app/ai/`
      because it touches no provider and must check a human-entered citation by
      the same code. `record_ref` resolution goes through an **allowlist of
      table and column names**: it is jsonb a model wrote, and resolving an
      arbitrary table from it would be an injection point into our own schema.
      The literal rule extracts dates and money **only in figures** —
      "a hundred and fifty thousand" is deliberately not matched, because the
      rule exists to catch a model *converting* speech into a precise-looking
      literal that was never said.

- [x] **3.4 Zero-evidence rejection.** A claim with no surviving link is not
      written. Enforced in the service layer, not by the caller.
      *Done when:* inserting a claim with no evidence raises, and nothing lands.
      **Done** — `app/services/facts.py` is the only writer of model-produced
      facts and runs Gate 0 *inside* the insert. A caller may pass checks it
      already computed (the graph does, so the gate is not run twice against
      the database) but they are honoured, not trusted: a check that did not
      pass means no write, whoever computed it. Rejections are returned and
      logged, never persisted — a row for something that failed verification is
      a row somebody eventually renders.
      `attach_evidence` also had to grow `char_start`/`char_end`/`speaker`/
      `occurred_at`: it could not record a document span at all, and a span
      with no offsets can never verify.

- [x] **3.5 Wire into the pipeline** as stages 2–4, writing
      `extracted_facts` (`status='pending'`), `evidence`, `claim_evidence`.
      *Done when:* one queued meeting produces pending facts visible through the
      existing API, and a re-run with the same document is a no-op.
      **Done** — stages 2/3/4 in `app/ai/stages.py`, and again as graph nodes
      in 3.7. Re-analysis is idempotent at the **document** level, not the fact
      level: a fact has no natural key, so "this document has already been
      extracted" is the only safe statement.

- [x] **3.6 Measure.** Run 1.3 against the three fixtures; record the baseline.
      *Done when:* Gate 0 pass rate and fact precision/recall are written down.
      **Baseline recorded** — `openai/gpt-oss-120b`, `extract@1`, whole corpus,
      27,877 tokens, 0 retries. Replayable from
      `tests/eval/recorded/extract-baseline-gpt-oss-120b.json`.

      | | |
      |---|---|
      | reported / located / **unlocatable** | 52 / 52 / **0** |
      | payload narrowing errors | 5 |
      | recall (type + span) | **0.65** — 17 of 26 labels |
      | precision | 0.33 — **not interpretable, see below** |

      Recall by type: `budget 3/3 · competitor 1/1 · requirement 4/5 ·
      decision_criteria 3/5 · stakeholder 2/3 · deadline 2/3 ·
      **commitment 1/3** · **objection 1/3**`.

      **Zero unlocatable snippets is the headline.** Every quote the model
      returned was found verbatim in a chunk, so the "copy, don't describe"
      instruction holds and Gate 0 has something real to check in every case.

      **Precision 0.33 is mostly a measurement artifact.** Of the 35
      unmatched predictions: **10 cite a labelled span under a different
      `fact_type`** (taxonomy disagreement, not fabrication) and 25 fall
      outside any labelled span — and inspection shows most of those are real
      facts the labels simply do not contain ("the audit is scheduled for Q4",
      "finance will need to sign off"). The labels were written as *the planted
      cases plus notable facts*, never as an exhaustive enumeration, so
      precision cannot be computed against them. **Recall is the valid number
      here; precision needs either exhaustive labels or an adjudicated
      sample.**

- [x] **3.7 Convert the pipeline to a LangGraph `StateGraph`.** Design:
      `README.md` §4, *The LangGraph graph*. The payoff is stage 2's fan-out,
      which is why the conversion waits until there are real facts to merge.
      Four parts:
      (a) `PipelineState` becomes a merged state schema with
      `facts: Annotated[List[...], operator.add]` — without the reducer, N
      parallel `extract_window` nodes writing one key is an error;
      (b) stages return partial dicts instead of mutating state;
      (c) the `AsyncSession` moves from the stage signature to graph context,
      since state is merged and may be serialized;
      (d) a `plan_windows` node emitting `Send("extract_window", ...)` per
      window, and a conditional edge after `drop_unevidenced` routing to `END`
      when nothing survived.
      `retry_policy` only on `extract_window` — it writes nothing until gate 0,
      and every other stage would write twice. `compile()` with no checkpointer.
      *Done when:* the graph runs the same 13 stages with identical outcomes to
      the sequential runner, parallel windows merge into one fact list, and
      `get_graph().draw_mermaid()` reproduces the diagram in §4.
      **Done** — `app/ai/graph.py`. `facts: Annotated[List, operator.add]` is
      the line that justifies the graph: without it two `extract_window` nodes
      writing one key in a single step is an error. The `AsyncSession` and the
      `Meeting` travel as `AnalysisContext` rather than in the state, which is
      merged and may be serialized. `retry_policy` is on `extract_window`
      alone — it writes nothing before returning, so it is the only node a
      retry cannot duplicate. `draw_mermaid()` reproduces §4's shape including
      the fan-out and the `drop_unevidenced → END` branch. Compiled with no
      checkpointer.
      **The sequential runner is gone** (see Found during implementation). The
      stage *declaration* it carried — the thirteen names and the
      critical/degradable flags — moved to `app/ai/stage_registry.py`, which a
      test now asserts the graph's nodes against in both directions. `_StateShim`
      is gone with it: stage functions return dicts, so graph nodes are three
      lines each.

- [x] **3.8 Usage accounting through a callback.** `_usage(raw)` sees only the
      response `client.py` holds, so a model call made inside a graph node or an
      agent loop is invisible to `AIRun` and uncharged to `RunBudget` — which
      under-counts exactly where a runaway is likeliest. Wrap runs in
      `get_usage_metadata_callback()` and charge the budget from the aggregate.
      `README.md` §7, *Governance*.
      *Done when:* a run whose model calls bypass `client.structured()` still
      charges `RunBudget`, and a deliberately uncapped loop trips
      `TokenCeilingExceeded`.
      **Done** — `client.track_usage()` wraps a run in
      `get_usage_metadata_callback()`, which registers a context-scoped handler
      and so sees anything LangChain runs inside the block without that code
      knowing about it. Verified live: the callback's total matched the count
      read off our own response exactly (225 tokens, one model).

- [x] **3.9 Expose the governor as a `BaseRateLimiter`.** Needed as soon as a
      graph node invokes a runnable directly, since `ChatGroq(rate_limiter=...)`
      is what reaches calls `client.structured()` never sees. **Only the
      request/concurrency half can move:** `BaseRateLimiter.acquire(*,
      blocking)` takes no request size, so tokens/min is not representable
      there and stays in `client.py` (`README.md` §7, *Governance*).
      *Done when:* a model call made inside a node is subject to the RPM and
      concurrency ceiling, and the TPM bucket still sees it through
      `client.py`.
      **Done, with one honest limitation.** `client.GovernorRateLimiter`
      implements `BaseRateLimiter` over the governor's request bucket and is
      attached by `client.agent_model()`. It can carry **requests only**:
      `acquire(*, blocking)` takes no argument describing the request, so
      tokens-per-minute is not expressible through the interface — and
      concurrency is not either, because a limiter gates entry and is never
      told the call finished, so it has nothing to release. Both stay in
      `client.structured()`. Never attached to the `structured` path, which
      takes from the same bucket itself and would be charged twice.

---

## Phase 4 — Gate 1, the Evidence Validator

- [x] **4.1 The validator.** `app/ai/validate.py` — one `openai/gpt-oss-120b` call
      per claim, given the claim text and **only** its cited spans. No tools,
      no transcript, no deal record. Returns
      `supported|partial|contradicted|unsupported` plus a rationale.
      *Done when:* a hand-built supported case, a partial case and a
      contradicted case each get the right verdict.

- [x] **4.2 Persist.** Append a `claim_validations` row per run with `verdict`,
      `method='llm'`, `rationale`, `model`, `validator_version` from 0.4.
      *Done when:* two runs leave two rows, never an update.

- [x] **4.3 Quarantine policy.** `contradicted` and `unsupported` never render;
      `partial` renders with a caution badge. Enforced in the read path so no
      future route can forget it.
      *Done when:* a contradicted fact is absent from the API response.

- [x] **4.4 Log the pairs.** `confidence × verdict` into the metrics from 1.3.
      *Done when:* high-confidence contradicted cases are countable.

---

## Phase 5 — Meeting synthesis

- [x] **5.1 Migration `0011_meeting_analysis_origin`.** `meetings` +
      `analysis_origin`, `confidence`, `model`; `deal_contacts` + `origin`.
      Note the two schema traps in `docs/schema/TASKS.md` — a native enum
      survives `DROP COLUMN`, and `create_check_constraint` runs names through
      the naming convention.
      *Done when:* `alembic upgrade head` then `downgrade -1` both succeed.

- [x] **5.2 Summary from facts, not the transcript.** `app/ai/synthesize.py` —
      `openai/gpt-oss-120b` over the surviving fact set plus the attendee roster.
      ~1K input tokens, and every sentence traces to a fact that traces to a
      span.
      *Done when:* the summary asserts nothing absent from the fact set.

- [x] **5.3 Sentiment from the transcript.** `openai/gpt-oss-20b`,
      returning a `Sentiment` value plus two or three illustrative quotes
      attached with `relevance` — explicitly outside the Gate 1 contract.
      *Done when:* `meetings.sentiment` is set and the quotes resolve verbatim.

- [x] **5.4 Wire as stages 9–11** and set `analyzed_at`,
      `analysis_status='complete'`. A failure here leaves the facts intact and
      logs the stage error.
      *Done when:* the Meeting Analyzer projection returns a real `summary` and
      `sentiment`, and killing the process at stage 9 still leaves facts.

---

## Phase 6 — Reconciliation and supersession

- [x] **6.1 Commitment reconciliation.** `app/ai/reconcile.py` —
      `openai/gpt-oss-120b` matches new commitment-facts against that deal's open
      `commitments`. Output is a **proposed** status change, written as a
      `recommendation`, never a direct write to `commitments.status`.
      *Done when:* the fixture's fulfilled commitment yields a proposal and an
      unrelated fact yields none.

- [x] **6.2 Supersession.** Compare new facts against live accepted facts of the
      same `fact_type` for the deal. On contradiction: mark the old
      `superseded`, link the new evidence, **keep both**.
      *Done when:* the fixture's contradicting fact supersedes its predecessor
      and neither row is deleted.
      The comparison set was `extracted_at DESC LIMIT 10`, which bounded
      recall rather than cost; it is now chosen per new fact by trigram
      similarity (`0016`). See *Found during implementation*.

- [x] **6.3 Staleness (SQL).** Flag claims whose newest `evidence.occurred_at`
      predates `deals.last_activity_at`; write `stale` to
      `claim_evidence.verification_status`.
      *Done when:* an old claim on a recently-active deal reads `stale`.

---

## Phase 7 — AI risk and recommendation detection

Shadow-run before it writes anything. Step 7.4 is the dress rehearsal against
free labels and should not be skipped.

- [x] **7.1 Migration `0012_detector_provenance`.** `risks`, `recommendations` +
      `model`, `detector_version`. Backfill existing rows with
      `detector_version='deterministic-1'` so there is a labelled "before".
      *Done when:* upgrade/downgrade both succeed and existing rows are tagged.

- [x] **7.2 Dossier builder.** `app/ai/dossier.py` — pure Python, no model.
      Deal fields, stage history, stakeholder map with attendance counts, open
      commitments, accepted facts, meeting cadence, prior dismissals, and the
      currently-open risks. Every entry is `(handle, source_kind, ref, text)`.
      *Done when:* a unit test asserts every handle resolves to a real record or
      fact, with no API call.

- [x] **7.3 Handle validation.** Any `evidence_ref` not in `dossier.keys()` is
      rejected before any write. This is what makes a fabricated citation
      structurally impossible.
      *Done when:* a doctored model output with an unknown ref is refused.

- [x] **7.4 Shadow mode on the four deterministic types.** Run the detector
      call, write **nothing**, and compare against `detect.py`: recall on the
      four overlapping types, and risk-key Jaccard across two runs on an
      unchanged deal.
      *Done when:* recall and stability are recorded and meet your bar.
      **Run, and it did its job twice over.** Against the three risks the SQL
      rules confirm on the fixture deal:

      | | |
      |---|---|
      | recall (proposed + affirmed) | **1.00** — all three |
      | added beyond the rules | `budget_unconfirmed`, `missed_commitment`, `security_review_pending` — three of the six semantic types SQL cannot express |
      | **stability (Jaccard, two runs, unchanged deal)** | **0.67** |
      | tokens, two runs | 9,380 |

      Stability 0.67 is the number to watch: `budget_unconfirmed` was proposed
      in run 1 and not in run 2. `temperature=0` does not make a reasoning
      model deterministic. **The write path absorbs it** -- a proposal flickers
      harmlessly because the upsert keeps an open risk open and absence never
      resolves anything (7.9); flicker in a *verdict* would matter far more,
      and all three verdicts agreed across both runs. Worth re-measuring over
      more runs before trusting the figure.

- [x] **7.5 Turn on writes.** Stage D reconciliation in Python — upsert by
      `(deal_id, risk_type, risk_key)`, bump `last_seen_at`, never insert a
      duplicate, suppress anything inside a dismissal cooldown. Risk and its
      recommendation from the same call. Gate 0 and Gate 1 applied as in
      Phases 3–4.
      *Done when:* two consecutive runs on one deal leave one row per risk, and
      the six semantic types appear with surviving citations.
      **Done** — proposals and recommendations now pass Gate 0 and the starved
      Gate 1 before either row is written. Record evidence stores the literal
      field value (not the dossier's explanatory rendering), accepted facts
      contribute their original source spans, existing risks are bumped, and
      the suggested recommendation is upserted with them. A database test pins
      the two-run identity and verified-citation contract.

- [x] **7.6 Severity policy.** Compute the band in Python for `stalled_stage`,
      `close_date_at_risk` and `missed_commitment` and clamp the model's
      proposal to it. Hysteresis: raising is immediate, lowering needs new
      evidence or N days.
      *Done when:* two runs on an unchanged deal never change a severity.

- [x] **7.7 Migration `0013_open_risk_taxonomy`.** `risks` + `risk_key text not
      null default ''`; `RiskType.OTHER` (a CHECK swap); drop and recreate the
      partial unique index as `(deal_id, risk_type, risk_key) WHERE
      status='open'`. Dropping a CHECK by name needs raw SQL — see the trap in
      `docs/schema/TASKS.md`.
      *Done when:* the ten known types behave exactly as before, and two
      distinct `other` keys coexist on one deal.

- [x] **7.8 Key canonicalization.** Lowercase and strip a proposed `risk_key`,
      then trigram-match it against that deal's open keys and reuse on a hit.
      *Done when:* `champion_going_quiet` and `champion_disengaged` collapse to
      one row.

- [x] **7.9 Resolution by verdict.** Feed open risks into the dossier; require a
      per-risk `still_present|resolved|unclear`. Resolve only on a
      Gate-0-surviving `resolved` verdict, and only after two consecutive ones
      or a human confirmation. Deterministic types keep their SQL auto-resolve.
      *Done when:* a risk whose cause is gone resolves with an attached
      evidence row, and one omitted from the output is left untouched.

- [x] **7.10 Dismissal feedback policy.** Enforce the four rules from
      `README.md` §5 in Python, not in the prompt.
      *Done when:* a recommendation dismissed as `already_handled` is not
      re-proposed inside the cooldown, and one dismissed as `bad_timing` is.

---

## Phase 8 — Meeting briefs

- [x] **8.1 Brief generator.** `app/ai/brief.py` — one `openai/gpt-oss-120b` call over
      prefetched rows: deal, open risks, pending commitments, prior meeting
      summaries, stakeholder map. Structured into `objectives`, `key_risks`,
      `recommended_questions`, `context_summary`.
      *Done when:* a brief persists with `model` and `generated_at`.

- [x] **8.2 Regenerate only on `force`.** `UniqueConstraint(meeting_id)` already
      enforces one per meeting.
      *Done when:* a second request returns the stored brief without a call.
      **Done** — `POST .../meetings/{id}/brief` persists structured objectives,
      risks, questions, context, model and timestamp; `GET` reads it. A second
      generation returns the stored row, while `force=true` replaces it.

---

## Phase 9 — Chat

- [x] **9.1 Read-only tool set.** `app/ai/tools/` — the eight tools in
      `README.md` §6, each returning content **plus evidence handles**.
      *Done when:* every tool result carries a `chunk_id` + offsets or a
      `record_ref`.
      **Done** — eight `StructuredTool` wrappers return bounded JSON entries;
      each entry has a generated evidence handle backed by either an exact
      document chunk span or an allowlisted record field. There is no write
      tool and no generic SQL surface.

- [x] **9.2 Deal scoping in Python.** When `chat_sessions.scope='deal'`, the
      `deal_id` comes from the session row, never from a model argument.
      *Done when:* a tool call naming another deal's id is refused.
      **Done** — the registry is constructed with the session's immutable
      `deal_id`; omitted ids inherit it and mismatched or malformed ids raise a
      tool error before a query is constructed.

- [x] **9.3 The loop.** LangGraph's prebuilt ReAct agent, streaming, writing
      into a `chat_messages` row created with `status='streaming'` so a refresh
      mid-answer does not lose the turn.
      *Done when:* an interrupted stream leaves a recoverable row, and the
      turn's usage covers **every** model call the agent made, not just the
      first (task 3.8).
      **Done** — the user and empty assistant rows commit before inference,
      response deltas stream over SSE and checkpoint every 500 characters,
      and failures retain partial content with `status='error'`. The
      context-scoped usage callback accounts for every ReAct iteration.

- [x] **9.4 Citations.** Attach `claim_evidence` with
      `claim_type='chat_message'`, and run Gate 0 over the handles the answer
      actually used.
      *Done when:* an answer renders with clickable citations and an uncited
      assertion is visibly uncited.
      **Done** — only inline `[eN]` handles actually used in the final answer
      are attached, every link passes Gate 0, and handle-to-evidence identity
      is retained in the message usage envelope for the read API. Text without
      a matching inline handle remains visibly uncited.

- [x] **9.5 Session titles.** `openai/gpt-oss-20b` after the first turn,
      fire-and-forget.
      *Done when:* `chat_sessions.title` is populated.
      **Done** — a best-effort background task uses the cheap model and the
      versioned `chat-title@1` structured prompt, then writes through its own
      database session so it does not hold the streaming request open.

---

## Phase 10 — Triggering and cadence

Events invalidate, the clock sweeps. Design: `README.md` §3, Triggers.
**10.1–10.4 need only Phase 3; 10.5 needs Phase 7.**

- [x] **10.1 Migration `0014_analysis_triggers`.** `deals` +
      `analysis_dirty_first_at`, `analysis_dirty_last_at`,
      `analysis_dirty_reason`, `analysis_swept_at`. Partial index on
      `(analysis_dirty_last_at) WHERE analysis_dirty_first_at IS NOT NULL`, and
      one on `analysis_swept_at`. Debounce and sweep windows go in
      `app/core/config.py`, not in the SQL.
      *Done when:* upgrade and downgrade both succeed and the poll queries in
      `README.md` §3 use an index.
      **Done** — the four nullable timestamps/reason fields, partial dirty
      index, and sweep index round-trip at head `0014`; both worker claims use
      the indexed columns and `FOR UPDATE SKIP LOCKED`.

- [x] **10.2 Dirty marking in the service layer.** One helper —
      `services/analysis.py::mark_dirty(deal_id, reason)` — called from the
      writes in the "Which writes matter" table. Set `first_at` only if null,
      always push `last_at`. **Not** a Postgres trigger: `services/` is already
      the choke point, and a DB trigger fires on migrations and seeders and
      cannot see `origin`.
      *Done when:* changing a deal's stage marks it dirty, editing
      `deals.description` does not, and a unit test covers each row of the table.
      **Done** — `services/analysis.py` owns the three tiers and the relevant
      deal, meeting/attendee, stakeholder, commitment, task, transcript, and
      delete write paths call it. Non-semantic edits do not.

- [x] **10.3 Loop guards.** Writes with `origin='ai'` and recommendation
      dismissals never mark a deal dirty. Without this the detector
      re-triggers itself.
      *Done when:* a full AI detection run leaves the deal clean, and
      dismissing a recommendation does not enqueue anything.
      **Done** — `mark_dirty` rejects AI origin centrally; detector writes and
      recommendation dismissals have no trigger edge. Both are regression-tested.

- [x] **10.4 Tier 0 — eager re-verification.** On any update to a field named by
      some `evidence.record_ref`, re-resolve those links in the same
      transaction and write `value_drifted` where the value moved. No model.
      *Done when:* moving a deal from `discovery` to `negotiation` flips the
      citations on risks that cited the stage, in the same request.
      **Done** — JSON record refs are selected by table/id/field and rechecked
      before commit; the stage test observes `value_drifted` without a second
      request. Deletes recheck their surviving record refs too.

- [x] **10.5 Tier 1 — eager deterministic detection.** Call `detect.py` from the
      service layer on the tier-1 events, not only from
      `POST /deals/{id}/analysis`. Cheap and exhaustive, so it cannot flicker.
      *Done when:* completing a meeting refreshes the four SQL risks without a
      separate API call.
      **Done** — relevant writes synchronously refresh the deterministic floor.
      The floor is now six rules: the original four plus `missed_commitment`
      and `gone_quiet`.

- [x] **10.6 Tier 2 — the debounced AI pass.** Second poll query in the worker,
      using the debounce from `README.md` §3, with per-deal single-flight via an
      advisory lock or `FOR UPDATE SKIP LOCKED`. Clear the dirty columns and set
      `analysis_swept_at` on success.
      *Done when:* five field edits inside the debounce window produce exactly
      one AI run, and two concurrent triggers on one deal produce one run.
      **Done** — quiet and maximum debounce windows are configurable; five
      committed edits collapse into one run and concurrent claims skip the
      locked deal. Success clears dirty state and stamps `analysis_swept_at`.

- [x] **10.7 The nightly sweep.** Third poll query —
      `analysis_swept_at IS NULL OR < now() - 24h` — on the same loop. No cron,
      no scheduler, no second container.
      *Done when:* a deal untouched for a day is re-analysed, and a
      time-only risk (`missed_commitment` on a commitment that just went
      overdue) is detected with no write having occurred.
      **Done** — the worker claims one clean stale deal per pass. A focused
      test advances only the clock and proves the sweep creates the overdue
      commitment risk; `gone_quiet` is covered independently.

- [x] **10.8 Bulk-import suppression.** A settings flag or context manager that
      suppresses tier 2 marking, for the seeder and any bulk import. Tiers 0 and
      1 stay on — they are cheap and deterministic.
      *Done when:* importing 200 deals enqueues zero AI runs, and the nightly
      sweep picks them up instead.
      **Done** — both a process setting and a nest-safe context manager suppress
      only Tier 2. A 200-row regression test leaves dirty state empty; Tier 1
      remains active and the ordinary sweep remains the catch-up mechanism.

---

## Phase 11 — Closing the loop

Phases 0–10 compute more than the API exposes, and one stage produces a result
with nowhere to put it. Nothing here is new capability; each item connects
something that already works to something that can see it. **Single-user MVP —
there is no `users` table and none is planned**, so ownership, per-user chat
sessions and human-vs-human attribution are explicitly not questions here;
`origin` already carries the only distinction that matters (`ai` vs `human`).

Depends on Phases 0–10. 11.1 needs migration `0015`; the rest need none.

- [x] **11.1 Migration `0015_commitment_correction_action`.**
      `ActionType.CORRECT_RECORD` — a CHECK swap, since
      `recommendations.action_type` is `String(50)` plus a `check_in` constraint
      (`models/assertion.py:201,250`) precisely so this is cheap. Dropping a
      CHECK by name needs raw SQL; see the trap in `docs/schema/TASKS.md`.
      A **distinct** value rather than reusing `update_close_date`, for the
      reason stage 7's docstring gives: the `dismissal_reason` and action-type
      distributions are two of the few signals here that are not self-reported,
      and filing a data correction under an action would corrupt both.
      *Done when:* upgrade and downgrade both succeed, the seven existing action
      types behave exactly as before, and a `correct_record` row inserts.
      **Done** — head at `0015`, round-tripped. It carries more than the
      planned CHECK swap: `source_commitment_id` plus a partial unique index
      proved necessary, because the existing one-live-suggestion index is
      keyed on `source_risk_id` and a correction has none — without it stage 7
      would file a duplicate on every run. `SET NULL`, not `CASCADE`, since
      `claim_evidence.claim_id` has no foreign key and a database-level
      cascade would orphan links behind `services/claims.py`'s back.

- [x] **11.2 Stage 7's proposals get a surface.** `app/ai/stages.py:460` runs
      `reconcile.reconcile_commitments`, logs the result and **writes nothing** —
      the gap its own docstring records. Write each proposal as a
      `recommendation` with `action_type='correct_record'`, `status='suggested'`,
      the commitment as its subject and the reconciliation rationale as its
      description. It then flows through the existing
      `GET /deals/{id}/recommendations` and the accept/dismiss routes with no new
      endpoint. Accepting applies the commitment status change; dismissing feeds
      the same cooldown policy as every other recommendation (7.10).
      *Done when:* a commitment the transcript shows was satisfied produces a
      dismissible proposal instead of a log line, accepting it moves the
      commitment, and dismissing it suppresses re-proposal inside the cooldown.
      **Done** — `stages._file_correction` upserts one `correct_record`
      suggestion per commitment and `_link_fact_evidence` points it at the
      facts' own evidence rows, re-running Gate 0 per link and dispatching on
      `source_kind`. The commitment itself is never touched. Three tests:
      filed-not-applied, second-run-updates-not-duplicates, and
      already-satisfied-is-moot.

- [x] **11.3 `GET /deals/{id}/analysis`.** Phase 10 added
      `analysis_dirty_first_at`, `analysis_dirty_last_at`,
      `analysis_dirty_reason` and `analysis_swept_at`, and nothing reads them —
      only `POST /deals/{id}/analysis` exists, so the UI can trigger a run but
      cannot say one is pending. Return the four columns plus a derived state
      (`clean` / `queued` / `stale`) computed from the same debounce settings the
      worker claims with, so the badge and the claim query cannot disagree.
      *Done when:* five edits inside the debounce window read as one `queued`
      state carrying a reason, and the state returns to `clean` with a fresh
      `analysis_swept_at` after the worker runs.
      **Done** — four states, not three: `debouncing` and `due` are kept apart
      because one means "still collecting your edits" and the other "running
      shortly", and collapsing them would make the quiet window look like
      latency. A test asserts `due` and `worker._claim_dirty_deal` agree on
      the same deal.

- [x] **11.4 Expose `meetings.analysis_error`.** The column exists (`0011`) and
      `stages.py:448` writes it; `MeetingAnalysis` does not return it, so the
      Analyzer screen can say *that* a run degraded but not *why*. One field on
      the schema and the projection.
      *Done when:* a meeting with a failed degradable stage reports the stage
      error through `GET .../meetings/{id}/analysis`.
      **Done** — one field on `MeetingAnalysis` and `_analysis_payload`. The
      column and its writer already existed; only the projection was missing.

- [x] **11.5 `GET /system/ai-status`.** `client.py` already logs every run as
      `ai.run task=… model=… version=… outcome=… in=… out=… ms=…` (task 0.7), so
      per-run tracing exists — what is missing is *aggregate current state*:
      governor token-bucket headroom, configured TPM/RPM, `ai_enabled`, the
      count of dirty deals, the oldest unswept deal, and the last poll time.
      Given that the configured TPM was once wrong by 31x and that a throttled
      extraction window waits 77 seconds, this is the surface that makes both
      legible without tailing the worker.
      **Langfuse and LangSmith stay out** — see Out of scope. Structured logs
      plus this endpoint are the whole observability story for the MVP, and the
      tradeoff is deliberate: no per-run history, no trace tree.
      *Done when:* a throttled run and a backed-up queue are both visible from
      one request.
      **Done** — config, queue depth and token budget. The budget is labelled
      `api:pid-N` rather than presented as global: the governor is a
      per-process singleton and the worker is a separate container, so the
      API's bucket is not the one doing the work. A shared bucket would need
      Redis.

- [x] **11.6 Chat session lifecycle.** Three gaps in Phase 9, all small:
      `PATCH /chat/{session_id}` to rename over the auto-generated title (9.5
      writes it; nothing can correct it), `DELETE /chat/{session_id}`, and a
      reaper for abandoned `status='streaming'` rows. 9.3 deliberately commits
      the assistant row before inference so a refresh mid-answer is recoverable,
      but nothing ever closes a row whose client never came back. The reaper is a
      fourth poll query in `worker.py` — which is written to carry more than one
      for exactly this reason (0.5) — finalizing rows older than a configurable
      window as `status='error'` with their partial content retained.
      *Done when:* a killed stream's row reaches a terminal status with no new
      request, and a renamed session survives a later turn without being
      re-titled.
      **Done** — `PATCH` (rename over the auto-title), `DELETE` (clearing
      `claim_evidence` through `services/claims.py` first, since `claim_id`
      has no foreign key), and `reap_abandoned_streams` as the worker's fourth
      poller. Reaped rows become `error`, not `complete`, keeping their
      partial content: a truncated answer presented as finished is worse than
      one that admits it.

- [x] **11.7 Document upload refreshes the deterministic floor.**
      `routes/deals/documents.py:218` calls `analysis.mark_dirty` directly rather
      than `record_change`, so an upload marks the deal for the AI pass but skips
      tiers 0 and 1. That matters for one rule specifically: the same request
      advances `last_activity_at` (via `activity.touch_deal`, line 204), and
      `gone_quiet` reads that column — so an upload that should clear a
      `gone_quiet` risk leaves it standing until the nightly sweep. Route it
      through `record_change` like the other sixteen call sites.
      *Done when:* uploading a document to a quiet deal clears `gone_quiet` in
      the same request.
      **Done** — `record_change` instead of a bare `mark_dirty`, with no
      table/row_id, since a new document changes no existing row and tier 0
      has nothing to re-resolve.

- [x] **11.8 `propose_task`, the one write tool.** `README.md` §6 allows exactly
      this one, "after the read path is trusted" — which 9.1–9.4 establish:
      every tool result carries an evidence handle, scoping comes from
      `chat_sessions.deal_id` in Python, and answers are Gate 0 checked. The tool
      creates a `recommendation` with `status='suggested'`, **never** a `task`:
      the middle arrow in `risk → recommendation → [human] → task` is the
      product. It takes the session's immutable `deal_id` like the eight read
      tools and needs no new route — accept and dismiss already exist.
      *Done when:* a chat turn can propose an action that appears in the
      recommendations list, a foreign `deal_id` is refused before any write, and
      no code path lets the agent create a `task` row.
      **Done** — ninth tool, `status='suggested'`,
      `detector_version='chat-propose-1'` so agent suggestions are
      distinguishable from detector ones. Three tests:
      creates-a-suggestion-not-a-task, refuses-another-deal, and a guard that
      no second write tool has appeared. `test_chat.py`'s exact-set assertion
      caught the contract change, which is what it is for.

- [x] **11.9 A real stability sample for detection.** 7.4 recorded Jaccard 0.67
      across **two** runs, with its own note saying the figure is worth
      re-measuring. Run the detector ten times against the unchanged fixture
      deal, write nothing, and record risk-key Jaccard plus — the number that
      matters more — whether per-risk verdict flicker stays at zero. Stability is
      the first thing to watch (`README.md` §7): a flickering risk panel reads as
      brokenness regardless of precision.
      *Done when:* a stability figure with a stated sample size is in this file,
      and verdict flicker is quantified rather than assumed.
      **Done**, and the headline is not the Jaccard. Ten runs against the
      unchanged fixture deal, nothing written
      (`tests/eval/run_stability_eval.py`, raw runs in
      `tests/eval/recorded/stability-10.json`):

      | | before the parse fix | after |
      |---|---|---|
      | runs that failed outright | **3** of 10 (0.30) | **0** of 10 |
      | runs scored | 7 | 10 |
      | mean pairwise Jaccard | 0.959 | **0.882** |
      | unanimous Jaccard | 0.857 | **0.750** |
      | **per-risk verdict flicker** | **0** | **0** |
      | recall vs the six SQL rules | 1.00 every run | 1.00 every run |

      **Take the right-hand column.** The first sample's 0.96 was survivorship
      bias: the three runs it dropped are precisely the ones that disagreed,
      so excluding them measured the detector's good days. Once the parse fix
      (below) let those runs complete, the honest figure is **0.882 mean
      pairwise / 0.750 unanimous over ten of ten runs** — still far better
      than 7.4's 0.67, which was a two-run artefact, and arrived at without
      discarding evidence.

      Verdict flicker is **zero in both samples, twenty runs total**, which is
      the condition the task set and the one `README.md` §7 cares about. No
      open risk ever changed verdict.

      Churn is confined to two keys, and neither is a rule-covered type:
      `unresolved_objection` 0.6, `single_threaded` 0.2, every other key 1.0.
      `single_threaded` appearing in 2 of 10 runs is worth a look on its own —
      the deterministic rule stays silent on this deal, so those are AI-only
      proposals, i.e. a probable false positive at 20%. Recall against the SQL
      rules never dropped below 1.00, so the instability is all additive: the
      model never loses a real risk, it occasionally invents a marginal one.

      Two metrics were added rather than reusing `risk_stability`, which
      compares exactly two runs: `metrics.risk_stability_n` (mean pairwise and
      unanimous Jaccard, plus a per-key appearance rate, because "one key at
      0.86" and "seven keys at 0.98" are different product problems) and
      `metrics.verdict_flicker` (a changed verdict is a flip; an absent one is
      reported separately, since "changed its mind" and "said nothing" have
      different fixes). Failed runs are excluded from both and reported on
      their own: folding an empty panel in as a run that agreed with nothing
      would drag Jaccard toward zero and bury which key actually wobbles.

- [ ] **11.10 `extract@3`.** **One** measured regression from the `@2` prompt,
      not two — this item's premise was half wrong and the correction is the
      first thing it bought:

      **objection recall 0/3** — real, and confirmed by re-scoring the kept
      baseline. `@2`'s "a condition is `decision_criteria`, not an objection"
      pushed every objection out of the type, and `objection` feeds
      `unresolved_objection`.

      ~~commitment recall 1/3~~ — **not a regression. It is 3/3.** `1/3` was
      `extract@1`'s figure; `@2`'s precedence list is what fixed it, and
      `recorded/extract-baseline-gpt-oss-120b.json`'s own comment says so
      ("fixed commitment recall (1/3 -> 3/3), at the cost of objection recall
      (1/3 -> 0/3)"). The claim survived in this task because the replay path
      could not read that recording — see *Found during implementation*. Now
      pinned by a test so it cannot drift back.

      Measured `@2` baseline, reproduced from the recording with no API calls:
      recall 0.73, f1 0.54, precision 0.43; by type `budget` 3/3,
      `commitment` 3/3, `competitor` 1/1, `deadline` 2/3, `decision_criteria`
      4/5, **`objection` 0/3**, `requirement` 4/5, `stakeholder` 2/3.

      One of the three objection labels is itself wrong: `n3` is "My team has
      signed off." — s2's objection being *resolved*, not an objection. No
      prompt should extract it as a concern, so **objection recall against the
      current labels is capped at 2/3**. Fixing the label is out of this
      task's scope; hill-climbing against it is not an option, so the target
      is restated accordingly.
      *Done when:* objection recall ≥ 2/3 excluding the mislabelled `n3`, with
      no regression against `@2`'s f1 of 0.54, measured on the existing
      fixtures.

      **Still open, and the done-when above is now known to be unmeasurable as
      written.** A candidate prompt exists and has been run; the comparison it
      asks for cannot be made at n=1. Measured 2026-10-05, primary model
      throughout:

      | arm | runs | recall | f1 | objection | deadline |
      |---|---|---|---|---|---|
      | `@2` recorded (Oct 1) | 1 | 0.73 | 0.54 | 0/3 | 2/3 |
      | `@2` fresh | 1 | 0.81 | 0.58 | 1/3 | 3/3 |
      | `@2` fresh | 1 | 0.77 | 0.55 | 1/3 | — |
      | `@2` fresh | 1 | 0.77 | 0.58 | 1/3 | — |
      | `@2` fresh | 1 | 0.69 | 0.51 | 0/3 | — |
      | **`@3`** | 1 | 0.73 | 0.56 | **2/3** | 1/3 |

      `@2` alone spans **f1 0.51–0.58 and objection 0/3–1/3 with no prompt
      change at all.** `@3`'s f1 of 0.56 sits inside that band, so the single
      comparison that looked like a win (0.56 vs the recorded 0.54) was
      measuring the dice. Against a same-day `@2` it is a *loss* (0.56 vs
      0.58), and it trades two objections (`s1`, `s2` — both found) for two
      other facts in the same transcript (`s3` requirement, `s6` deadline),
      with deadline recall falling 3/3 → 1/3. All four swapped labels are in
      `02-security-review`, which suggests a roughly fixed per-window output
      budget being reallocated rather than a genuine comprehension change.

      Also corrected: **`@2`'s objection recall is not 0/3.** It is 1/3 in four
      of five fresh runs. The 0/3 in the kept recording is one sample, and this
      task's premise rested on it.

      **What is needed to close this:** n≈4 per arm, which the harness now
      supports (`--repeat N`, with `aggregate` reporting per-label hit rates
      rather than a single verdict). Not run — at 8,000 TPM on the free tier
      one corpus pass is ~26K tokens and ~3 min of pure rate-limit wait, so
      eight passes is ~25 min, and the measurement was stopped by choice rather
      than spend it. **`@3` must not ship on the evidence above**; `extract`
      stays at `@2` and `prompts/extract_v3.py` stays registered under its own
      name, unused by the pipeline.

- [ ] **11.11 Task 1.5's three-way model comparison.** Still open from Phase 1.
      `openai/gpt-oss-120b` is the primary by assertion, not measurement, and
      `groq_model_challenger` exists in settings for this.
      *Done when:* primary, cheap and challenger are scored on the same fixtures
      and the choice is recorded with its numbers.

      **Still open. Unblocked but not run.** The blocker was that
      `run_extraction_eval --live` did not exist — it raised
      `SystemExit("live scoring is wired in task 3.6")` and 3.6 shipped without
      wiring it, so both this task and 1.5 were waiting on a code path
      everything assumed was there. That path now exists and works against the
      real extractor, with `role` and `prompt` overrides on
      `extract.extract_window`/`extract_facts`.

      Not run because 11.10 established that **one run per arm cannot separate
      an arm from the noise** — `@2` spans f1 0.51–0.58 unchanged — and a
      three-way comparison at n=1 would produce a ranking of three dice rolls
      and record it as a model choice. That is worse than leaving the primary
      unmeasured, because it would look settled.

      **To close:** `--live --model all --repeat 4` (12 corpus passes, ~40 min
      at 8,000 TPM, and it includes the Preview-tier challenger at 5-6x the
      primary's price). Until then `openai/gpt-oss-120b` remains the primary by
      assertion, which is the status quo and is now at least labelled.

### Order

11.1 → 11.2 → (11.3, 11.4, 11.7 in parallel) → 11.5 → 11.6 → 11.8 → 11.9 →
11.10 → 11.11.

11.1–11.4 and 11.7 are plumbing against verified gaps and should land together.
11.8 waits on nothing technical but is better placed after 11.2, since both
write `recommendations` and 11.2 is the one with a reconciliation rationale
behind it. 11.10 is last of the substantive items because it is hill-climbing
with no fixed finish line, and 11.9 should precede it so there is a stability
baseline to regress against.

### What Phase 11 is explicitly not

| Not doing | Why |
|---|---|
| `users` / auth | single-user MVP; `ae_display_name` in settings stays the AE identity, and `origin` carries the only attribution distinction the AI layer needs |
| Langfuse / LangSmith / `ai_runs` | structured per-run logging (0.7) plus 11.5 is the whole observability story; accepted cost is no trace history |
| Anything in Out of scope, below | unchanged by this phase, except that 11.8 takes the one chat write tool that table always allowed |

---

## Out of scope

| Item | Reason |
|---|---|
| Embeddings and similarity retrieval | `README.md` §7 — corpus too small, second vendor not earned. Worth stating plainly: pgvector gives a column type, distance operators and index types and **no embedding model**, so `document_chunks.embedding` is NULL on every row and its HNSW index is empty. Groq serves no embeddings endpoint, so `settings.embedding_model` (an OpenAI name) is unreachable as well as unread. Retrieval is lexical `ilike` in one of nine chat tools, and `pg_trgm` covers the one place relevance was load-bearing (`0016`). The declarations stay for a later backfill |
| Cross-deal prioritization | not until single-deal detection is trusted |
| Document classification at upload | five values in a dropdown; ingest is synchronous by design |
| Chat write tools beyond `propose_task` | the read path is now trusted, so `propose_task` moves **in** as task 11.8; nothing further |
| Fine-tuning | the dismissal feedback loop covers adaptation |
| Langfuse / LangSmith | **decided against for the MVP.** `client.py`'s structured per-run logging (0.7) plus `GET /system/ai-status` (11.5) is the observability story; the accepted cost is no per-run history and no trace tree |
| `ai_runs` table | same ground as above — tracing is logs plus one status endpoint, not a table (`docs/schema/README.md` §10) |
| `users` table / auth | single-user MVP. `ae_display_name` in settings is the AE identity, and `origin` already distinguishes AI writes from human ones |
| LangGraph checkpointers | pipeline and chat are already durable in Postgres (`README.md` §7) |

---

## Found during implementation

- **Supersession recall was bounded by recency-ordered candidates.**
  `reconcile.supersede_facts` chose which older facts a new fact might
  replace with `ORDER BY extracted_at DESC LIMIT 10` -- recency standing in
  for relevance, which holds only while a deal has fewer than ten accepted
  facts of one `fact_type`. Past that the one older
  fact genuinely about the same subject can fall outside the window, the model
  never sees it, and it stays `accepted` forever while a newer fact
  contradicts it -- **both then render on the panel, each with a valid
  citation.** Nothing is wrong with the evidence; the shortlist was wrong.
  Fixed: `0016_fact_content_trgm` adds a GIN `gin_trgm_ops` index on
  `extracted_facts.content` (no `CREATE EXTENSION` -- `0010` owns `pg_trgm`,
  and two revisions each believing they may drop it is its own trap), and the
  shortlist is now selected **per new fact**, `ORDER BY similarity(content,
  :new_content) DESC LIMIT 10`, above a floor of `SIMILARITY_FLOOR = 0.15`.
  The call count cannot rise -- it was already one per new fact -- and falls
  whenever a shortlist comes back empty.
  **The floor is measured, not guessed.** On real fact pairs a verbatim
  revision scores ~0.57, a revision reworded end to end ~0.19, two facts
  sharing only deal vocabulary ~0.13, unrelated noise ~0.08. The pg_trgm
  default of 0.3 would discard the reworded revision, which is the exact case
  this function exists for. The gap between the weakest keep (0.19) and the
  strongest drop (0.13) is narrow, so the floor prunes the long tail and the
  *ordering* does the real work; the model still answers null when unsure.
  `reconcile_commitments` was unbounded and is now capped at 10, ordered by
  `greatest(similarity(description, ...))` over the new facts with the due
  date as tiebreak, and **deliberately has no floor**: a commitment and the
  fact that satisfies it routinely share almost no wording ("send the SOC 2
  report" against "Dana circulated the security pack"), so a floor there would
  discard the pairs worth catching.
  **Trigram rather than embeddings, deliberately** -- `pg_trgm` is already
  installed, there is no provider to call (Groq has no embeddings endpoint),
  no dimension to pin, no second code path to keep working, and the existing
  test harness can measure it. Embeddings only if a measured need appears.
  `tests/test_reconcile.py` is new: 4 tests, and the one that exercises the
  gap failed before the change and passes after.

- **One stray array element fails the whole detect run, and it happens 30% of
  the time.** Task 11.9's ten-run sample lost 3 runs, in what looked like two
  different ways and is one defect: the model emits a spurious non-object
  element inside the `risks` array — a bare `""` in runs 4 and 5, a truncated
  `"{""risk_type"` in run 1 — in between otherwise valid risk objects. Groq's
  schema enforcement rejects one shape up front (`400 json_validate_failed`)
  and the other arrives and fails Pydantic (`Failed to parse DetectionOut`),
  which is why it reads as two problems in the log. Either way **two or three
  perfectly good risks are discarded with it and the panel renders nothing.**
  Not fixed — 11.9's remit was to measure and write nothing — but it dominates
  every other stability number on this page, and the same client path carries
  `ExtractionResult.facts`, so it can corrupt 11.10's and 11.11's measurements
  the same way. **Fixed** — tolerance at the parse boundary, decided
  deliberately over the alternative of retrying: `client._strip_stray_elements`
  drops an element only when its siblings are objects and it is not, which is
  exactly the observed shape, and `_salvage` re-validates what is left.
  Covers both arrival shapes, since the rejected text lives in a different
  place in each: `error.failed_generation` on the 400, and the message content
  when the completion arrives and fails Pydantic. Deliberately narrow — a list
  of scalars is untouched, an object is never repaired field by field, a
  missing required field still fails, unparseable JSON is not reconstructed,
  and more than `_MAX_SALVAGED_DROPS` strays is treated as a lost response
  rather than averaged into data. Logged at WARNING with outcome
  `ok_salvaged` and the dropped paths, because the accepted cost here is a
  quietly partial result and it must at least be countable.
  Re-measured: **0 failures in 10**, recall still 1.00, flicker still 0.
  14 tests in `tests/test_salvage.py`, built from the two payloads actually
  observed rather than invented ones.

- **Run-to-run variance is larger than the prompt change being measured, so
  every single-run extraction score in this file is weaker evidence than it
  looks.** Five fresh runs of the *unchanged* `extract@2` span **f1 0.51–0.58,
  recall 0.69–0.81, objection recall 0/3–1/3**. The first `@3` comparison
  therefore "won" (0.56 vs the recorded 0.54) purely by landing high in `@2`'s
  own band, and a same-day `@2` at 0.58 reversed the verdict. This also means
  the kept baseline's 0.73/0.54 is *a sample*, not *the* figure for `@2` — and
  task 11.10 spent its existence chasing an objection recall of 0/3 that is
  1/3 in four runs out of five.
  Mitigated rather than solved: `--repeat N` plus `aggregate` now report mean,
  min and max per arm and a **per-label hit rate**, because "found in 1 of 4
  runs" and "found in 4 of 4" are different claims that a single run flattens
  into "found". The corpus is 44 labels over three transcripts, so a handful of
  reallocated facts moves f1 by several points; the honest reading is that this
  fixture set can detect a *large* prompt effect and nothing subtler.

- **The same `MissingGreenlet` trap twice, both times from a rollback between
  iterations.** `session.rollback()` expires every loaded ORM attribute, so the
  next loop iteration's first attribute read lazy-loads — and outside a
  greenlet that raises rather than refetching. It cost runs 2-10 of the first
  stability sample (fixed by reading `deal.id` into a local before the loop)
  and then runs 2-3 of the first `--repeat 3`, where the expired objects were
  the *chunks*, so all three transcripts failed and the aggregate reported
  recall 0.26 for a prompt that scores 0.73. Fixed properly the second time:
  the eval loads chunks as plain `DetachedChunk` namedtuples and the rollback
  is gone, since the extractor takes `Sequence[Any]` and never queries through
  them. The general lesson is that an ORM object held across an iteration
  boundary in a loop that commits or rolls back is a latent failure, and the
  symptom (a zeroed metric) looks exactly like a model regression.

- **The eval replay scored the wrong file, in silence, and a wrong number
  outlived it in this document.** `run_extraction_eval.replay` required
  `predicted_facts` at the top level. The kept baseline
  (`extract-baseline-gpt-oss-120b.json`) stores `by_transcript` instead, so it
  was skipped with no message, and the only recording scored was
  `extraction-discovery-example.json` — a hand-designed three-fact fixture for
  the *metric* tests (one exact hit, one shifted span, one wrong type, one
  invention), pooled against all three transcripts' labels for a corpus recall
  of **0.08**. The visible cost: task 11.10 carried "commitment recall 1/3" as
  an open regression when the baseline had said 3/3 since Phase 3, because
  nothing could read the file that would have contradicted it. Fixed — replay
  reads both shapes, an unreadable recording now raises instead of being
  dropped, the designed fixture is flagged `_not_a_model_run`, and
  `tests/eval/test_replay_shapes.py` pins the baseline's numbers including the
  3/3 that was being re-litigated. The lesson is 7.4's again, one level out: a
  harness that silently measures the wrong input is worse than one that
  crashes.

- **`run_extraction_eval --live` never existed.** It raised
  `SystemExit("live scoring is wired in task 3.6, against the real
  extractor")`, and 3.6 shipped without wiring it — so tasks 1.5/11.11 and
  11.10 were both blocked on a code path everything assumed was there.
  Implemented against the real extractor: `extract.extract_window` and
  `extract_facts` now take `role` and `prompt` overrides (defaulting to
  exactly what the pipeline did before), and the runner reads the transcripts'
  chunks out of the database rather than re-chunking, because every label's
  offsets point into those exact rows.

- **The first shadow run reported recall 0.0, and the metric was the thing that
  was broken.** The model does not re-propose a risk that is already open -- it
  returns a `still_present` verdict for it, which is exactly what the prompt
  asks for. Scoring only `proposed` against the SQL rules therefore counted
  zero overlap on a run where every rule-detected risk had been affirmed.
  Corrected to `proposed + affirmed`, which gives 1.00, and pinned with a test.
  The lesson is the one the shadow run exists to teach: it measures the
  measurement as well as the detector, and the detector was right.

- **`reasoning_effort="high"` makes detection fail outright.** Measured with a
  three-way probe: at `high` with any cap the model spends its output budget
  reasoning and never emits the JSON (`400 json_validate_failed`, **empty**
  `failed_generation`); at `low` it found two of the three risks the SQL rules
  confirm; at `medium` it found all three in ~2.6K output tokens. Set to
  `medium`. The lesson generalises -- on a reasoning model the effort setting
  and the output cap are *one* budget, and the failure when they collide looks
  like a schema error.

- **Every per-stage `max_tokens` was sized against the wrong TPM figure, and I
  did not revise them when I corrected it.** The caps were set when
  `groq_tokens_per_minute` said 250,000. After correcting that to the real
  8,000, `ai_max_tokens_detect` and `ai_max_tokens_extract` were still 8192 —
  **102% of the entire per-minute budget**, so a single request could never fit
  in a fresh minute. Extraction survived it because its actual output (3.1-4.3K)
  stayed under the cap; detection did not, and failed as
  `400 json_validate_failed` with an **empty** `failed_generation` — the shape
  that means nothing was produced, not that the schema was wrong. Resized
  against observed output, with the constraint written down: a per-request cap
  must leave room for its own input inside the TPM ceiling.

- **Migration 0013 broke the deterministic detector, silently until run.**
  `detect.run` upserts with
  `on_conflict_do_update(index_elements=[deal_id, risk_type], index_where=...)`,
  inferring `uq_risks_deal_id_risk_type_open` — which 0013 drops and replaces
  with `uq_risks_open_key` on `(deal_id, risk_type, risk_key)`. Index inference
  then fails outright: there is no longer a unique index on the pair alone.
  Caught by running the four SQL rules after migrating, not by any test. Fixed
  by adding `risk_key` to both the inserted values and `index_elements`.

- **Reconciliation has nowhere to put its proposal.** The design said a
  proposed commitment status change lands as a `recommendation`, and it does
  not fit: every `ActionType` names an action to *take* — `send_document`,
  `engage_stakeholder` — while "this commitment now looks satisfied" is a
  proposed **data correction**. Filing it under the nearest action would
  corrupt the action-type and `dismissal_reason` distributions, two of the few
  signals here that are not self-reported. Stage 7 therefore returns and logs
  the proposals and writes nothing; closing a promise nobody kept is the error
  that matters. Needs an `ActionType` value (a one-line CHECK swap, since the
  set is `text + CHECK` for exactly this) or a surface of its own.

- **Gate 1 is stricter than the person writing its test.** The first
  "supported" case in `verify_ai_gate1.py` claimed *"auditors require twelve
  months of log retention"* against the quote *"Our auditors want twelve
  months."* — and the validator returned `partial`, because the quote says
  nothing about log retention. It was right and the expectation was wrong. That
  is the behaviour the gate exists for, measured.

- **`anything_survived` routed to `END`, which skipped `finalize`.** A meeting
  whose facts all failed Gate 0 would never have `analysis_status` set, so the
  worker would find it `queued` and re-run it on every poll, forever. It now
  routes to `finalize`. The worker keeps a floor underneath, because
  `finalize` is a *degradable* stage: if it raises, `guarded` swallows it and
  nothing would set the status at all.

- **The first graph conversion silently dropped the degradation rule, and my
  own note claimed otherwise.** The sequential runner caught a stage failure
  and, for a degradable stage, recorded it and carried on — "a failed summary
  must not discard the facts that already landed". A raw LangGraph node has no
  such notion: anything it raises propagates out of `ainvoke` and rolls back
  the transaction, twenty verified facts included. Task 3.7's note said
  "identical outcomes to the sequential runner", which was true of the happy
  path I tested and false of failure behaviour — I should not have written it
  that broadly.
  `graph.guarded()` restores it by consulting `stage_registry`. It cost nothing
  while stages 5-12 were no-ops that could not fail, which is exactly why it
  survived; Gate 1 is stage 5 and the first thing that genuinely can.

- **Keeping both implementations was not defensible.** The plan was to retain
  the sequential runner as a test oracle. That only works if the two paths
  *share* stage logic and differ in orchestration — and they did not: all five
  implemented stages existed as two independent bodies, with two separate
  `gate0.check_claim` loops. A fix to one would not reach the other, and Phase
  4's Gate 1 would have been written twice. Removed, leaving one copy of each
  stage in `stages.py` and the wiring in `graph.py`.

- **`graph.NODES` exists so failure is injectable.** Two rewritten tests passed
  while asserting nothing, for two different reasons: the test meeting has no
  transcript, so `fan_out` routes past `extract_window` entirely, and the eight
  late stages are closures rather than module attributes, so patching
  `graph.<name>` found nothing to replace. A node registry gives one patchable
  seam; `monkeypatch.setitem` keeps it from leaking between tests.

- **`extract@2` traded objection recall for everything else.** The precedence
  list fixed what it was aimed at and broke one thing:

  | | `extract@1` | `extract@2` |
  |---|---|---|
  | recall · precision · f1 | 0.65 · 0.33 · 0.44 | **0.73 · 0.43 · 0.54** |
  | facts reported | 52 | 44 (less over-generation) |
  | payload errors | 5 | **1** |
  | commitment recall | 1/3 | **3/3** |
  | decision_criteria | 3/5 | **4/5** |
  | **objection recall** | 1/3 | **0/3** ← regression |

  "A condition is decision_criteria, not an objection" pushed too hard: every
  objection now lands elsewhere. One of the three labels is my own fault (`n3`
  labels *"My team has signed off"* as an objection, which is a resolution, not
  one), so the real figure is nearer 0/2 — still bad, and `objection` feeds
  `unresolved_objection`. `extract@3` should strengthen branch 5 rather than
  weaken branch 2. Kept @2 because the net is clearly better and the
  regression is legible.

- **Self-reported `confidence` is worthless from this model.** All 20 written
  facts came back at exactly `1.00`, across six fact types, including the ones
  Gate 0 would have rejected had they been paraphrased. This is the
  poorly-calibrated signal `docs/schema/README.md` section 5 warns about,
  measured: it validates never gating on confidence, and it means the
  `confidence × verdict` matrix will be degenerate until Gate 1 supplies the
  other axis. Consider dropping `confidence` from the wire schema entirely —
  it costs output tokens and carries no information.

- **Throttling is visible in the latency, not just the logs.** With the TPM
  ceiling correct, the second extraction window of a transcript waited 77
  seconds for bucket refill (9.5s → 77s for a comparable call). Working as
  designed, but it makes a full-corpus eval run a multi-minute operation on the
  free tier — worth knowing before treating the eval loop as interactive.

- **The configured TPM was wrong by 31x, and that silently disabled the
  governor.** The published table quotes the **Developer** plan at 250K TPM;
  this account is on the free `on_demand` tier, where `openai/gpt-oss-120b` is
  **8,000 TPM**. Set 31x too high the bucket never throttled, and the first
  extraction run died with `429 ... Limit 8000, Used 7835` part-way through the
  second transcript. The mechanism was fine — the number was wrong, which is
  the more dangerous failure because everything looks healthy until it isn't.
  Now 8,000 TPM / 30 RPM / concurrency 2, and the re-run completed with **zero
  retries**. The real limit is on every response as
  `x-ratelimit-limit-tokens`; reading it from there would delete this setting
  and is the obvious next improvement.

- **A flat backoff retried into the same exhausted window.** Groq's 429 says
  exactly when it reopens ("Please try again in 13.3575s") and the retry slept
  one second, burning the attempt. `_retry_after()` now prefers the
  `retry-after` header, falls back to parsing the message, and caps at 60s —
  the header and the message are both needed because which one exists depends
  on the SDK version.

- **The fact taxonomy genuinely overlaps, and it costs recall.** Ten
  predictions cite a *labelled span* under a *different* `fact_type`. The
  confusions are systematic, not random: `objection`/`requirement` →
  `decision_criteria` (4), `deadline` ↔ `commitment` (4). "We're not signing
  anything until my team has run a load test" is defensibly a requirement, an
  objection and a decision criterion, and a commitment with a date in it is
  defensibly one fact or two.
  This is not only a scoring problem — `decision_criteria` and `objection`
  feed **different** downstream rules, so a statement filed under the wrong one
  reaches the wrong detector. Three ways out, and it is a product call:
  (a) give the prompt explicit precedence rules ("a condition of signing is
  `decision_criteria`, even when phrased as a refusal");
  (b) collapse the overlapping types in the enum;
  (c) let a statement carry more than one `fact_type`.
  Recorded rather than chosen.

- **Fixture labels are not exhaustive, so precision is not measurable.** They
  were written as the planted cases plus notable facts. The extractor found 52
  facts against 26 labels and most of the excess is real — "the audit is
  scheduled for Q4", "finance will need to sign off". Either label the corpus
  exhaustively, or report precision only against an adjudicated sample.
  Recall is unaffected and remains the number to hill-climb on.

- **Commitment recall is 1/3, and commitments are load-bearing.** They feed
  `missed_commitment` and the Phase 6 reconciliation. The model tends to split
  "Maya will send the SOC 2 by the seventh of August" into a `commitment`
  without the date and a `deadline` with it — arguably more granular than the
  label, but it means neither row alone can become a `commitments` record with
  a `due_date`. Worth fixing in the prompt before Phase 6 depends on it.

- **The test suite must be *forced* offline, not assumed to be.** One test
  asserted the tiebreak was a no-op "when AI is disabled" and passed for the
  wrong reason: it was reading `AI_ENABLED` from `backend/.env`. The moment a
  key was configured and the flag flipped, that test started making a real
  paid API call during `pytest`. An autouse `_offline` fixture in
  `conftest.py` now sets `settings.ai_enabled = False` for every test, so the
  suite's behaviour cannot depend on the developer's environment and no test
  can spend money. Verified by running the suite with `AI_ENABLED=true` in the
  environment: 51 pass either way. Tests that need a model stub
  `client.structured` directly, which never reaches `chat_model` and so is
  unaffected; the live checks live in `verify_ai_*.py` and are run by hand.

- **`single_threaded` counts only *resolved* external attendees**
  (`contact_id IS NOT NULL`, `detect.py`), and it fires at **exactly one**. So
  the roster's reluctance to link and this filter compound: a deal with one
  known contact plus several unresolved speakers reads as single-threaded even
  though four people are in the room. On the fixture corpus Tom Alvarez attends
  all three meetings and contributes nothing to breadth. Before Phase 2 the
  rule was *inert* rather than wrong — the table was empty, so `len(people)`
  was 0 and it never fired. Now it has real input, and the question is whether
  breadth should count distinct `raw_name` for externals instead of distinct
  `contact_id`. A product call, not a bug fix, so it is recorded rather than
  changed.

- **Migrations renumbered.** Phase 2 needed `pg_trgm` before any of the planned
  revisions, so `0010` is `pg_trgm` and the planned ones shifted up one:
  `0011_meeting_analysis_origin`, `0012_detector_provenance`,
  `0013_open_risk_taxonomy`, `0014_analysis_triggers`.

- **`PipelineState` moved to `app/ai/state.py`.** `pipeline.py` declares the
  graph and so must name the stage functions, while `stages.py` needs the state
  type — a cycle. The first attempt deferred the import inside a closure, which
  silently failed (`name 'stages' is not defined`, surfaced as a stage-0 error
  on every meeting). Extracting the type is the fix; the lazy indirection is
  gone.

- **Recorded eval runs are coupled to chunking.** The example recording's spans
  are chunk coordinates, so re-cutting the fixtures invalidated three metric
  tests. `tests/eval/recorded/regenerate.py` rebuilds it from the current
  labels, preserving the designed story. In production this cannot happen:
  chunks are immutable and re-ingest is delete-and-re-upload.

- **Pydantic reserves the `model_` prefix.** `model_primary` and friends emit a
  protected-namespace warning on a class that also declares `model_config`, so
  the settings are `groq_model_primary` / `_cheap` / `_challenger`.

- **`meetings` has no error column.** A critical stage failure sets
  `analysis_status='failed'` and the reason survives only in the worker log, so
  the Analyzer screen can say *that* it failed but not *why*. Add
  `analysis_error text` to migration `0011_meeting_analysis_origin` (task 5.1).

- **The pipeline is a sequential runner, not a `StateGraph`.** All thirteen
  stages are declared with the critical/degradable split, but a graph whose
  nodes are every one a pass-through is structure with nothing flowing through
  it. Convert in Phase 3, when extraction and Gate 0 give `PipelineState` its
  first real contents.

- **Worker crash semantics chosen: hold the transaction.** The run commits at
  the end, so a crash leaves the row `queued` and self-healing — no reaper, no
  lease. The cost is that `running` is never visible to a polling UI. A
  *handled* critical-stage failure is different and is written as `failed` in a
  second transaction, because it will not succeed on retry.

- **LangChain is installed on the host venv, and it is the 0.x line.**
  `langchain` 0.3.30, `langchain-core` 0.3.86, `langchain-groq` 0.3.8,
  `langgraph` 0.6.11. Narrower a gap than 0.9 warned: `method="json_schema"`,
  `include_raw` and `reasoning_effort` are all present. Installing it also
  downgraded `async-timeout` 5.0.1 -> 4.0.3, which the app tolerates.

- **0.3.8 cannot request Groq's `strict` mode**, so the host venv gets
  best-effort structured output rather than constrained decoding: it builds
  `response_format` without `"strict": true` and exposes no parameter to add
  it. `client.py` probes `with_structured_output` for the argument and passes it
  only where present, so the container is guaranteed and the host is not. The
  consequence to remember when reading extraction results locally: **the
  Pydantic validation is the only guarantee there**, and a schema-shaped
  response is not assured. `README.md` §7 has the table.

- **~~`pytest` is not installed.~~** Done in 1.3: `pytest` + `pytest-asyncio`,
  `backend/pytest.ini` (`pythonpath = .`, `asyncio_mode = auto`), and the Phase
  0 database checks absorbed into `test_worker.py` / `test_activity.py`.

- **The engine must be disposed between tests.** `app.db.session.engine` is a
  module-level singleton whose pool binds each asyncpg connection to the loop
  that first used it, and pytest-asyncio gives every test its own loop — so a
  reused pooled connection raises "attached to a different loop". An autouse
  fixture in `conftest.py` disposes it per test. Same trap made `TestClient`
  unusable for the HTTP checks, which is why those run against a real uvicorn
  subprocess.

- **Chunk overlap means a snippet can be citable from two chunks.** `s7`
  appears in two, because `chunk_overlap_chars` is 150. Either citation passes
  Gate 0; `resolve_labels.py` records the lowest `chunk_index` so the label is
  deterministic, and counts the rest in `appears_in_chunks`. It also detects the
  case that would matter — a snippet present in the file but in **no** single
  chunk, which no extractor could cite within one window. None of the 26 hit
  it, but the check stays.
