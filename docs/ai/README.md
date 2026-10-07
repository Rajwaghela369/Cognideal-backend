# CogniDeal — AI Layer

The model layer over the schema in `docs/schema/README.md` and the HTTP layer in
`docs/api/README.md`. Task list: `TASKS.md` in this directory.

The schema decides most of this document. Five tables hold things a model
*asserts* — `extracted_facts`, `commitments`, `risks`, `recommendations`,
`chat_messages` — and `claim_evidence` already demands that each one point at a
locatable source. So the AI layer is not an open design space: it is whatever
fills those tables without breaking the contract.

---

## 1. Design principles

**1. A model is used only where a rule cannot do the job.** Six of the ten
`risk_type` values are joins over existing tables and can never hallucinate,
including the clock-driven `missed_commitment` and `gone_quiet`. Work that
SQL can do stays in SQL — not to save money (see §7, the money is trivial) but
because a deterministic rule cannot flicker, cannot be injected, and gives the
model pass something to be measured against.

**2. Claims and their evidence are produced in one structured output.** Never
generate a claim and then retrieve evidence for it. Post-hoc retrieval always
finds something plausible, which is how confident nonsense is manufactured.

**3. The model never queries the database.** Every task gets a context
assembled in Python, where each line already carries the citation handle that
backs it. The model references evidence *by handle from a closed set*, so it is
structurally incapable of citing a chunk it was not given.

**4. Grounding is enforced in the service layer, not in a prompt.** Gate 0 runs
inside the insert. A claim with zero surviving evidence links is a rejected
write, not a warning.

**5. Identity is always a key, never prose.** Every upsertable assertion has a
deterministic key — `(deal_id, risk_type, risk_key)`, `(deal_id, source_risk_id)`,
`meeting_id`. Free text is allowed in `title`/`description`/`rationale` and
nowhere that a unique index depends on.

**6. Nothing a model writes enters Layer A without a human.** The path is
`fact → [human accepts] → commitment/task`, and `risk → recommendation →
[human accepts] → task`. The model proposes; the person decides.

**7. Degrade per stage, not per run.** A failed summary must not discard the
facts that already landed.

---

## 2. Task inventory

Eleven producers. Only three are genuinely AI-hard; the rest are composition,
classification or matching.

| # | Task | Writes to | Shape | Model |
|---|---|---|---|---|
| 1 | **Transcript parse** — speaker labels, chunk metadata | `document_chunks.metadata` | deterministic parser | — |
| 2 | **Roster + entity resolution** — labels → attendees, name → contact | `meeting_attendees` | `pg_trgm` + LLM tiebreak on the residue | `openai/gpt-oss-20b` |
| 3 | **Extraction** — text → typed facts with exact spans | `extracted_facts`, `evidence`, `claim_evidence` | N parallel structured calls, one per chunk window | `openai/gpt-oss-120b` |
| 4 | **Gate 0** — span integrity | `claim_evidence.verification_status` | pure Python | — |
| 5 | **Gate 1** — entailment, starved context | `claim_validations` | 1 call per claim, no tools | `openai/gpt-oss-120b` |
| 6 | **Commitment reconciliation** — was a promise kept? | proposal → `recommendations` | 1 call, candidates prefiltered | `openai/gpt-oss-120b` |
| 7 | **Supersession** — does new evidence contradict a live fact? | `extracted_facts.status='superseded'` | 1 call per fact_type group | `openai/gpt-oss-120b` |
| 8 | **Meeting synthesis** — summary from the *facts*; sentiment from the *transcript* | `meetings.summary`, `.sentiment`, `.analyzed_at` | 2 calls | `gpt-oss-120b` / `gpt-oss-20b` |
| 9 | **Risk + recommendation detection** | `risks`, `recommendations`, `evidence` | 1 call per deal over a cited dossier | `openai/gpt-oss-120b` |
| 10 | **Meeting brief** | `meeting_briefs` | 1 call over prefetched rows | `openai/gpt-oss-120b` |
| 11 | **Chat** | `chat_messages` + citations | agent loop, read-only tools | `openai/gpt-oss-120b` |
| — | Chat session titles | `chat_sessions.title` | 1 call, fire-and-forget | `openai/gpt-oss-20b` |
| — | Embeddings | `document_chunks.embedding` | deferred, see §7 | none — see §7 |

**Explicitly not AI:** `documents.source_type` / `title` / `occurred_at` at
upload. Ingest is synchronous by design specifically to keep external calls out
of the request path, and there are five source types in a dropdown. A human
picks.

---

## 3. Execution model

**One workflow, two short pipelines, six single-call tasks, one conversational
agent. Zero autonomous agents.**

Two tests, applied in order:

1. *Who picks the next step?* Your code — a workflow. The model — an agent.
2. *Does step N+1 consume what step N produced?* If not, it is not a workflow
   however many calls it makes.

The second test is the one that gets miscounted. Gate 1 runs fifteen validator
calls for one meeting, but no state flows between them: that is a `gather` over
fifteen single calls, not a workflow. Only meeting analysis has steps that feed
each other.

| Task | Shape | Orchestration |
|---|---|---|
| **Meeting analysis** (§4) | **workflow** — 13 stages | **LangGraph `StateGraph`** |
| Risk + recommendation detection (§5) | Python → one call → Python | a function |
| Gate 1 validation | one call, fanned out N times | `gather` |
| Reconciliation · supersession · summary · sentiment · brief · titles | single call | none |
| Roster / entity resolution | SQL + occasional cheap call | none |
| **Chat** (§6) | **agent** | the model orchestrates |

Expressing any of the single-call tasks as a graph would be a graph with one
model node and some Python either side — all of the abstraction and none of the
payoff.

| Task | Why not an agent |
|---|---|
| Extraction | fixed pipeline; giving it retrieval violates principle 2 |
| Validator | a starved context is the entire point; tools would defeat it |
| Synthesis / brief / detection | context is prefetched in Python and reproducible |

### The worker

`POST /deals/{id}/meetings/{id}/analysis` already sets `queued` and nothing
consumes it. No Redis, no Celery — compose has neither. One process polling:

```sql
SELECT ... FROM meetings WHERE analysis_status = 'queued'
ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
```

At-least-once delivery, crash-safe, no new infrastructure. The same loop carries
three poll queries — the meeting queue above, dirty deals, and the nightly sweep
(see Triggers below) — so there is no scheduler and no second deployment
artifact. `analysis_status`
moves `queued → running → complete | failed`, where **`failed` means stages 0–4
failed**; a later stage failing still yields `complete` with a per-stage error
logged, because the facts landed.


### Triggers: events and the clock

Two independent inputs, and neither substitutes for the other. **Events say
*this deal* changed; the clock says *time* changed.**

The clock is not optional, because four risk types and all of Gate 2 are
functions of elapsed time rather than of writes:

| Risk type | What makes it true |
|---|---|
| `stalled_stage` | dwell days crossing a threshold |
| `close_date_at_risk` | the close date getting nearer |
| `gone_quiet` | `last_activity_at` receding |
| `missed_commitment` | `due_date` passing with `status='pending'` |

Nothing writes a row when a commitment becomes overdue. The deal sits untouched
and silently becomes riskier, so only a sweep catches it — the whole point of
`stalled_stage` is that nothing has been happening.

Events are not optional either. `docs/schema/README.md` promises that a claim
self-invalidates — "change the deal's stage and the citation goes
`value_drifted` without anyone having to notice" — and that is only true if
something re-resolves the `record_ref`. Waiting for the nightly sweep means the
UI renders a stale citation as current for up to a day, which is the exact
failure the product thesis exists to prevent.

**Events invalidate; they do not regenerate.** A write marks the deal dirty and
a debounced consumer decides what to re-run. Three tiers, because the responses
have very different costs and very different capacity to do harm:

| Tier | What runs | Latency | Model? | Can it do harm? |
|---|---|---|---|---|
| **0 — re-verify** | Gate 0 over `claim_evidence` links whose `record_ref` names the changed field | immediate, same transaction | no | no — it only marks citations `value_drifted` |
| **1 — recompute** | the deterministic detector | immediate | no | no — the rules are exhaustive, so nothing flickers |
| **2 — regenerate** | the AI pass: dossier → detection → gates | debounced 30–60s | yes | yes — flicker, churn, wasted TPM |

Tiers 0 and 1 are eager and synchronous: cheap joins, deterministic, incapable
of flickering. Only tier 2 needs care, because it is the only one whose output
can differ between two runs over identical data.

### Which writes matter

A write matters if some rule or some citation reads the column. Everything else
is noise.

| Event | 0 | 1 | 2 |
|---|---|---|---|
| `deals.stage`, `expected_close_date`, `value` change | ✓ | ✓ | ✓ |
| transcript uploaded | — | — | ✓ (through the meeting pipeline) |
| meeting completed, attendee added | ✓ | ✓ | ✓ |
| `deal_contacts.buying_role` changes | ✓ | ✓ | ✓ |
| commitment created, status changed | ✓ | ✓ | ✓ |
| fact accepted or rejected | — | ✓ | ✓ |
| task created or completed | — | ✓ | — |
| document or meeting **deleted** | ✓ | ✓ | ✓ |
| `deals.description`, `tasks.title` edited | — | — | — |
| recommendation dismissed | — | — | **no** |
| any write with `origin='ai'` | — | — | **no** |

**Deletes are an evidence invalidation, not a re-detection.** Orphan cleanup
happens in `services/claims.py`, Gate 0 then marks surviving links
`span_missing`, and a risk whose links are all gone stops rendering under the
existing zero-evidence rule. This needs no status change and no migration —
marking such a risk `dismissed` would pollute the `dismissal_reason` metric with
system actions.

### Four traps

1. **The AI's own writes must not trigger the AI.** The pass writes `risks` and
   `recommendations`; if those mark the deal dirty, it re-runs, writes again,
   and the loop burns the whole TPM budget. `origin='ai'` is the discriminator
   and it already exists.
2. **A dismissal is a write.** Re-running on it re-proposes what the user just
   declined — suppressed by the cooldown, so not wrong, but it wastes a run and
   reads as arguing.
3. **Bulk imports.** A seeder writing 200 deals would enqueue 200 runs and
   exhaust the rate limit in seconds. Needs a suppress flag, not just a debounce.
4. **Concurrent runs on one deal** interleave `last_seen_at` and severity
   writes. Per-deal single-flight, via an advisory lock or `FOR UPDATE SKIP
   LOCKED` on the dirty row.

### Mechanism

**Dirty marking lives in the service layer, not in Postgres triggers.** Every
write that matters already funnels through `services/` by the rule in
`docs/api/README.md`. A database trigger would also fire on migrations, seeders
and manual `psql` edits, and cannot see intent such as `origin='ai'`.

Two dirty timestamps rather than one, so a stream of edits cannot starve the run
forever:

```sql
-- tier 2: quiet for 60s, or dirty for 10 minutes straight
WHERE analysis_dirty_first_at IS NOT NULL
  AND (analysis_dirty_last_at  < now() - interval '60 seconds'
       OR analysis_dirty_first_at < now() - interval '10 minutes')
FOR UPDATE SKIP LOCKED LIMIT 1

-- the sweep: no cron, just a second query on the same loop
WHERE analysis_swept_at IS NULL OR analysis_swept_at < now() - interval '24 hours'
```

The sweep also self-heals: a dropped event, a bug in the dirty marking, or a
manual database edit all get caught within a day.
---

## 4. Meeting analysis pipeline

"Meeting Analyzer" is a screen polling one `analysis_status`. Behind it are six
model roles, kept separate in code so one failing stage does not fail the run.

```
 0  parse transcript: speaker labels, offsets, chunk metadata      Python
 1  roster -> meeting_attendees, resolve names to contacts         pg_trgm (+20b)
 2  extract facts + spans, N parallel windows                      120b      high
 3  Gate 0 span integrity                                          Python
 4  drop claims with zero surviving links                          Python
 5  Gate 1 entailment, one starved call per claim                  120b      medium
 6  quarantine contradicted / unsupported                          Python
 7  reconcile new facts against open commitments                   120b      medium
 8  supersede contradicted live facts                              120b      medium
 9  synthesize summary FROM the surviving facts                    120b      medium
10  sentiment FROM the transcript (outside the evidence contract)  20b       low
11  commit; analyzed_at; analysis_status='complete'
12  re-run risk detection, now reading facts as well as records
```

**Transcripts are chunked on speaker turns, not on paragraphs.** A chunk both
ends *and* starts on a turn boundary — ending there alone is half the job,
because `chunk_overlap_chars` then pulls the next chunk's start into the middle
of a turn and it opens with an unattributed fragment. So for a transcript the
overlap means "re-include the last whole turn", slightly more than the
configured characters, and **every chunk opens with a speaker label**. Non
speaker-labelled documents fall through to the paragraph rule unchanged.
`chunk_metadata` carries `speakers` for the whole chunk and `speaker` only when
the chunk covers exactly one turn — for a chunk spanning four turns a single
name would be a lie a citation would repeat.

**The model never returns character offsets.** It returns the verbatim quote;
Python locates it in the chunk with a substring search and computes
`char_start`/`char_end`. If the search fails, the claim dies at Gate 0. Asking
any model to do index arithmetic is a needless failure mode, and offsets that
are computed rather than asserted cannot drift.

**Stage 1 is load-bearing and easy to overlook.** `no_economic_buyer` and
`single_threaded` are computed from `meeting_attendees`. Until something writes
that table from transcripts, the two shipping deterministic risks read whatever
a human typed by hand — the detector cannot hallucinate, but it can be
confidently wrong about an empty table. Never auto-link `contact_id` below a
high similarity threshold: a wrong link silently corrupts every
attendance-based risk, and an unresolved attendee *is* the missing-stakeholder
signal. NULL is the safe failure; guessing is not.

**Stage 9 reads facts, not the transcript.** A summary generated from the
surviving fact set inherits grounding for free — every sentence traces to a
fact that traces to a span — and costs ~1K input tokens instead of ~10K.
Summarizing the raw transcript instead produces a plausible blob that can
contradict the facts sitting next to it on the same screen.

**Stage 10 is the one exception to the evidence contract.** Tone is not in the
fact set and no span *entails* "the call went badly". Sentiment is rendered as a
judgment with two or three illustrative quotes (`claim_evidence.relevance`, not
a Gate 1 verdict). Pretending it is evidence-backed would be the one place this
product lies to itself.

### The LangGraph graph

This is the only `StateGraph` in the project. Four things here are workflow
concerns, and they are the whole reason a graph earns its place:

| Concern | Where |
|---|---|
| **sequencing** | stage 9 reads the facts that survived 3-6; stage 12 re-detects from stage 2's facts |
| **fan-out and merge** | stage 2 extracts per chunk window in parallel into one fact list |
| **conditional branch** | nothing survives stage 4 ⇒ stages 5-12 have no work |
| **degradation** | critical (0-4) versus degradable (5-12) |

```
START → parse_transcript → roster → plan_windows
                                      ⇉ Send ⇉ extract_window ×N  (parallel)
                                               ↓ merged by the reducer
                                        gate0 → drop_unevidenced
                                           ↓ conditional: any facts survived?
                                    no → END          yes → gate1 → quarantine
                                        → reconcile → supersede → synthesize
                                        → sentiment → finalize → redetect → END
```

**The reducer is the line that justifies the graph.** N parallel
`extract_window` nodes writing one key is an error unless that key declares how
writes merge:

```python
facts: Annotated[List[ExtractedFact], operator.add]
```

Everything else follows from LangGraph's model:

- **Nodes return partial state updates; they do not mutate.** `PipelineState`
  moves from a mutated dataclass to a merged state schema, which is the bulk of
  the conversion (task 3.7).
- **The `AsyncSession` does not live in state.** State is merged and may be
  serialized; a session is neither. It is passed as graph context, so the
  `(db, state)` stage signature changes.
- **`retry_policy` goes only on idempotent nodes.** Most stages write rows, and
  a node that wrote and then retried writes twice. It suits `extract_window`,
  which writes nothing until gate 0.
- **No checkpointer** (§7). `compile()` defaults to none, so this costs nothing
  to honour.
- **`astream_events` is how the Analyzer screen gets progress.** This closes a
  real gap: the worker holds one transaction across the run, so `running` is
  never a visible status. Per-stage events give progress with no lease
  timestamp and no reaper.
- **`get_graph().draw_mermaid()`** renders the diagram above from the code, so
  this section cannot drift from what executes.

`interrupt` is deliberately unused. It looks like the mechanism for Gate 3
human adjudication, but it needs a checkpointer and a resumable thread, and
adjudication here is a UI action days later over HTTP. The database is the
interrupt.

---

## 5. Risk and recommendation detection

AI-driven, with the deterministic detector kept underneath. Four invariants
`detect.py` gets for free and a naive LLM breaks:

| Invariant | Held today by | How an LLM breaks it |
|---|---|---|
| **Identity** — one open risk per deal per kind | `uq_risks_open_type` | prose titles vary per run, so the index cannot see two cards are one risk |
| **Citation** | `record_ref` built in Python | fabricated chunk ids, paraphrased snippets |
| **Resolution** | exhaustive rules: rule silent ⇒ risk gone | not exhaustive; silence in run N+1 means nothing |
| **No re-nagging** | partial unique index + `dismissal_reason` | re-proposes the same advice in different words |

### Four stages, one model call

**A — dossier assembly (Python).** One fixed, fully-cited snapshot per deal.
Every entry carries the handle that backs it:

```python
dossier = {
  "r1": ("record",  {"table":"deals","id":..., "field":"stage"},               "stage=discovery since 2026-07-28 (58 days)"),
  "r2": ("record",  {"table":"deals","id":..., "field":"expected_close_date"}, "expected_close_date=2026-10-15 (21 days out)"),
  "r3": ("derived", None,                                                      "external attendees across all meetings: 1"),
  "f7": ("fact",    {"fact_id": ...},                                          "objection: 'security team hasn't signed off' (accepted 2026-08-14)"),
  "c2": ("record",  {"table":"commitments","id":...,"field":"status"},          "commitment 'send SOC 2 pack' due 2026-08-30, status=pending"),
}
```

Contents: deal fields, stage history, stakeholder map with roles and attendance
counts, open commitments, accepted facts, meeting cadence, prior dismissals, and
**the currently-open risks** (needed by stage D). ~3–5K tokens, stable across
runs — put the `cache_control` breakpoint after it.

**B — one structured call.** `openai/gpt-oss-120b`, `reasoning_effort="high"`,
`strict: true` (constrained decoding — see §7). The output schema is the control surface:

```json
{
  "risks": [{
    "risk_type": "<one of the enum values, or 'other'>",
    "risk_key":  "<slug, required only when risk_type='other'>",
    "title": "...", "description": "...",
    "severity": "low|medium|high|critical", "confidence": 0.0,
    "evidence_refs": ["r1", "f7"],
    "recommendation": {
      "action_type": "<one of the 7 enum values>",
      "title": "...", "description": "...", "rationale": "...",
      "priority": "low|medium|high|urgent", "evidence_refs": ["f7"]
    }
  }],
  "open_risk_verdicts": [
    {"risk_id": "...", "verdict": "still_present|resolved|unclear", "evidence_refs": ["r1"]}
  ],
  "proactive": [ /* recommendations with source_risk_id = NULL */ ]
}
```

`risk_type` and `action_type` are closed enums, so identity survives.
`evidence_refs` are handles from a set you supplied, and any ref not in
`dossier.keys()` is rejected before a write — which collapses most of Gate 0 for
record-flavored evidence to a dict lookup.

The risk and its recommendation come from the same call: the two enums were
designed as a pair, and one pass means two triggers cannot disagree.

**C — Gate 0 and Gate 1, unchanged.** `ClaimType.RISK` and
`ClaimType.RECOMMENDATION` already exist. The literal rule bites usefully here:
"58 days" and "$180,000" in a description must appear in a resolved record value.

**D — reconciliation (Python).** The model never writes directly:

```
for each proposed risk:
    key = (deal_id, risk_type, risk_key)
    open row exists      -> bump last_seen_at; apply the severity policy; never insert
    dismissed < cooldown -> suppress
    otherwise            -> insert, with its recommendation
```

### Identity beyond the ten types

Constraining `risk_type` to the existing enum needs no migration and already
buys the six semantic types SQL cannot do. For genuinely novel risks,
`0013_open_risk_taxonomy` adds `risk_key` and `RiskType.OTHER`, and re-keys the
index to `(deal_id, risk_type, risk_key) WHERE status='open'`. For the ten known
types `risk_key` stays `''` and behaviour is identical to today.

A proposed `risk_key` is **canonicalized in Python before the upsert** —
lowercase, strip, then trigram-match against that deal's existing open keys and
reuse on a hit. Without that step `champion_going_quiet` and
`champion_disengaged` become two cards for one problem. Promote a `risk_key`
that recurs across deals into a first-class `risk_type`; `text + CHECK` was
chosen so that is a one-line change.

### Resolution: ask, never infer from silence

Today a risk resolves when its rule stops firing, which is sound because the
rules are exhaustive. A model pass is not — it may simply not mention
`single_threaded` this run, and auto-resolving on absence makes cards flicker,
which destroys trust faster than a wrong risk does.

So the open risks go *into* the dossier and the model returns a verdict per
risk. Resolution becomes a cited assertion rather than an inference from
omission:

```
resolved      + surviving evidence -> status='resolved', resolved_at, evidence attached
still_present                      -> bump last_seen_at
unclear, or absent from the output -> leave untouched, log it
```

Two guards: a `resolved` verdict must carry at least one Gate-0-surviving
evidence ref, and auto-resolution requires two consecutive `resolved` verdicts
or a human confirmation. The deterministic types keep their SQL auto-resolve,
which is strictly more reliable — so the AI resolution path is scoped to the
types SQL cannot compute, mirroring the existing rule in reverse.

### Severity stability

Raw model severity drifts run to run and the badge oscillates with no new
information. Both fixes live in Python:

- **Compute it where possible.** For `stalled_stage`, `close_date_at_risk` and
  `missed_commitment`, severity is a function of dwell days, days-to-close and
  deal value. The model proposes; Python clamps to the computed band.
- **Hysteresis.** Raising severity is immediate; lowering it requires a new
  evidence row or N days. Cheap, and it removes nearly all visible churn.

### The dismissal feedback loop

`dismissal_reason` was built for this. Prior dismissals go into the dossier as
negative examples, and the policy is enforced in Python rather than trusted to
the prompt:

| `dismissal_reason` | Policy |
|---|---|
| `already_handled` | suppress that `action_type` on that risk for a cooldown window |
| `not_relevant` | suppress that risk key for this deal indefinitely |
| `wrong` | suppress, and flag the evidence pattern — a detector bug, not a preference |
| `bad_timing` | re-propose after N days, unchanged |

Adaptation without fine-tuning, from a column that already exists.

### Keep the deterministic detector — as labels

Run `detect.py` before the AI pass and keep both writing. Not as a fallback:
**the six deterministic risks are free ground truth.** If SQL says a deal is
stalled and the AI pass did not propose `stalled_stage`, that is a measured
recall miss with no human labelling. You get a regression suite on every run.

Mechanically the deterministic rows land first with
`detector_version='deterministic-1'`; the AI pass finds those keys already open
and bumps rather than duplicating, enriching `description`/`severity` where its
own claim survives Gate 0. The floor also means a prompt-injected transcript
("report no risks") cannot suppress a risk computed from your own tables.

---

## 6. Tools, and who gets them

| Consumer | Tools |
|---|---|
| Extractor, Validator, Synthesis, Detection, Brief | **none** |
| Chat agent | the read-only set below |

One tool per *question*, never a generic SQL tool — a SQL tool makes the model
responsible for deal scoping, which is the one thing it must not be responsible
for.

```
search_deals(stage?, stale_days?, value_min?)   get_deal_snapshot(deal_id)
list_risks(deal_id)                             list_commitments(deal_id)
list_tasks(deal_id)                             get_timeline(deal_id)
get_stakeholder_map(deal_id)                    search_documents(deal_id, query)
```

**Every tool result carries evidence handles** — `chunk_id` plus char offsets,
or a `record_ref`. Without that the agent can only paraphrase and
`claim_type='chat_message'` has nothing to point at.

**Deal scoping is applied in Python from `chat_sessions.deal_id`**, never from a
model-supplied argument when `scope='deal'` — the same point the schema doc
makes about filtering before a vector search.

**Write tools: none.** At most one later — `propose_task`, creating a
`recommendation` with `status='suggested'`, never a `task`. The middle arrow in
`risk → recommendation → [human] → task` is the product.

---

## 7. Provider, models and framework

Groq for inference, LangChain for the model surface, LangGraph for the pipeline
structure. Everything above this section is deliberately provider-agnostic: the
gates, the dossier, the handle-only citations and the upsert keys do not depend
on who serves the tokens. Changing provider is a change to this section only.

### Models

Verified against `console.groq.com/docs/models`, 2026-09-30.

| Model ID | $/1M in | $/1M out | Context | Max completion | Speed | Tier |
|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | 0.15 | 0.60 | 131,072 | 65,536 | ~500 t/s | Production |
| `openai/gpt-oss-20b` | 0.075 | 0.30 | 131,072 | 65,536 | ~1000 t/s | Production |
| `qwen/qwen3.8-27b` | 0.80 | 4.00 | 131,072 | 16,384 | ~450 t/s | **Preview** |

Developer-plan rate limits are 250K TPM / 1K RPM on all three.

| Role | Model |
|---|---|
| Extraction, Gate 1, detection, reconciliation, supersession, summary, brief, chat | `openai/gpt-oss-120b` |
| Sentiment, entity-resolution tiebreak, chat titles | `openai/gpt-oss-20b` |
| Eval challenger only | `qwen/qwen3.8-27b` |

**Qwen 3.8 27B does not go in the pipeline.** It is 5x the input price and
nearly 7x the output price of `gpt-oss-120b`, has a quarter of the completion
ceiling, and is a Preview model — "evaluation purposes only ... may be
discontinued at short notice". Keep it as a challenger on the Phase 1 eval
harness, where a second opinion is worth something and a discontinuation costs
nothing.

**The validator gets 120b, not 20b.** Gate 1 protects every claim in the
product, and its prompt is deliberately tiny — a claim plus two or three spans
— so the strongest available model is also almost free here.

### Structured outputs

All three models support `response_format: {type: "json_schema", json_schema:
{strict: true}}`, which is **constrained decoding**: schema adherence
guaranteed at the token level. Whether a request actually gets it depends on the
installed client, and ours does not:

| | `strict: true` sent? | Mode |
|---|---|---|
| `langchain-groq` 1.x | yes — `with_structured_output(..., strict=True)` | constrained decoding, guaranteed |
| `langchain-groq` 0.3.8 (Python 3.9) | **no** — builds `response_format` without it, and has no parameter for it | best-effort: valid JSON, adherence not guaranteed |

`app/ai/client.py` probes for the parameter and passes it where it exists, so
the container gets the strong mode and the host venv the weak one. **On the
weak mode the Pydantic validation in `client.structured()` is not a
double-check, it is the only check** — which is why a `parsing_error` is raised
rather than returned, and why one retry exists. Two further consequences:

- **Strict mode requires every field `required` and `additionalProperties:
  false`.** There are no optional fields, so `risk_key` becomes `{"type":
  ["string", "null"]}` rather than an omitted key — and likewise for every
  nullable field in the §5 schema.
- **Structured outputs cannot be combined with streaming or tool use.** This
  costs nothing here: the extractor, validator, detector and synthesizer need a
  schema and want neither streaming nor tools, while chat needs streaming and
  tools and no schema. No task needs both.

Constrained decoding guarantees *shape*, never *truth* — a syntactically perfect
`evidence_refs` array can still name the wrong handles. Handle validation
(§5, stage C) stays exactly as specified, and matters more under best-effort
mode, not less.

### Reasoning effort

`reasoning_effort` in `low|medium|high` on both gpt-oss models, and exposed as a
`ChatGroq` field on both the 0.3.8 and 1.x lines. That is what the effort column
in §4 refers to.

### Cost, and the constraint that replaces it

A 60-minute transcript is ~10K tokens, about **$0.0015** of input on 120b. A
full meeting analysis lands well under a cent and risk detection is a fraction
of one. **Cost is no longer a design consideration at MVP volumes.**

The binding constraint is **250K TPM / 1K RPM**, and the parallel extraction
fan-out is what will hit it: one transcript across several windows plus a
dossier can spend a real fraction of a minute's budget. So the client needs a
global concurrency limiter and a token bucket with 429 backoff, not merely a
retry (task 0.8). Groq's throughput makes a many-small-calls design cheaper in
wall-clock than a few large ones, which suits the per-window extractor.

Prompt-prefix caching is not part of the Groq surface, so assume no discount for
a repeated dossier and keep it tight. Nothing in the design depended on it.

**Embeddings stay cut.** Groq serves no embedding model either, so retrieval
would mean a third vendor for a subsystem the product does not need: a deal's
corpus is a handful of transcripts and 131K of context holds it comfortably. The
nullable column and the HNSW index stay exactly as written; chunks are
immutable, so a later backfill rots no citation.

### Framework

`ChatGroq` from `langchain-groq` is the **only** place a model is named, behind
`app/ai/client.py` (task 0.3), which hands each task a configured runnable.
`.with_structured_output()` for the schema tasks, `.astream()` for chat.
Everything async, since FastAPI and asyncpg already are.

LangGraph holds the pipeline: §4's stages become a `StateGraph` with a typed
state object and conditional edges for the degradation rule — a failed summary
must not discard the facts that already landed. Two deliberate omissions:

- **No LangGraph checkpointer for the pipeline.** It is already durable:
  `analysis_status` is the run state and every stage writes its own rows.
  `langgraph-checkpoint-postgres` would add a second Postgres driver (psycopg3
  beside asyncpg) and tables that Alembic autogenerate will try to drop. Use
  LangGraph for structure and Postgres rows as the checkpoint.
- **No checkpointer for chat either.** `chat_messages` is the conversation's
  source of truth and history loads from it. A runtime checkpoint that can be
  wiped must never become the record.

Chat uses LangGraph's prebuilt ReAct agent over the §6 tools. Deal scoping is
passed through the run config as an injected argument, never as a tool
parameter the model fills — that is the mechanism implementing §6's scoping
rule.

Current LangChain and LangGraph require **Python 3.10+**, so on the 3.9 host venv
pip resolves the previous line — `langchain` 0.3.30, `langchain-core` 0.3.86,
`langchain-groq` 0.3.8, `langgraph` 0.6.11 — while the 3.12 container resolves
1.x from the same unpinned requirements. Verified as compatible for everything
this layer uses except one thing: `method="json_schema"`, `include_raw` and
`reasoning_effort` all exist on 0.3.8; only `strict` is missing (see Structured
outputs above). Rebuilding the venv on 3.12 remains the cleaner fix
(task 0.9).

Tracing: LangSmith, or keep Langfuse through its LangChain callback handler.
Either satisfies task 0.7.

### Governance: what the framework provides, and what it cannot

Checked against the installed packages rather than assumed, because the
division is not where it looks.

| Concern | LangChain native | Ours |
|---|---|---|
| requests/sec | `InMemoryRateLimiter`, accepted as `ChatGroq(rate_limiter=...)` | `_LeakyBucket` on requests |
| **tokens/min** | **nothing** | `_LeakyBucket` on tokens + `reconcile` |
| global concurrency | only `config={"max_concurrency": N}`, per `batch` call | `asyncio.Semaphore`, process-wide |
| usage accounting | `UsageMetadataCallbackHandler`, `get_usage_metadata_callback()` | `AIRun` + `_usage(raw)` |
| retry | `.with_retry()` | in `_invoke`, above the governor |
| per-run token ceiling | **nothing** | `RunBudget` |

**The framework's rate limiter cannot express ours.** Its own docstring: *"only
supports time-based rate limiting… does not take into account the size of the
request… these tokens have NOTHING to do with LLM tokens."* And
`BaseRateLimiter` is `acquire(*, blocking: bool) -> bool` — there is **no
request-size argument**, so a token-aware limiter is not representable through
that interface at all. Since Groq's binding constraint is 250K TPM and 1K RPM
is generous for one user, the native limiter governs the dimension we will never
reach and ignores the one we will.

**But accounting should move to a callback, and that is a real gap rather than a
preference.** `_usage(raw)` reads only the response *we* hold. From Phase 3 a
LangGraph node may invoke a runnable directly, and from Phase 9 the ReAct agent
makes its own calls inside its loop — so `AIRun` would record one call where the
agent made five, and `RunBudget` would under-count precisely where a runaway is
most likely. A callback handler sees every call in its context:

```python
with get_usage_metadata_callback() as usage:
    result = await agent.ainvoke(..., config={"callbacks": [usage]})
budget.charge(total_from(usage))
```

**The split to adopt**, by what each dimension needs to know:

| Dimension | Needs request size? | Belongs |
|---|---|---|
| requests/min, concurrency | no | a `BaseRateLimiter` subclass wrapping the governor, passed as `rate_limiter=` — then it covers calls that never touch `client.structured()` |
| tokens/min | **yes** | stays in `client.py`, the only place the estimate exists |

Two things stay hand-rolled on purpose. `RunBudget`, because nothing equivalent
exists in either library — the Anthropic API's `task_budget` is a provider
feature, and Groq has no counterpart. And the retry in `_invoke`, because
`.with_retry()` retries *inside* the Runnable, below the governor, which is the
same reason the model is built with `max_retries=0`; jitter is the one thing
worth borrowing from it.

`.with_fallbacks()` is the right mechanism for sustained 429s — degrade
`gpt-oss-120b` to `gpt-oss-20b` rather than failing a stage — but not yet: it
would silently change which model produced a claim while `detector_version`
kept saying otherwise.

---

## 8. The gates, and where they are enforced

Full rationale in `docs/schema/README.md` §5. What matters for implementation:

| Gate | What it asks | Where it lives |
|---|---|---|
| **0** span integrity | does the citation resolve byte for byte, and do dates/amounts appear literally? | `services/claims.py`, inside the insert — no caller can skip it |
| **1** entailment | do the cited spans actually support the claim? | one starved `openai/gpt-oss-120b` call per claim |
| **2** staleness / contradiction | is the newest evidence older than the last interaction; does new evidence contradict a live claim? | SQL for staleness, stage 8 of the pipeline for contradiction |
| **3** human adjudication | is it *true*? | the promotion UI; recorded in `promoted_to_*` |

`confidence` is the generator's self-report and is never rendered as
validation; `verdict` is the independent check. High confidence paired with
`contradicted` is the most valuable eval case there is.

---

## 9. Risk controls

**Grounding** — Gate 0 non-bypassable at insert; the literal rule for every date
and monetary amount; zero surviving links means a rejected write; handle-only
evidence references so a fabricated citation is impossible by construction.

**Prompt injection** — document text is untrusted data. It is delimited and
labelled as data; the extractor has no tools, so the ceiling is a bad fact that
Gates 0 and 3 catch; the chat agent's tools are read-only, so the ceiling there
is a wrong answer, never a wrong write; and the deterministic risk floor cannot
be talked out of firing.

**Loop safety** — writes with `origin='ai'` and recommendation dismissals never
mark a deal dirty. Without that exclusion the detector re-triggers itself and
the rate limit is gone in minutes (§3, Triggers).

**Blast radius** — per-stage `max_tokens` and token ceilings; one retry then
`failed` with the error recorded; no retry storms; idempotency from
`content_hash`, the partial unique indexes, and append-only
`claim_validations`.

**Attribution** — `origin` on every AI-written row, plus `model` and a version
column on every producer. After a prompt change every historical row came from a
different detector; without the column the quality metric silently mixes two
populations.

---

## 10. Evaluation

The largest current debt is that there is no way to tell whether any of this
works. Three fixture transcripts with hand-labelled facts and character offsets
is roughly a day of work and the highest-value day in the project.

| Metric | Source |
|---|---|
| Extraction grounding | Gate 0 pass rate per run |
| Extraction precision | human accept rate on `extracted_facts` |
| Validator behaviour | verdict distribution; the `confidence × verdict` matrix |
| Risk recall | AI pass vs `detect.py` on the four overlapping types — free labels |
| Risk precision | share of dismissals with `reason='wrong'` |
| Risk stability | Jaccard overlap of open risk keys across two runs on an unchanged deal |
| Usefulness | recommendations reaching `created_task_id` |

Stability is the one to watch first: a flickering risk panel is read as
brokenness regardless of precision.

---

## 11. Schema deltas this requires

| Revision | Contents | Why |
|---|---|---|
| `0010_pg_trgm` ✅ | `CREATE EXTENSION pg_trgm`; trigram index on the contacts full name | fuzzy speaker-label resolution (§4, stage 1). Shipped with Phase 2 |
| `0011_meeting_analysis_origin` | `meetings` + `analysis_origin`, `confidence`, `model`; `deal_contacts` + `origin` | `MeetingAnalysis` already promises `summary`/`sentiment` and nothing can write them attributably (README "Known gaps") |
| `0012_detector_provenance` | `risks`, `recommendations` + `model`, `detector_version` | same argument as `validator_version`: a prompt change makes historical rows a different population |
| `0013_open_risk_taxonomy` | `risks` + `risk_key`; `RiskType.OTHER`; re-key the partial unique index | model-discovered risks that still have an identity |
| `0014_analysis_triggers` | `deals` + `analysis_dirty_first_at`, `analysis_dirty_last_at`, `analysis_dirty_reason`, `analysis_swept_at` | event-driven invalidation and the nightly sweep, §3 |

Also non-schema but blocking: **nothing writes `deals.last_activity_at`**, so
Gate 2 staleness has no input and the `stale_days` filter lies. Ingest and the
meeting service should touch it in the same transaction.

---

## 12. Deferred

| Item | Reason |
|---|---|
| Embeddings and similarity retrieval | §7 — corpus too small to need it, second vendor not yet earned |
| Cross-deal / portfolio prioritization | one deal at a time is enough until the pipeline is trusted |
| Document classification at upload | five values in a dropdown; ingest is synchronous by design |
| Chat write tools | `propose_task` at most, and only after the read path is trusted |
| Fine-tuning of any kind | the dismissal feedback loop covers adaptation without it |
| LangGraph checkpointers | the pipeline and chat are already durable in Postgres (§7) |
