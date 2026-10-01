"""Phase 0 acceptance checks that need the real HTTP layer.

Companion to ``verify_phase0.py``, which needs no server. Split because these
exercise the route -> service -> worker path end to end, and because the API has
to run in **its own process**: ``TestClient`` drives the app on its own event
loop while this script's connection pool is bound to another, and SQLAlchemy's
asyncpg connections cannot cross loops ("attached to a different loop").

Covers: the real `POST .../analysis` -> worker -> `complete` round trip; all
three `last_activity_at` write paths including the occurred_at dating and the
monotonic guard; the `stale_days` filter in both directions; and the crash
semantics -- a hard `os._exit` mid-run must leave the row `queued`.

Run from `backend/`, with postgres and minio up:

    ../deal-pilot-env/bin/uvicorn app.main:app --port 8099 --log-level warning &
    ../deal-pilot-env/bin/python tests/verify_phase0_http.py

Seeds rows named "AUDIT <hex>"; clean up with
``DELETE FROM accounts WHERE name LIKE 'AUDIT %';``.
"""
import asyncio, sys, os, json, subprocess, uuid, logging
from datetime import datetime, timezone, timedelta
sys.path.insert(0, '.')
logging.disable(logging.WARNING)

import httpx
from sqlalchemy import select, text
from app.db.session import SessionLocal
from app.models import Account, Deal, Meeting
from app.models.enums import AnalysisStatus, DealStage, MeetingStatus

API = "http://127.0.0.1:8099/api/v1"
h = httpx.Client(base_url=API, timeout=30)
R = []
def check(label, ok, detail=""):
    R.append(bool(ok)); print("  %s %s%s" % ("PASS" if ok else "FAIL", label, (" -- "+detail) if detail else ""))

async def seed():
    async with SessionLocal() as db:
        a = Account(name="AUDIT %s" % uuid.uuid4().hex[:6]); db.add(a); await db.flush()
        d = Deal(account_id=a.id, name="SecureFlow", stage=DealStage.DISCOVERY); db.add(d); await db.flush()
        did = d.id; await db.commit(); return did

async def val(model, pk, col):
    async with SessionLocal() as db:
        return await db.scalar(select(col).where(model.id == pk))

async def main():
    import worker
    deal_id = await seed()
    base = "/deals/%s" % deal_id

    print("\n0.5  real HTTP path: POST .../analysis -> queued -> worker -> complete")
    m = h.post(base + "/meetings", json={"title": "Discovery call", "meeting_type": "discovery"})
    check("POST /meetings -> 201", m.status_code == 201, "%s %s" % (m.status_code, m.text[:100]))
    mid = m.json()["id"]
    a = h.post(base + "/meetings/%s/analysis" % mid, json={})
    check("POST .../analysis -> 202", a.status_code == 202, "%s %s" % (a.status_code, a.text[:100]))
    check("reported status is queued", a.json().get("analysis_status") == "queued", json.dumps(a.json()))
    await worker.poll_queued_meetings()
    g = h.get(base + "/meetings/%s/analysis" % mid).json()
    check("GET .../analysis -> complete", g.get("analysis_status") == "complete", json.dumps(g))
    check("analyzed_at returned to the screen", g.get("analyzed_at"), str(g.get("analyzed_at")))

    print("\n0.6  the three real write paths touch last_activity_at")
    check("deal starts with none", (await val(Deal, deal_id, Deal.last_activity_at)) is None)

    june = datetime.now(timezone.utc) - timedelta(days=100)
    up = h.post(base + "/documents",
                files={"file": ("june-call.txt", b"Priya: we need SOC 2 Type II before signature.\n", "text/plain")},
                data={"source_type": "meeting_transcript", "occurred_at": june.isoformat()})
    check("POST /documents -> 201", up.status_code == 201, "%s %s" % (up.status_code, up.text[:150]))
    doc_v = await val(Deal, deal_id, Deal.last_activity_at)
    check("upload set last_activity_at", doc_v is not None, str(doc_v))
    check("dated by occurred_at, not now()",
          doc_v is not None and abs((doc_v - june).total_seconds()) < 5,
          "got %s, sent %s" % (doc_v, june))

    p = h.patch(base + "/meetings/%s" % mid, json={"status": "completed"})
    check("PATCH meeting -> completed 200", p.status_code == 200, "%s %s" % (p.status_code, p.text[:100]))
    meet_v = await val(Deal, deal_id, Deal.last_activity_at)
    check("completion advanced it past the June doc", meet_v and doc_v and meet_v > doc_v, "%s -> %s" % (doc_v, meet_v))

    older = datetime.now(timezone.utc) - timedelta(days=300)
    h.post(base + "/documents", files={"file": ("ancient.txt", b"Old note on budget.\n", "text/plain")},
           data={"source_type": "note", "occurred_at": older.isoformat()})
    old_v = await val(Deal, deal_id, Deal.last_activity_at)
    check("an older upload does NOT regress it", old_v == meet_v, str(old_v))

    print("\n0.6  stale_days now has input")
    ids = lambda r: [d["id"] for d in r.json()["items"]]
    check("stale_days=1 excludes a deal active now", str(deal_id) not in ids(h.get("/deals", params={"stale_days": 1})))
    async with SessionLocal() as db:
        await db.execute(text("UPDATE deals SET last_activity_at = now() - interval '30 days' WHERE id=:i"),
                         {"i": str(deal_id)}); await db.commit()
    check("stale_days=7 includes it once stale", str(deal_id) in ids(h.get("/deals", params={"stale_days": 7})))

    print("\n0.5  crash semantics: hard kill mid-run leaves the row queued")
    d2 = await seed()
    async with SessionLocal() as db:
        m2 = Meeting(deal_id=d2, title="Crash", status=MeetingStatus.COMPLETED,
                     analysis_status=AnalysisStatus.QUEUED)
        db.add(m2); await db.flush(); m2_id = m2.id; await db.commit()
    proc = subprocess.run([sys.executable, "-c", """
import asyncio, os, sys, logging
sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)
from app.ai import graph
import worker
async def hang(state, runtime):
    await asyncio.sleep(30)
graph.NODES["extract_window"] = graph.guarded("extract_window")(hang)
async def go():
    asyncio.ensure_future(worker.poll_queued_meetings())
    await asyncio.sleep(3)
    os._exit(9)
asyncio.get_event_loop().run_until_complete(go())
"""], capture_output=True, cwd=".")
    check("worker process died hard (no rollback, no handlers)", proc.returncode == 9, "rc=%s" % proc.returncode)
    st = await val(Meeting, m2_id, Meeting.analysis_status)
    check("row is still queued -- self-healing, no reaper", st == AnalysisStatus.QUEUED, "status=%s" % st)
    did = await worker.poll_queued_meetings()
    st2 = await val(Meeting, m2_id, Meeting.analysis_status)
    check("a later worker claims it and finishes", did and st2 == AnalysisStatus.COMPLETE, "status=%s" % st2)

    print("\n%d/%d checks passed" % (sum(R), len(R)))
    return 0 if all(R) else 1

sys.exit(asyncio.get_event_loop().run_until_complete(main()))
