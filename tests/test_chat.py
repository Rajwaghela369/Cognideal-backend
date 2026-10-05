import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import ToolException
from sqlalchemy import select

from app.ai import chat, client
from app.ai.schemas import ChatTitleOut
from app.ai.tools import ToolRegistry
from app.models import (
    ChatMessage,
    ChatSession,
    Commitment,
    Contact,
    Deal,
    DealContact,
    DealStageHistory,
    Document,
    DocumentChunk,
    Meeting,
    Risk,
    Task,
)
from app.models.enums import (
    BuyingRole,
    ChatRole,
    ChatScope,
    ClaimType,
    DocumentSourceType,
    MessageStatus,
    OwnerSide,
    RiskType,
    Severity,
    SourceKind,
    VerificationStatus,
)
from app.services import claims


@pytest.mark.asyncio
async def test_deal_scope_is_enforced_in_python(db, deal_id):
    registry = ToolRegistry(db, deal_id)
    assert registry._deal(None) == deal_id
    with pytest.raises(ToolException):
        registry._deal(str(uuid.uuid4()))
    with pytest.raises(ToolException):
        registry._deal("not-a-uuid")


@pytest.mark.asyncio
async def test_all_read_tools_return_evidence_handles(db, deal_id):
    deal = await db.get(Deal, deal_id)
    contact = Contact(
        account_id=deal.account_id,
        first_name="Priya",
        last_name="Raman",
        title="CFO",
    )
    db.add(contact)
    await db.flush()
    db.add(DealContact(
        deal_id=deal_id,
        contact_id=contact.id,
        buying_role=BuyingRole.ECONOMIC_BUYER,
    ))
    db.add(Task(deal_id=deal_id, title="Send pricing"))
    db.add(Commitment(
        deal_id=deal_id,
        description="Send the security report",
        owner_side=OwnerSide.US,
    ))
    db.add(DealStageHistory(deal_id=deal_id, to_stage=deal.stage))
    db.add(Meeting(deal_id=deal_id, title="Discovery call"))
    document = Document(
        deal_id=deal_id,
        account_id=deal.account_id,
        source_type=DocumentSourceType.NOTE,
        title="Security note",
        storage_uri="documents/%s" % uuid.uuid4().hex,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        occurred_at=datetime.now(timezone.utc),
    )
    db.add(document)
    await db.flush()
    db.add(DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        content="The customer requested the security report.",
    ))
    risk = Risk(
        deal_id=deal_id,
        risk_type=RiskType.SECURITY_REVIEW_PENDING.value,
        risk_key="",
        title="Security review pending",
        description="The report is still required.",
        severity=Severity.HIGH,
    )
    db.add(risk)
    await db.flush()
    await claims.attach_evidence(
        db,
        claim_type=ClaimType.RISK,
        claim_id=risk.id,
        deal_id=deal_id,
        source_kind=SourceKind.RECORD,
        snippet=deal.stage.value,
        record_ref={"table": "deals", "id": str(deal_id), "field": "stage"},
        verification_status=VerificationStatus.VERIFIED,
    )
    await db.flush()

    registry = ToolRegistry(db, deal_id)
    tools = registry.langchain_tools()
    READ_TOOLS = {
        "search_deals",
        "get_deal_snapshot",
        "list_risks",
        "list_commitments",
        "list_tasks",
        "get_timeline",
        "get_stakeholder_map",
        "search_documents",
    }
    # Eight read tools plus `propose_task` (task 11.8), which is the only write
    # and the only tool that returns no evidence handle -- it creates a row
    # rather than citing one, so it is excluded from the loop below. Asserted as
    # an exact set so a tenth tool cannot appear unnoticed.
    assert {tool.name for tool in tools} == READ_TOOLS | {"propose_task"}
    outputs = [
        await registry.search_deals(),
        await registry.get_deal_snapshot(),
        await registry.list_risks(),
        await registry.list_commitments(),
        await registry.list_tasks(),
        await registry.get_timeline(),
        await registry.get_stakeholder_map(),
        await registry.search_documents("security"),
    ]

    for raw in outputs:
        entries = json.loads(raw)["entries"]
        assert entries
        for entry in entries:
            citation = registry.evidence[entry["handle"]]
            assert citation.get("record_ref") or (
                citation.get("chunk_id") is not None
                and citation.get("char_start") is not None
                and citation.get("char_end") is not None
            )
    await db.commit()


@pytest.mark.asyncio
async def test_chat_citations_preserve_the_inline_handle(db, deal_id):
    session = ChatSession(scope=ChatScope.DEAL, deal_id=deal_id)
    db.add(session)
    await db.flush()
    message = ChatMessage(
        session_id=session.id,
        role=ChatRole.ASSISTANT,
        content="SecureFlow is the deal [e1].",
        status=MessageStatus.COMPLETE,
    )
    db.add(message)
    await db.flush()
    registry = ToolRegistry(db, deal_id)
    registry.evidence["e1"] = {
        "source_kind": "record",
        "deal_id": str(deal_id),
        "record_ref": {"table": "deals", "id": str(deal_id), "field": "name"},
        "snippet": "SecureFlow",
    }

    attached = await chat._attach_citations(
        db, session, message, message.content, registry
    )
    evidence = list(await claims.evidence_for(db, ClaimType.CHAT_MESSAGE, message.id))

    assert attached[0]["handle"] == "e1"
    assert attached[0]["evidence_id"] == str(evidence[0].id)
    assert evidence[0].verification_status == VerificationStatus.VERIFIED
    await db.commit()


@pytest.mark.asyncio
async def test_interrupted_stream_leaves_a_recoverable_row(db, deal_id, monkeypatch):
    session = ChatSession(scope=ChatScope.DEAL, deal_id=deal_id)
    db.add(session)
    await db.commit()
    session_id = session.id

    class BrokenAgent:
        async def astream(self, *args, **kwargs):
            yield AIMessageChunk(content="Partial answer"), {"langgraph_node": "agent"}
            raise RuntimeError("stream interrupted")

    monkeypatch.setattr(chat, "create_react_agent", lambda *a, **k: BrokenAgent())
    monkeypatch.setattr(chat.client, "agent_model", lambda **kwargs: object())

    events = [event async for event in chat.stream_turn(db, session, "What changed?")]
    messages = list((await db.scalars(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )).all())

    assert json.loads(events[0])["content"] == "Partial answer"
    assert json.loads(events[-1])["type"] == "error"
    assert messages[0].status == MessageStatus.COMPLETE
    assert messages[1].status == MessageStatus.ERROR
    assert messages[1].content == "Partial answer"


@pytest.mark.asyncio
async def test_completed_turn_records_all_call_usage_and_citations(
    db, deal_id, monkeypatch
):
    session = ChatSession(
        scope=ChatScope.DEAL, deal_id=deal_id, title="Existing title"
    )
    db.add(session)
    await db.commit()
    session_id = session.id

    class CompletedAgent:
        async def astream(self, *args, **kwargs):
            yield AIMessageChunk(content="SecureFlow is the deal [e1]."), {
                "langgraph_node": "agent"
            }

    class Registry:
        def __init__(self, *args, **kwargs):
            # **kwargs so `handle_offset` reaches it: the real registry takes
            # one now, and a double with a narrower signature turns a wiring
            # change into a TypeError rather than a test failure that explains
            # itself.
            self.handle_offset = kwargs.get("handle_offset", 0)
            self.evidence = {
                "e1": {
                    "source_kind": "record",
                    "deal_id": str(deal_id),
                    "record_ref": {
                        "table": "deals", "id": str(deal_id), "field": "name"
                    },
                    "snippet": "SecureFlow",
                }
            }

        def langchain_tools(self):
            return []

    @asynccontextmanager
    async def fake_usage(_budget):
        yield object()

    monkeypatch.setattr(chat, "create_react_agent", lambda *a, **k: CompletedAgent())
    monkeypatch.setattr(chat, "ToolRegistry", Registry)
    monkeypatch.setattr(chat.client, "agent_model", lambda **kwargs: object())
    monkeypatch.setattr(chat.client, "track_usage", fake_usage)
    monkeypatch.setattr(chat.client, "usage_total", lambda callback: 321)

    events = [event async for event in chat.stream_turn(db, session, "Which deal?")]
    assistant = await db.scalar(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == session_id,
            ChatMessage.role == ChatRole.ASSISTANT,
        )
    )

    assert json.loads(events[-1])["type"] == "done"
    assert assistant.status == MessageStatus.COMPLETE
    assert assistant.token_usage["total_tokens"] == 321
    assert assistant.token_usage["citation_handles"][0]["handle"] == "e1"
    evidence = list(await claims.evidence_for(
        db, ClaimType.CHAT_MESSAGE, assistant.id
    ))
    assert len(evidence) == 1
    assert evidence[0].verification_status == VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_first_turn_title_is_persisted(db, deal_id, monkeypatch):
    session = ChatSession(scope=ChatScope.DEAL, deal_id=deal_id)
    db.add(session)
    await db.commit()
    session_id = session.id

    async def fake_structured(*args, **kwargs):
        return ChatTitleOut(title="Security review follow-up"), client.AIRun(
            "chat_title", "test-model", "chat-title@1"
        )

    monkeypatch.setattr(chat.client, "structured", fake_structured)
    await chat._title_session(session_id, "What remains?", "The review is pending.")
    await db.refresh(session)

    assert session.title == "Security review follow-up"
