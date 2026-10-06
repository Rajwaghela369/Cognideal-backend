"""Transcript chunks -> candidate facts with resolved spans -- task 3.2.

The division of labour here is the whole design:

    the model          returns a claim and the words that support it
    Python             finds those words and computes the offsets

The model is never asked for a character position. Index arithmetic is a
needless failure mode for any model and a hopeless one for a 20B, and -- more
importantly -- an offset that is *computed* cannot disagree with the text,
whereas one that is *asserted* can. A snippet Python cannot find is a fact that
never existed as far as the database is concerned.

Nothing here writes to the database. This module produces candidates; Gate 0
(task 3.3) decides which of them may be written, and the service layer does the
writing. Keeping extraction side-effect-free is what lets the eval harness
score it by replaying a recorded run.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.ai import client
from app.ai.prompts import get as get_prompt
from app.ai.schemas import ExtractedFactOut, ExtractionResult, narrow_payload
from app.core.config import settings

logger = logging.getLogger("cognideal.ai.extract")

#: Why a model-reported fact was dropped before it reached Gate 0.
UNLOCATABLE = "snippet_not_found"


@dataclass
class CandidateFact:
    """One fact the model reported, with its span resolved if it could be.

    ``chunk_id`` and the offsets are what Gate 0 re-checks and what
    ``evidence`` ultimately stores. ``rejected`` is set rather than the fact
    being silently dropped, because the *rate* of unlocatable snippets is the
    single most useful signal about extraction quality -- it is the share of
    answers that were paraphrased rather than quoted.
    """

    fact_type: str
    content: str
    snippet: str
    speaker: Optional[str]
    confidence: float
    payload: Optional[dict] = None
    payload_error: Optional[str] = None
    chunk_id: Optional[Any] = None
    chunk_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    rejected: Optional[str] = None

    @property
    def located(self) -> bool:
        return self.chunk_id is not None and self.rejected is None


def build_windows(chunks: Sequence[Any], size: Optional[int] = None) -> List[List[Any]]:
    """Group chunks into windows of roughly equal size.

    Even rather than greedy: filling windows to the maximum leaves a remainder
    of one chunk, and a one-chunk window gives the model a fragment with no
    surrounding exchange -- the worst input shape for a task that depends on
    reading context.
    """
    size = size or settings.extract_window_chunks
    if not chunks:
        return []
    if len(chunks) <= size:
        return [list(chunks)]

    count = -(-len(chunks) // size)  # ceil
    per = -(-len(chunks) // count)
    return [list(chunks[i : i + per]) for i in range(0, len(chunks), per)]


def render_window(chunks: Sequence[Any]) -> str:
    """The text the model sees.

    Chunks overlap, so joining them raw repeats a passage at every boundary --
    which invites the model to extract the same fact twice and makes the
    duplicate look like two pieces of evidence. The overlap is trimmed using
    the document offsets each chunk already carries.
    """
    parts: List[str] = []
    previous_end: Optional[int] = None
    for chunk in chunks:
        metadata = chunk.chunk_metadata or {}
        start = metadata.get("char_start")
        content = chunk.content
        if previous_end is not None and start is not None and start < previous_end:
            content = content[previous_end - start :]
        parts.append(content)
        previous_end = metadata.get("char_end")
    return "\n".join(part for part in parts if part)


def locate(snippet: str, chunks: Sequence[Any]) -> Optional[Tuple[Any, int, int, int]]:
    """Find the snippet in one chunk: (chunk, index, start, end).

    Exact first, then **case-insensitively**. The fallback is not a weakening
    of the verbatim rule, because what gets stored is the *document's*
    characters at the offsets found, never the model's string -- a model
    quoting mid-sentence writes "we'd have to put it out" where the transcript
    has "We'd", and that is the same quote. Gate 0 re-checks the span against
    the chunk on every read, so storing the document's text keeps it passing.

    Nothing looser than case. A whitespace-tolerant or fuzzy search would make
    ``evidence.snippet`` something that is merely *nearly* in the document, and
    the resulting span would fail Gate 0 later, further from the cause.

    The first chunk wins. Chunks overlap, so a snippet near a boundary
    legitimately sits in two of them; either citation verifies, and taking the
    lowest index makes the result deterministic.
    """
    for chunk in chunks:
        at = chunk.content.find(snippet)
        if at != -1:
            return chunk, chunk.chunk_index, at, at + len(snippet)
    folded = snippet.casefold()
    for chunk in chunks:
        at = chunk.content.casefold().find(folded)
        if at != -1:
            return chunk, chunk.chunk_index, at, at + len(snippet)
    return None


def _to_candidate(reported: ExtractedFactOut, chunks: Sequence[Any]) -> CandidateFact:
    candidate = CandidateFact(
        fact_type=reported.fact_type.value,
        content=reported.content.strip(),
        snippet=reported.snippet.strip(),
        speaker=reported.speaker,
        confidence=max(0.0, min(1.0, reported.confidence)),
    )

    found = locate(candidate.snippet, chunks)
    if found is None:
        candidate.rejected = UNLOCATABLE
        return candidate

    chunk, chunk_index, start, end = found
    candidate.chunk_id = chunk.id
    candidate.chunk_index = chunk_index
    candidate.char_start = start
    candidate.char_end = end
    # The document's characters, not the model's string. They differ only in
    # case here, and this is the copy Gate 0 will re-verify forever.
    candidate.snippet = chunk.content[start:end]

    # A payload that fails its type's shape is recorded, not fatal: the claim
    # is still cited and readable, it just cannot be promoted into an amount or
    # a commitment later.
    candidate.payload, candidate.payload_error = narrow_payload(
        candidate.fact_type, reported.payload
    )
    return candidate


async def extract_window(
    chunks: Sequence[Any],
    *,
    meeting_type: str,
    occurred_at: str,
    budget: Optional[client.RunBudget] = None,
    role: str = client.ROLE_PRIMARY,
    prompt: Optional[Any] = None,
) -> Tuple[List[CandidateFact], Optional[client.AIRun]]:
    """One structured call over one window.

    ``role`` and ``prompt`` exist for the eval harness and default to exactly
    what the pipeline used before they were added. Both are overridable for the
    same reason: tasks 11.10 and 11.11 have to score *this* extractor under a
    different prompt version and a different model, and an eval that reaches
    its own copy of the call measures its copy. ``run_extraction_eval.py``'s
    docstring makes the point -- an eval scored against a prompt nothing else
    uses measures nothing -- and that applies just as much to the client call
    around it.
    """
    prompt = prompt or get_prompt("extract")
    result, run = await client.structured(
        ExtractionResult,
        prompt.messages(
            window=render_window(chunks),
            meeting_type=meeting_type,
            occurred_at=occurred_at,
        ),
        task="extract",
        prompt_version=prompt.version,
        role=role,
        reasoning_effort="medium",
        budget=budget,
    )
    return [_to_candidate(fact, chunks) for fact in result.facts], run


async def extract_facts(
    chunks: Sequence[Any],
    *,
    meeting_type: str,
    occurred_at: str,
    budget: Optional[client.RunBudget] = None,
    role: str = client.ROLE_PRIMARY,
    prompt: Optional[Any] = None,
) -> List[CandidateFact]:
    """Extract from every window, concurrently.

    Concurrency is bounded by the governor rather than here: it holds the real
    ceiling (250K TPM as well as a request count), and a second limiter would
    mean neither is authoritative.
    """
    windows = build_windows(chunks)
    if not windows:
        return []

    results = await asyncio.gather(
        *(
            extract_window(
                window, meeting_type=meeting_type, occurred_at=occurred_at,
                budget=budget, role=role, prompt=prompt,
            )
            for window in windows
        )
    )

    candidates: List[CandidateFact] = []
    for window_facts, _run in results:
        candidates.extend(window_facts)

    located = [c for c in candidates if c.located]
    logger.info(
        "extract.done windows=%d reported=%d located=%d unlocatable=%d payload_errors=%d",
        len(windows), len(candidates), len(located),
        sum(1 for c in candidates if c.rejected == UNLOCATABLE),
        sum(1 for c in candidates if c.payload_error),
    )
    return candidates
