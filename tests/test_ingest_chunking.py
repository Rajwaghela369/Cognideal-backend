"""Speaker-aware chunking -- task 2.1. No database.

Two properties matter, and the second is the one that was wrong on the first
attempt: a chunk must *end* on a turn boundary, and it must also *start* on
one. Ending there alone leaves every chunk after the first opening with an
unattributed fragment, because the overlap pulls the start back into the middle
of the previous turn -- and a span quoted from that fragment has no speaker.
"""

import pathlib
import re

from app.services.ingest import (
    chunk_document,
    parse_speaker_turns,
    speakers_in,
    split_into_chunks,
)

TRANSCRIPTS = pathlib.Path(__file__).parent / "fixtures" / "transcripts"
LABEL_START = re.compile(r"^[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}: ")


def transcript(name):
    return (TRANSCRIPTS / name).read_text()


def test_finds_every_speaker():
    turns = parse_speaker_turns(transcript("01-discovery-2026-07-28.txt"))
    assert {speaker for speaker, _, _ in turns} == {
        "Maya Chen", "Priya Raman", "Marcus Webb", "Tom Alvarez",
    }


def test_turn_offsets_are_contiguous_and_include_the_label():
    text = transcript("01-discovery-2026-07-28.txt")
    turns = parse_speaker_turns(text)
    assert turns[0][1] == 0
    for (speaker, start, end), (_, next_start, _) in zip(turns, turns[1:]):
        assert end == next_start, "turns must tile the text with no gaps"
        assert text[start:].startswith(speaker + ":")
    assert turns[-1][2] == len(text)


def test_prose_is_not_treated_as_a_transcript():
    """An email or contract must chunk exactly as it always did.

    Its metadata carries offsets and nothing else. A non-transcript document
    has no turns to attribute, and an empty speakers list would read as
    "nobody spoke" rather than "not applicable".
    """
    prose = "Para one is here.\n\nPara two follows on.\n\n" + ("x" * 2500)
    assert parse_speaker_turns(prose) == []
    for chunk in chunk_document(prose):
        assert set(chunk.metadata) == {"char_start", "char_end"}
    # Unchanged from the paragraph-based behaviour that predates 2.1: the break
    # at offset 39 is before the halfway mark, so it is correctly ignored.
    assert [(c.char_start, c.char_end) for c in chunk_document(prose)] == [
        (0, 1000), (850, 1850), (1700, 2541)
    ]


def test_a_single_colon_line_is_not_a_transcript():
    """One match is far likelier to be prose than a one-speaker transcript."""
    assert parse_speaker_turns("Note: this is a memo, not a call.\n\nBody text.") == []


def test_every_chunk_starts_with_a_speaker_label():
    for name in sorted(p.name for p in TRANSCRIPTS.glob("*.txt")):
        for index, chunk in enumerate(chunk_document(transcript(name))):
            assert LABEL_START.match(chunk.content), (
                "%s chunk %d opens mid-turn: %r" % (name, index, chunk.content[:60])
            )


def test_chunks_cover_the_whole_text_and_overlap():
    text = transcript("01-discovery-2026-07-28.txt")
    chunks = split_into_chunks(text)
    assert chunks[0][1] == 0
    assert chunks[-1][2] == len(text)
    for (_, _, end), (_, next_start, _) in zip(chunks, chunks[1:]):
        assert next_start < end, "consecutive chunks must overlap"


def test_metadata_records_speakers_and_offsets():
    chunks = chunk_document(transcript("03-negotiation-checkin-2026-09-18.txt"))
    for chunk in chunks:
        assert chunk.metadata["char_start"] == chunk.char_start
        assert chunk.metadata["char_end"] == chunk.char_end
        assert chunk.metadata["speakers"], "a transcript chunk must name its speakers"


def test_speaker_is_set_only_when_unambiguous():
    """A chunk spanning four turns has no single speaker, so the field is null.

    Naming one of them would be a lie that a citation would then repeat.
    """
    for chunk in chunk_document(transcript("01-discovery-2026-07-28.txt")):
        if len(chunk.metadata["speakers"]) == 1:
            assert chunk.metadata["speaker"] == chunk.metadata["speakers"][0]
        else:
            assert chunk.metadata["speaker"] is None


def test_speakers_in_is_span_scoped():
    text = transcript("01-discovery-2026-07-28.txt")
    turns = parse_speaker_turns(text)
    first_turn_end = turns[0][2]
    assert speakers_in(turns, 0, first_turn_end) == ["Maya Chen"]
    assert len(speakers_in(turns, 0, len(text))) == 4
