"""The clean-up source loader and the sole generation writer (#758 slice 2)."""

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_translation_jobs import seed_run
from voxint.adjudication.attribution import walk_attributions
from voxint.adjudication.splits import record_split
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.db.models import RunCleanup, TranscriptSegment
from voxint.enrichment import cleanups
from voxint.enrichment.cleanups import (
    PROPOSAL_SLACK_CHARS,
    CleanupError,
    CleanupSource,
    ConflictingReplayError,
    cleanup_omits,
    cleanup_texts,
    current_cleanup,
    filler_list_snapshot,
    load_cleanup_source,
    record_cleanup,
)
from voxint.enrichment.translations import (
    TranslationError,
    load_translation_source,
    translation_source_hash,
)
from voxint.export.filler_lists import DEFAULT_FILLER_LIST

TEXTS = (
    "I mean, I think it's fine.",
    "Um, so we went to Paris, uh, yesterday.",
    "Okay then.",
    "So -- we go.",
)
# Index 2 has no word timings, so it is anchorless; index 1 is split in two.
TIMED = (0, 1, 3)
CONFIG: dict[str, Any] = {
    "prompt_version": 1,
    "filler_list": filler_list_snapshot(DEFAULT_FILLER_LIST),
    "batch_max_segments": 20,
}


def seed(session: Session, texts: Sequence[str] = TEXTS) -> uuid.UUID:
    run_id = seed_run(session, texts=list(texts))
    segments = session.scalars(
        select(TranscriptSegment)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .order_by(TranscriptSegment.segment_index)
    ).all()
    for index, segment in enumerate(segments):
        if index not in TIMED:
            continue
        start = segment.start_seconds
        segment.words = [
            {
                "word": (" " if i else "") + word,
                "start": start + i * 0.1,
                "end": start + i * 0.1 + 0.1,
            }
            for i, word in enumerate(segment.raw_text.split())
        ]
    session.flush()
    if len(segments) > 1:
        record_split(session, parent=segments[1], word_index=4, operator="test")
    session.commit()
    return run_id


def record(
    session: Session,
    source: CleanupSource,
    proposals: Mapping[int, str | None] | None = None,
    *,
    exact: bool = False,
    config: Mapping[str, Any] | None = None,
    model: str = "test-model",
    idempotency_key: str | None = None,
    started_at: datetime | None = None,
) -> RunCleanup:
    """Record ``proposals``; unless ``exact``, unlisted lines propose their source."""
    given: dict[int, str | None] = dict(proposals or {})
    if not exact:
        sources: dict[int, str | None] = {
            line.line.line_index: line.line.text for line in source.lines
        }
        given = sources | given
    started = started_at or datetime(2026, 10, 6, 9, tzinfo=UTC)
    row = record_cleanup(
        session,
        source=source,
        proposals=given,
        config=CONFIG if config is None else config,
        model=model,
        producer="cleanup.llm",
        producer_version="1",
        started_at=started,
        completed_at=started.astimezone(UTC) + timedelta(seconds=3),
        idempotency_key=idempotency_key or str(uuid.uuid4()),
    )
    session.commit()
    return row


def stored(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(RunCleanup)) or 0


def test_source_pairs_every_line_with_its_emission_anchors(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = seed(session)
        source = load_cleanup_source(session, run_id)
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        assert [line.line.text for line in source.lines] == [line.text for line in lines]
        assert [(s.line.word_start, s.line.word_end) for s in source.lines] == [
            (None, None), (0, 4), (4, 8), (None, None), (None, None),
        ]
        assert [
            [word.text for word in line.words.words] for line in source.lines
        ] == [
            ["I", "mean,", "I", "think", "it's", "fine."],
            ["Um,", "so", "we", "went"],
            ["to", "Paris,", "uh,", "yesterday."],
            [],
            ["So", "--", "we", "go."],
        ]
        for line in source.lines:
            assert all(w.anchor.segment_id == line.line.segment_id for w in line.words.words)
        # The freshness hash is the translation hash, unchanged.
        assert translation_source_hash(source.translation) == translation_source_hash(
            load_translation_source(session, run_id)
        )


def test_source_refuses_unknown_and_empty_runs(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        with pytest.raises(CleanupError, match="unknown pipeline run"):
            load_cleanup_source(session, uuid.uuid4())
        run_id = seed_run(session, texts=[])
        with pytest.raises(CleanupError, match="nothing to clean up"):
            load_cleanup_source(session, run_id)


def test_source_refuses_a_walk_that_disagrees_with_the_lines(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id = seed(session)
        # One emission fewer than lines: a commit landed between the reads.
        monkeypatch.setattr(
            cleanups, "walk_attributions", lambda s, r: list(walk_attributions(s, r))[:-1]
        )
        with pytest.raises(CleanupError, match="changed while it was being read"):
            load_cleanup_source(session, run_id)
        monkeypatch.setattr(
            cleanups, "walk_attributions", lambda s, r: list(walk_attributions(s, r))[::-1]
        )
        with pytest.raises(CleanupError, match="changed while it was being read"):
            load_cleanup_source(session, run_id)


def test_source_bounds_and_translation_race(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id = seed(session)
        monkeypatch.setattr(cleanups, "MAX_LINES", 4)
        with pytest.raises(CleanupError, match="5 transcript lines against the 4-line bound"):
            load_cleanup_source(session, run_id)
        monkeypatch.undo()

        def vanished(session: Session, run_id: uuid.UUID) -> None:
            raise TranslationError("run has no transcript")

        # The segments disappeared between the walk and the line read.
        monkeypatch.setattr(cleanups, "load_translation_source", vanished)
        with pytest.raises(CleanupError, match="changed while it was being read"):
            load_cleanup_source(session, run_id)


def test_records_a_generation_with_derived_counts(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = seed(session)
        source = load_cleanup_source(session, run_id)
        row = record(session, source, {
            0: "i think its fine",            # deletes "I mean,"
            1: "so we went",                  # deletes "Um,"
            2: "to Paris yesterday",          # deletes "uh,"
            3: "Okay.",                       # anchorless and changed
            4: "we go",                       # deletes "So" (keeps punctuation-only "--")
        })
        assert row.generation == 1
        assert row.superseded_by_cleanup_id is None
        assert row.source_content_hash == translation_source_hash(source.translation)
        assert row.counts == {
            "lines": 5,
            "lines_changed": 4,
            "words_removed": 5,
            "rejected": {
                "not_deletion": 0, "protected": 0, "whole_line": 0,
                "unanchored": 1, "malformed": 0,
            },
        }
        assert cleanup_texts(row) == [
            "I think it's fine.", "So we went", "to Paris, yesterday.", "Okay then.",
            "-- we go.",
        ]
        first = row.lines[0]
        assert first == {
            "i": 0, "segment_id": str(source.lines[0].line.segment_id),
            "word_start": None, "word_end": None,
            "source": "I mean, I think it's fine.", "text": "I think it's fine.",
            "deleted": [[0, 1, 0, 1], [1, 2, 2, 7]], "outcome": "changed", "reason": None,
        }
        assert row.lines[3]["outcome"] == "rejected"
        assert row.lines[3]["reason"] == "unanchored"
        segment_1 = source.lines[1].line.segment_id
        assert segment_1 is not None
        segment_0, segment_4 = source.lines[0].line.segment_id, source.lines[4].line.segment_id
        assert cleanup_omits(row) == {
            (segment_0, 0, 1): "omit", (segment_0, 1, 2): "omit",
            (segment_1, 0, 1): "omit", (segment_1, 6, 7): "omit",
            (segment_4, 0, 1): "omit",
        }
        assert current_cleanup(session, run_id) == row
        assert row.config == CONFIG


def test_counts_record_every_rejection_and_malformed_line(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = seed(session)
        source = load_cleanup_source(session, run_id)
        row = record(session, source, {
            0: "I mean I think it is fine",    # "is" is not in the source
            2: "to yesterday",                 # drops the protected name Paris
            4: "--",                           # deletes every keyed word
        })
        assert [line["reason"] for line in row.lines] == [
            "not_deletion", None, "protected", None, "whole_line",
        ]
        assert row.counts["rejected"] == {
            "not_deletion": 1, "protected": 1, "whole_line": 1,
            "unanchored": 0, "malformed": 0,
        }
        assert all(line["text"] == line["source"] for line in row.lines)
        malformed: dict[int, str | None] = dict.fromkeys(range(5))
        malformed[1] = source.lines[1].line.text
        other = record(session, source, malformed)
        assert other.counts["rejected"]["malformed"] == 4
        assert other.counts["lines_changed"] == 0
        assert other.generation == 2


def test_new_generation_supersedes_the_head(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = seed(session)
        source = load_cleanup_source(session, run_id)
        first = record(session, source)
        second = record(session, source, {0: "I think it's fine."})
        session.refresh(first)
        assert first.superseded_by_cleanup_id == second.id
        assert second.generation == 2
        assert current_cleanup(session, run_id) == second


def test_replay_adopts_identical_and_refuses_conflicting(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = seed(session)
        source = load_cleanup_source(session, run_id)
        row = record(session, source, {0: "I think it's fine."}, idempotency_key="k1")
        again = record(session, source, {0: "I think it's fine."}, idempotency_key="k1")
        assert again.id == row.id
        assert stored(session) == 1
        for kwargs in (
            {"proposals": {0: "i mean i think its fine"}},
            {"proposals": {0: "I think it's fine."}, "model": "other-model"},
            {"proposals": {0: "I think it's fine."}, "config": {**CONFIG, "prompt_version": 2}},
            {
                "proposals": {0: "I think it's fine."},
                "started_at": datetime(2026, 10, 6, 10, tzinfo=UTC),
            },
        ):
            with pytest.raises(ConflictingReplayError):
                record(session, source, idempotency_key="k1", **kwargs)
            session.rollback()
        assert stored(session) == 1


def test_replay_is_type_sensitive(session_factory: sessionmaker[Session]) -> None:
    """``True == 1`` in Python, but a boolean knob is not a number."""
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        record(session, source, idempotency_key="k", config={**CONFIG, "knob": 1})
        with pytest.raises(ConflictingReplayError):
            record(session, source, idempotency_key="k", config={**CONFIG, "knob": True})


def test_replay_survives_a_dst_fold(session_factory: sessionmaker[Session]) -> None:
    folded = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=ZoneInfo("America/New_York"))
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        row = record(session, source, idempotency_key="k", started_at=folded)
    with session_factory() as session:
        again = record(session, source, idempotency_key="k", started_at=folded)
        assert again.id == row.id
        assert again.started_at == folded.astimezone(UTC)


@pytest.mark.parametrize(
    ("proposals", "message"),
    [
        ({0: 7}, "proposal is not a string"),
        ({3: "Okay\x00then."}, "proposal contains NUL"),
        ({0: "x" * (len(TEXTS[0]) + PROPOSAL_SLACK_CHARS + 1)}, "-char bound"),
        ({99: "x"}, "do not cover the source exactly"),
        ({"0": "x"}, "do not cover the source exactly"),
    ],
)
def test_writer_refuses_malformed_proposals(
    session_factory: sessionmaker[Session], proposals: dict[Any, Any], message: str,
) -> None:
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        with pytest.raises(CleanupError, match=message):
            record(session, source, proposals)
        # Nothing reached the transaction: committing it stores nothing.
        session.commit()
    with session_factory() as fresh:
        assert stored(fresh) == 0


def test_writer_refuses_missing_lines_and_an_all_malformed_generation(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        with pytest.raises(CleanupError, match="missing \\[2\\]"):
            record(session, source, {i: "x" for i in (0, 1, 3, 4)}, exact=True)
        with pytest.raises(CleanupError, match="no line received a usable reply"):
            record(session, source, dict.fromkeys(range(5)), exact=True)
        session.commit()
    with session_factory() as fresh:
        assert stored(fresh) == 0


def test_the_writer_chooses_the_deleted_occurrence(
    session_factory: sessionmaker[Session],
) -> None:
    """A duplicate is resolved by the validator, never by the submitter."""
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session, texts=["Uh, uh we go."]))
        row = record(session, source, {0: "uh we go"})
        assert row.lines[0]["deleted"] == [[0, 1, 0, 3]]
        assert row.lines[0]["text"] == "Uh we go."


_CYCLE: list[object] = []
_CYCLE.append(_CYCLE)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ([], "must be an object"),
        ({"filler_list": {}}, "prompt_version"),
        ({"prompt_version": True, "filler_list": {}}, "prompt_version"),
        ({"prompt_version": 0, "filler_list": {}}, "prompt_version"),
        ({"prompt_version": 1}, "filler_list"),
        ({**CONFIG, "x": object()}, "JSON-serializable"),
        ({**CONFIG, "x": _CYCLE}, "JSON-serializable"),
        ({**CONFIG, "x": 10**5000}, "JSON-serializable"),
        ({**CONFIG, "x": 1e20}, "only strings without NUL"),
        ({**CONFIG, "x": float("nan")}, "only strings without NUL"),
        ({**CONFIG, "x": ["a\x00b"]}, "only strings without NUL"),
        ({**CONFIG, "x": {1: "a"}}, "only strings without NUL"),
        ({**CONFIG, "padding": ["um"] * 20_000}, "bytes"),
        ({"prompt_version": 1, "filler_list": {}}, "preset_version"),
        ({**CONFIG, "filler_list": {**CONFIG["filler_list"], "kept": "um"}}, "kept"),
        ({**CONFIG, "filler_list": {**CONFIG["filler_list"], "words": [1]}}, "words"),
    ],
)
def test_writer_refuses_bad_config(
    session_factory: sessionmaker[Session], config: Any, message: str,
) -> None:
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        with pytest.raises(CleanupError, match=message):
            record(session, source, config=config)
        # Nothing reached the transaction: committing it stores nothing.
        session.commit()
    with session_factory() as fresh:
        assert stored(fresh) == 0


def test_writer_accepts_any_mapping_config(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        row = record(session, source, config=MappingProxyType(CONFIG))
        assert row.config == CONFIG


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"idempotency_key": " "}, "idempotency_key"),
        ({"model": ""}, "model"),
        ({"model": "m" * 201}, "model"),
        ({"producer": " "}, "producer"),
        ({"started_at": datetime(2026, 10, 6, 9)}, "timezone-aware"),
        ({"completed_at": datetime(2026, 10, 6, 8, tzinfo=UTC)}, "precedes"),
    ],
)
def test_writer_refuses_bad_provenance(
    session_factory: sessionmaker[Session], kwargs: dict[str, Any], message: str,
) -> None:
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        arguments: dict[str, Any] = {
            "source": source,
            "proposals": {line.line.line_index: line.line.text for line in source.lines},
            "config": CONFIG,
            "model": "m",
            "producer": "cleanup.llm",
            "producer_version": "1",
            "started_at": datetime(2026, 10, 6, 9, tzinfo=UTC),
            "completed_at": datetime(2026, 10, 6, 9, 1, tzinfo=UTC),
            "idempotency_key": "k",
        } | kwargs
        with pytest.raises(CleanupError, match=message):
            record_cleanup(session, **arguments)
        # Nothing reached the transaction: committing it stores nothing.
        session.commit()
    with session_factory() as fresh:
        assert stored(fresh) == 0


def test_payload_cap(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cleanups, "MAX_LINES_PAYLOAD_BYTES", 100)
    with session_factory() as session:
        source = load_cleanup_source(session, seed(session))
        with pytest.raises(CleanupError, match="payload over 100 bytes"):
            record(session, source)
        # Nothing reached the transaction: committing it stores nothing.
        session.commit()
    with session_factory() as fresh:
        assert stored(fresh) == 0
