"""The enhance_match dual pass, persisted (issue #82, epic #78).

Drives ``enhance_match.run`` against real Postgres and asserts the DURABLE
correction surface the pure ``test_enhance_match_composition.py`` unit tests
cannot: the ``enhanced_text`` / ``correction_trace`` / ``corrector_version``
columns, their atomic re-enhance reset, and the split machinery reading the
stored trace. The nuanced LLM-behavior cases (invents/undoes/amplifies) live in
the unit corpus, where the enhanced text is a fixture value; here the LLM is the
shared ``FakeLLM`` (capitalizes each segment), ``FailingLLM``, or ``None``.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FailingLLM, FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.adjudication.splits import (
    UnsplittableError,
    derive_children,
    record_split,
    splittable_words,
    word_count,
)
from voxint.adjudication.transcript import effective_text
from voxint.clients.base import LLMClient
from voxint.db.models import MediaItem, PipelineRun, TranscriptSegment
from voxint.domain_packs.base import DomainPack
from voxint.domain_packs.corrections import CorrectionRule
from voxint.pipeline.stages import enhance_match
from voxint.pipeline.stages.context import StageContext

GENERIC = DomainPack(name="generic")
ZB_RULE = CorrectionRule(id="zb", match="zoom board", replace="Zoning Board")
ZB_PACK = DomainPack(name="newsroom", corrections=(ZB_RULE,))

# faster-whisper word strings carry a leading space (except the first); raw_text
# is their exact concatenation, which is what makes a segment splittable.
_ZOOM_WORDS = [
    {"start": 0.0, "end": 1.0, "word": "the", "confidence": 0.9},
    {"start": 1.0, "end": 2.0, "word": " zoom", "confidence": 0.9},
    {"start": 2.0, "end": 3.0, "word": " board", "confidence": 0.9},
    {"start": 3.0, "end": 4.0, "word": " met", "confidence": 0.9},
]
_ZOOM_RAW = "the zoom board met"


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


def make_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.flush()
    return run.id


def add_segment(
    session: Session,
    run_id: uuid.UUID,
    *,
    index: int,
    raw_text: str,
    words: list[dict[str, object]] | None = None,
    start: float = 0.0,
    end: float = 4.0,
) -> uuid.UUID:
    seg = TranscriptSegment(
        pipeline_run_id=run_id,
        segment_index=index,
        start_seconds=start,
        end_seconds=end,
        raw_text=raw_text,
        diarization_label="SPEAKER_00",
        words=words,
    )
    session.add(seg)
    session.flush()
    return seg.id


def build_ctx(llm: LLMClient | None, pack: DomainPack) -> StageContext:
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=llm,
        media_root=Path("/data/media"),
        domain_pack=pack,
    )


def _reload(session: Session, seg_id: uuid.UUID) -> TranscriptSegment:
    session.expire_all()
    return session.execute(
        select(TranscriptSegment).where(TranscriptSegment.id == seg_id)
    ).scalar_one()


# --------------------------------------------------------------------------- #
# Deferred #81 Gate-A identity: a no-LLM, no-rules segment persists NOTHING and
# stays byte-identical, even with tricky Unicode.
# --------------------------------------------------------------------------- #
def test_gate_a_identity_no_llm_no_rules(session: Session) -> None:
    run_id = make_run(session)
    raw = "café — “农业” zoë naïve"  # curly quotes, em dash, CJK, combining marks
    seg_id = add_segment(session, run_id, index=0, raw_text=raw)

    enhance_match.run(build_ctx(None, GENERIC), session, run_id)

    seg = _reload(session, seg_id)
    assert seg.enhanced_text is None
    assert seg.correction_trace == []
    assert seg.corrector_version is None
    # effective_text is the ONE selector exports + search share; it must be the
    # raw text, byte-identical (no JSON/ensure_ascii mangling of the no-op).
    assert effective_text(seg, None) == raw


# --------------------------------------------------------------------------- #
# Rules-only (LLM off): the raw-base envelope is persisted, split is disabled,
# and export/search read the corrected text.
# --------------------------------------------------------------------------- #
def test_rules_only_persists_raw_envelope_and_disables_split(session: Session) -> None:
    run_id = make_run(session)
    corrected_id = add_segment(
        session, run_id, index=0, raw_text=_ZOOM_RAW, words=_ZOOM_WORDS
    )
    # Control: same shape, no rule matches — proves the guard fires on the
    # correction specifically, not on the word geometry.
    control_words = [
        {"start": 0.0, "end": 1.0, "word": "hello", "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "word": " there", "confidence": 0.9},
        {"start": 2.0, "end": 3.0, "word": " folks", "confidence": 0.9},
    ]
    control_id = add_segment(
        session, run_id, index=1, raw_text="hello there folks", words=control_words
    )

    enhance_match.run(build_ctx(None, ZB_PACK), session, run_id)

    corrected = _reload(session, corrected_id)
    assert corrected.enhanced_text == "the Zoning Board met"
    assert corrected.correction_trace == {
        "version": 1,
        "input_base": "raw",
        "entries": [
            {"id": "zb", "from": "zoom board", "to": "Zoning Board", "span": [4, 16]}
        ],
    }
    assert corrected.corrector_version == 1
    # Split disabled by the stored trace; export/search read the corrected text.
    assert splittable_words(corrected) is None
    assert word_count(corrected) is None
    assert effective_text(corrected, None) == "the Zoning Board met"
    with pytest.raises(UnsplittableError):
        record_split(session, parent=corrected, word_index=1, operator="op")

    # Control stays a clean no-op and remains splittable.
    control = _reload(session, control_id)
    assert control.enhanced_text is None
    assert control.correction_trace == []
    assert control.corrector_version is None
    assert word_count(control) == 3


# --------------------------------------------------------------------------- #
# corrector_version=1 for a materially-changed pure-LLM row (decision B), and the
# text-diff split guard covers it (empty entries, so the trace guard does not).
# --------------------------------------------------------------------------- #
def test_pure_llm_enhancement_versions_and_disables_split(session: Session) -> None:
    run_id = make_run(session)
    # generic pack: no rule fires, so any change is the LLM's. FakeLLM capitalizes.
    seg_id = add_segment(
        session, run_id, index=0, raw_text=_ZOOM_RAW, words=_ZOOM_WORDS
    )

    enhance_match.run(build_ctx(FakeLLM(), GENERIC), session, run_id)

    seg = _reload(session, seg_id)
    assert seg.enhanced_text == "The zoom board met"  # LLM's own edit
    assert seg.correction_trace == {"version": 1, "input_base": "llm", "entries": []}
    assert seg.corrector_version == 1  # the engine ran, even with no rule fired
    # Empty entries -> the trace guard is silent; the text-diff guard disables it.
    assert splittable_words(seg) is None


# --------------------------------------------------------------------------- #
# Atomic re-enhance reset: a second run with the rule removed clears all three
# columns together, so a stale correction never outlives its rule.
# --------------------------------------------------------------------------- #
def test_atomic_reenhance_reset_clears_correction(session: Session) -> None:
    run_id = make_run(session)
    seg_id = add_segment(session, run_id, index=0, raw_text=_ZOOM_RAW)

    enhance_match.run(build_ctx(None, ZB_PACK), session, run_id)
    seg = _reload(session, seg_id)
    assert seg.enhanced_text == "the Zoning Board met"
    assert seg.corrector_version == 1

    # Re-run with no rules and a failed LLM: the correction must be fully cleared.
    enhance_match.run(build_ctx(FailingLLM(), GENERIC), session, run_id)
    seg = _reload(session, seg_id)
    assert seg.enhanced_text is None
    assert seg.correction_trace == []
    assert seg.corrector_version is None


# --------------------------------------------------------------------------- #
# An already-split parent that later gets a correction renders WHOLE — the read
# path derives no misaligned children (mirrors the #59 already-split guard).
# --------------------------------------------------------------------------- #
def test_split_then_correct_renders_whole(session: Session) -> None:
    run_id = make_run(session)
    seg_id = add_segment(
        session, run_id, index=0, raw_text=_ZOOM_RAW, words=_ZOOM_WORDS
    )
    seg = _reload(session, seg_id)
    # Splittable before any correction; record a cut.
    assert word_count(seg) == 4
    record_split(session, parent=seg, word_index=2, operator="op")
    session.flush()

    # Now a correction fires on the same segment.
    enhance_match.run(build_ctx(None, ZB_PACK), session, run_id)

    seg = _reload(session, seg_id)
    assert seg.corrector_version == 1
    # The stored cut still exists, but the segment is no longer splittable, so the
    # read path renders it whole rather than deriving children at stale offsets.
    assert splittable_words(seg) is None
    assert derive_children(seg, [2]) is None
