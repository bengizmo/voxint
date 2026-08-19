"""Read-time correction provenance over real Postgres (issue #83).

The pure ``test_corrections_view.py`` unit tests cover the resolution/reconciliation
logic on hand-built envelopes; this asserts the DURABLE round-trip the console
actually reads: ``enhance_match.run`` persists a real ``correction_trace`` +
``corrector_version``, the run's frozen ``domain_pack`` snapshot resolves the fired
rule to its pack, and the shared ``_island_segment`` builder + ``run_reconciliation``
surface it — from the stored columns, no re-diffing of text.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder
from voxint.adjudication.corrections_view import run_reconciliation
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.api.app import (
    _island_segment,
    _load_run_rule_index,
)
from voxint.db.models import MediaItem, PipelineRun, TranscriptSegment
from voxint.domain_packs.base import DomainPack
from voxint.domain_packs.corrections import CorrectionRule
from voxint.pipeline.stages import enhance_match
from voxint.pipeline.stages.context import StageContext

ZB_RULE = CorrectionRule(id="zb", match="zoom board", replace="Zoning Board")
GHOST_RULE = CorrectionRule(id="ghost", match="quorum", replace="Quorum")
# Two declared rules: zb fires on the first segment; ghost matches no raw text.
NEWSROOM = DomainPack(name="newsroom", corrections=(ZB_RULE, GHOST_RULE))

_FIRES_RAW = "the zoom board met"
_QUIET_RAW = "nothing to correct here"


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


def _make_run(session: Session, pack: DomainPack) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    # The frozen per-run pack snapshot the console resolves provenance against.
    run = PipelineRun(media_item_id=media.id, domain_pack=pack.to_mapping())
    session.add(run)
    session.flush()
    return run.id


def _add_segment(session: Session, run_id: uuid.UUID, *, index: int, raw: str) -> uuid.UUID:
    seg = TranscriptSegment(
        pipeline_run_id=run_id,
        segment_index=index,
        start_seconds=float(index),
        end_seconds=float(index) + 1.0,
        raw_text=raw,
        diarization_label="SPEAKER_00",
    )
    session.add(seg)
    session.flush()
    return seg.id


def _ctx(pack: DomainPack) -> StageContext:
    # LLM off: the rules-only raw-base envelope is what gets persisted.
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=None,
        media_root=Path("/data/media"),
        domain_pack=pack,
    )


def test_island_payload_resolves_fired_rule_to_its_pack(session: Session) -> None:
    run_id = _make_run(session, NEWSROOM)
    _add_segment(session, run_id, index=0, raw=_FIRES_RAW)
    _add_segment(session, run_id, index=1, raw=_QUIET_RAW)

    enhance_match.run(_ctx(NEWSROOM), session, run_id)
    session.expire_all()

    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    index = _load_run_rule_index(session, run_id)
    assert index is not None and index.pack == "newsroom"
    payload = [_island_segment(ln, {}, index) for ln in lines]

    fired = next(p for p in payload if p["rawText"] == _FIRES_RAW)
    quiet = next(p for p in payload if p["rawText"] == _QUIET_RAW)

    # The fired segment carries resolved provenance from the stored trace + snapshot.
    assert fired["corrections"]["status"] == "shown"
    assert fired["corrections"]["inputBase"] == "raw"
    (entry,) = fired["corrections"]["entries"]
    assert entry["id"] == "zb"
    assert entry["pack"] == "newsroom"
    assert entry["from"] == "zoom board"
    assert entry["to"] == "Zoning Board"
    assert entry["resolved"] is True

    # The untouched segment materially corrected nothing -> no marker, raw present.
    assert quiet["corrections"] is None
    assert quiet["rawText"] == _QUIET_RAW


def test_reconciliation_surfaces_applied_and_declared_but_never_fired(
    session: Session,
) -> None:
    run_id = _make_run(session, NEWSROOM)
    _add_segment(session, run_id, index=0, raw=_FIRES_RAW)
    _add_segment(session, run_id, index=1, raw=_QUIET_RAW)
    enhance_match.run(_ctx(NEWSROOM), session, run_id)
    session.expire_all()

    index = _load_run_rule_index(session, run_id)
    raw_texts = (
        session.execute(
            select(TranscriptSegment.raw_text).where(
                TranscriptSegment.pipeline_run_id == run_id
            )
        )
        .scalars()
        .all()
    )
    recon = {r["id"]: r for r in run_reconciliation(index, raw_texts)}

    assert recon["zb"]["status"] == "applied"
    assert recon["zb"]["appliedCount"] == 1
    # ghost is declared in the pack but never matches any segment's raw text.
    assert recon["ghost"]["status"] == "no_raw_match"
    assert recon["ghost"]["appliedCount"] == 0


def test_null_snapshot_yields_no_provenance_not_a_default_pack(session: Session) -> None:
    # A legacy run with no frozen snapshot must show NO provenance (honest
    # "unavailable"), never a fabricated default-pack resolution.
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)  # domain_pack left NULL
    session.add(run)
    session.flush()
    _add_segment(session, run.id, index=0, raw=_FIRES_RAW)

    assert _load_run_rule_index(session, run.id) is None
    assert run_reconciliation(_load_run_rule_index(session, run.id), [_FIRES_RAW]) == []
