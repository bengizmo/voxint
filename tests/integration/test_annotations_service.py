"""The annotation DB writer + read resolver against real Postgres (issue #86).

Exercises ``voxint.adjudication.annotations`` capture/update/reanchor/refresh/
soft-delete/list against the migrated schema and the CHECK truth table, plus a
round trip through ``resolve_annotation_spans`` over the real
``attributed_transcript`` render. Routes are Step 3; this is the writer gate.
"""

import uuid

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.annotations import (
    SEGMENT_RANGE,
    TEXT_RANGE,
    WORD_RANGE,
    AnnotationIdempotencyError,
    AnnotationNotFoundError,
    AnnotationStaleError,
    AnnotationValidationError,
    CaptureEndpoint,
    CapturePayload,
    annotations_for_run,
    capture_annotation,
    load_covered_segments,
    reanchor_annotation,
    refresh_annotation,
    resolve_annotation_spans,
    soft_delete_annotation,
    stored_anchor_from_row,
    update_annotation,
)
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.db.models import (
    AnnotationTag,
    AnnotationTagLink,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    TranscriptAnnotation,
    TranscriptSegment,
)

# seg0 "Hello world there": content_start=[0,6,12] content_end=[5,11,17] len 17.
# seg1 "how are you":       content_start=[0,4,8]  content_end=[3,7,11] len 11.
_SEG0 = "Hello world there"
_SEG1 = "how are you"


def _tokens(raw: str, start: float, end: float) -> list[dict[str, object]]:
    pieces = raw.split(" ")
    step = (end - start) / len(pieces)
    out: list[dict[str, object]] = []
    t = start
    for i, w in enumerate(pieces):
        out.append(
            {"word": (w if i == 0 else " " + w), "start": round(t, 6), "end": round(t + step, 6)}
        )
        t += step
    return out


def _seed_run(session: Session) -> tuple[uuid.UUID, list[uuid.UUID]]:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    seg_ids: list[uuid.UUID] = []
    for index, (raw, (lo, hi)) in enumerate([(_SEG0, (0.0, 3.0)), (_SEG1, (3.0, 6.0))]):
        seg = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=index,
            start_seconds=lo,
            end_seconds=hi,
            raw_text=raw,
            diarization_label="S0",
            words=_tokens(raw, lo, hi),
        )
        session.add(seg)
        session.flush()
        seg_ids.append(seg.id)
    session.commit()
    return run.id, seg_ids


def _seed_run_segments(session: Session, raws: list[str]) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """A run with arbitrary short raw segments (each with word timings)."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    seg_ids: list[uuid.UUID] = []
    for index, raw in enumerate(raws):
        lo, hi = float(index * 3), float(index * 3 + 3)
        seg = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=index,
            start_seconds=lo,
            end_seconds=hi,
            raw_text=raw,
            diarization_label="S0",
            words=_tokens(raw, lo, hi),
        )
        session.add(seg)
        session.flush()
        seg_ids.append(seg.id)
    session.commit()
    return run.id, seg_ids


def _ep(seg_id: uuid.UUID, offset: int, child: tuple[int, int] | None = None) -> CaptureEndpoint:
    cws, cwe = child if child is not None else (None, None)
    return CaptureEndpoint(
        segment_id=seg_id, offset=offset, child_word_start=cws, child_word_end=cwe
    )


def _cap(start: CaptureEndpoint, end: CaptureEndpoint, quote: str) -> CapturePayload:
    return CapturePayload(start, end, quote)


def _add_tag(session: Session, name: str, color: int = 0) -> uuid.UUID:
    tag = AnnotationTag(name=name, name_normalized=name.strip().casefold(), color=color)
    session.add(tag)
    session.flush()
    return tag.id


def _correct(session: Session, run_id: uuid.UUID, seg_id: uuid.UUID, text: str) -> None:
    from datetime import UTC, datetime

    session.add(
        SegmentReviewState(
            transcript_segment_id=seg_id,
            pipeline_run_id=run_id,
            corrected_text=text,
            corrected_at=datetime.now(UTC),
        )
    )
    session.commit()


# --------------------------------------------------------------------------- #
# capture: classification persists
# --------------------------------------------------------------------------- #


def test_capture_word_range_persists(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=2,
        )
        session.commit()
        assert row.anchor_kind == WORD_RANGE
        assert (row.start_word_index, row.end_word_index) == (1, 2)
        assert row.start_char_offset is None
        assert row.start_seconds == pytest.approx(1.0) and row.end_seconds == pytest.approx(2.0)
        assert row.quote_text == "world"
        assert row.color_index == 2
        assert len(row.source_text_hash) == 64


def test_capture_text_range_persists(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 7), _ep(segs[0], 11), "orld"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        assert row.anchor_kind == TEXT_RANGE
        assert (row.start_char_offset, row.end_char_offset) == (7, 11)
        assert row.start_seconds is None and row.end_seconds is None


def test_capture_segment_range_persists(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[0], 17), _SEG0),
            operator="op",
            nonce="n1",
            color_index=1,
        )
        session.commit()
        assert row.anchor_kind == SEGMENT_RANGE
        assert row.start_word_index is None and row.start_char_offset is None
        assert row.start_seconds is None


# --------------------------------------------------------------------------- #
# capture: idempotency + validation + fail-closed
# --------------------------------------------------------------------------- #


def test_capture_replay_same_payload_returns_same_row(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        payload = _cap(_ep(segs[0], 6), _ep(segs[0], 11), "world")
        a = capture_annotation(
            session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=0
        )
        session.commit()
        b = capture_annotation(
            session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=0
        )
        session.commit()
        assert a.id == b.id
        count = session.query(TranscriptAnnotation).filter_by(pipeline_run_id=run_id).count()
        assert count == 1


def test_capture_replay_different_payload_conflicts(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        payload = _cap(_ep(segs[0], 6), _ep(segs[0], 11), "world")
        capture_annotation(
            session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=0
        )
        session.commit()
        with pytest.raises(AnnotationIdempotencyError):
            capture_annotation(
                session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=5
            )


def test_capture_replay_after_soft_delete_returns_deleted_row(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        payload = _cap(_ep(segs[0], 6), _ep(segs[0], 11), "world")
        a = capture_annotation(
            session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=0
        )
        session.commit()
        soft_delete_annotation(session, run_id=run_id, annotation_id=a.id)
        session.commit()
        b = capture_annotation(
            session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=0
        )
        session.commit()
        assert b.id == a.id
        assert b.deleted_at is not None  # not resurrected
        count = session.query(TranscriptAnnotation).filter_by(pipeline_run_id=run_id).count()
        assert count == 1


def test_capture_forged_cross_run_endpoint_is_404(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_a, segs_a = _seed_run(session)
        _run_b, segs_b = _seed_run(session)
        with pytest.raises(AnnotationNotFoundError):
            capture_annotation(
                session,
                run_id=run_a,
                payload=_cap(_ep(segs_b[0], 6), _ep(segs_a[0], 11), "world"),
                operator="op",
                nonce="n1",
                color_index=0,
            )


def test_capture_client_quote_mismatch_is_stale(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        with pytest.raises(AnnotationStaleError):
            capture_annotation(
                session,
                run_id=run_id,
                payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "wrong"),
                operator="op",
                nonce="n1",
                color_index=0,
            )


def test_capture_bad_color_is_422(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        with pytest.raises(AnnotationValidationError):
            capture_annotation(
                session,
                run_id=run_id,
                payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
                operator="op",
                nonce="n1",
                color_index=6,
            )


def test_capture_unknown_tag_is_404(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        with pytest.raises(AnnotationNotFoundError):
            capture_annotation(
                session,
                run_id=run_id,
                payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
                operator="op",
                nonce="n1",
                color_index=0,
                tag_ids=[uuid.uuid4()],
            )


def test_capture_too_many_tags_is_422(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        tag_ids = [_add_tag(session, f"t{i}") for i in range(9)]
        session.commit()
        with pytest.raises(AnnotationValidationError):
            capture_annotation(
                session,
                run_id=run_id,
                payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
                operator="op",
                nonce="n1",
                color_index=0,
                tag_ids=tag_ids,
            )


def test_capture_attaches_tags(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        t1 = _add_tag(session, "Key point")
        t2 = _add_tag(session, "Follow up")
        session.commit()
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
            tag_ids=[t1, t2, t1],  # duplicate collapses
        )
        session.commit()
        links = session.query(AnnotationTagLink).filter_by(annotation_id=row.id).count()
        assert links == 2


# --------------------------------------------------------------------------- #
# update / reanchor / refresh / delete
# --------------------------------------------------------------------------- #


def test_update_replaces_metadata(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        t1 = _add_tag(session, "one")
        t2 = _add_tag(session, "two")
        session.commit()
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
            tag_ids=[t1],
        )
        session.commit()
        updated = update_annotation(
            session,
            run_id=run_id,
            annotation_id=row.id,
            color_index=3,
            note="a margin note",
            tag_ids=[t2],
        )
        session.commit()
        assert updated.color_index == 3
        assert updated.note == "a margin note"
        tag_ids = {
            link.tag_id for link in session.query(AnnotationTagLink).filter_by(annotation_id=row.id)
        }
        assert tag_ids == {t2}
        # The anchor snapshot is untouched by a metadata edit.
        assert updated.anchor_kind == WORD_RANGE and updated.quote_text == "world"


def test_reanchor_replaces_anchor_keeps_metadata(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=4,
            note="keep me",
        )
        session.commit()
        re = reanchor_annotation(
            session,
            run_id=run_id,
            annotation_id=row.id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[0], 5), "Hello"),
        )
        session.commit()
        assert re.quote_text == "Hello"
        assert (re.start_word_index, re.end_word_index) == (0, 1)
        assert re.color_index == 4 and re.note == "keep me"


def test_refresh_noop_when_text_unchanged(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        before = row.source_text_hash
        refreshed = refresh_annotation(session, run_id=run_id, annotation_id=row.id)
        session.commit()
        assert refreshed.source_text_hash == before


def test_refresh_segment_range_rederives_after_correction(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[0], 17), _SEG0),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        _correct(session, run_id, segs[0], "A corrected line")
        refreshed = refresh_annotation(session, run_id=run_id, annotation_id=row.id)
        session.commit()
        assert refreshed.quote_text == "A corrected line"


def test_refresh_text_range_refuses_when_stale(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 7), _ep(segs[0], 11), "orld"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        _correct(session, run_id, segs[0], "Totally different text here")
        with pytest.raises(AnnotationStaleError):
            refresh_annotation(session, run_id=run_id, annotation_id=row.id)


def test_refresh_word_range_refuses_when_eligibility_lost(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        _correct(session, run_id, segs[0], "Hello world there edited")
        with pytest.raises(AnnotationStaleError):
            refresh_annotation(session, run_id=run_id, annotation_id=row.id)


def test_soft_delete_is_idempotent(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        first = soft_delete_annotation(session, run_id=run_id, annotation_id=row.id)
        session.commit()
        deleted_at = first.deleted_at
        again = soft_delete_annotation(session, run_id=run_id, annotation_id=row.id)
        session.commit()
        assert again.deleted_at == deleted_at  # unchanged, not re-stamped


def test_soft_delete_unknown_is_404(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, _segs = _seed_run(session)
        with pytest.raises(AnnotationNotFoundError):
            soft_delete_annotation(session, run_id=run_id, annotation_id=uuid.uuid4())


def test_mutations_do_not_touch_transcript_evidence(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        seg_before = session.get(TranscriptSegment, segs[0])
        assert seg_before is not None
        raw_before = seg_before.raw_text
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        reanchor_annotation(
            session,
            run_id=run_id,
            annotation_id=row.id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[0], 5), "Hello"),
        )
        soft_delete_annotation(session, run_id=run_id, annotation_id=row.id)
        session.commit()
        seg = session.get(TranscriptSegment, segs[0])
        assert seg is not None
        assert seg.raw_text == raw_before and seg.enhanced_text is None
        # No review-state correction row was written by any annotation path.
        assert session.query(SegmentReviewState).filter_by(pipeline_run_id=run_id).count() == 0


# --------------------------------------------------------------------------- #
# annotations_for_run: ordering + soft-delete exclusion + tag OR-union
# --------------------------------------------------------------------------- #


def test_annotations_for_run_orders_and_excludes_deleted(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        # seg1 annotation created first, seg0 second -> listing is by segment index.
        later = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[1], 0), _ep(segs[1], 3), "how"),
            operator="op",
            nonce="b",
            color_index=0,
        )
        capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="a",
            color_index=0,
        )
        gone = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[0], 5), "Hello"),
            operator="op",
            nonce="c",
            color_index=0,
        )
        session.commit()
        soft_delete_annotation(session, run_id=run_id, annotation_id=gone.id)
        session.commit()
        rows = annotations_for_run(session, run_id)
        quotes = [r.quote_text for r in rows]
        assert quotes[0] in {"world", "Hello"}  # seg0 before seg1
        assert quotes[-1] == "how"
        assert later.quote_text == "how"
        assert gone.id not in {r.id for r in rows}


def test_annotations_for_run_tag_or_union(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        t1 = _add_tag(session, "alpha")
        t2 = _add_tag(session, "beta")
        session.commit()
        a = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="a",
            color_index=0,
            tag_ids=[t1],
        )
        b = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[0], 5), "Hello"),
            operator="op",
            nonce="b",
            color_index=0,
            tag_ids=[t2],
        )
        capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[1], 0), _ep(segs[1], 3), "how"),
            operator="op",
            nonce="c",
            color_index=0,
        )
        session.commit()
        both = {r.id for r in annotations_for_run(session, run_id, tag_ids=[t1, t2])}
        assert both == {a.id, b.id}
        just_one = {r.id for r in annotations_for_run(session, run_id, tag_ids=[t1])}
        assert just_one == {a.id}


# --------------------------------------------------------------------------- #
# Read round trip: capture -> resolve against the real render
# --------------------------------------------------------------------------- #


def test_resolve_round_trip_over_real_render(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        covered = load_covered_segments(session, run_id)
        [resolved] = resolve_annotation_spans(lines, covered, [stored_anchor_from_row(row)])
        assert not resolved.stale
        painted = [lines[s.line_index].text[s.start : s.end] for s in resolved.spans]
        assert painted == ["world"]


# --------------------------------------------------------------------------- #
# Codex-review regressions (Step 2b): refresh quote cap + fingerprint canonical
# --------------------------------------------------------------------------- #


def test_refresh_over_quote_cap_is_422_not_db_error(
    session_factory: sessionmaker[Session],
) -> None:
    # A whole 3-segment segment_range whose segments are later each corrected to
    # the 20k corrected-text cap re-derives a ~60k quote on refresh: that must be
    # a domain 422, not an uncaught quote-length CHECK violation.
    with session_factory() as session:
        run_id, segs = _seed_run_segments(session, ["aa", "bb", "cc"])
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=_cap(_ep(segs[0], 0), _ep(segs[2], 2), "aa\nbb\ncc"),
            operator="op",
            nonce="n1",
            color_index=0,
        )
        session.commit()
        assert row.anchor_kind == SEGMENT_RANGE
        for seg_id in segs:
            _correct(session, run_id, seg_id, "x" * 20_000)
        with pytest.raises(AnnotationValidationError):
            refresh_annotation(session, run_id=run_id, annotation_id=row.id)


def test_capture_replay_empty_note_equals_absent_note(
    session_factory: sessionmaker[Session],
) -> None:
    # Empty note stores as NULL, so a retry that sends "" for the same nonce is the
    # SAME persisted annotation, not an idempotency conflict.
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        payload = _cap(_ep(segs[0], 6), _ep(segs[0], 11), "world")
        a = capture_annotation(
            session, run_id=run_id, payload=payload, operator="op", nonce="dup", color_index=0
        )
        session.commit()
        b = capture_annotation(
            session,
            run_id=run_id,
            payload=payload,
            operator="op",
            nonce="dup",
            color_index=0,
            note="",
        )
        session.commit()
        assert a.id == b.id


def test_capture_replay_duplicate_tag_ids_not_conflict(
    session_factory: sessionmaker[Session],
) -> None:
    # Duplicated / reordered tag ids deduplicate to the same link set, so a replay
    # carrying [t1, t1] matches the original [t1].
    with session_factory() as session:
        run_id, segs = _seed_run(session)
        t1 = _add_tag(session, "one")
        session.commit()
        payload = _cap(_ep(segs[0], 6), _ep(segs[0], 11), "world")
        a = capture_annotation(
            session,
            run_id=run_id,
            payload=payload,
            operator="op",
            nonce="dup",
            color_index=0,
            tag_ids=[t1],
        )
        session.commit()
        b = capture_annotation(
            session,
            run_id=run_id,
            payload=payload,
            operator="op",
            nonce="dup",
            color_index=0,
            tag_ids=[t1, t1],
        )
        session.commit()
        assert a.id == b.id
