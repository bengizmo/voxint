"""Turn projection loads word evidence and respects revoked segment rulings."""

import uuid

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.transcript import TranscriptText
from voxint.adjudication.turns import PieceRule, attributed_turns, join_pieces
from voxint.adjudication.undo import undo_key, undo_segment_decision
from voxint.db.models import (
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    TranscriptSegment,
)


@pytest.mark.parametrize("revoke_override", [False, True])
def test_word_turns_and_revoked_override(
    session_factory: sessionmaker[Session],
    revoke_override: bool,
) -> None:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        speakers = [Speaker(display_name=name) for name in ("Alex", "Sam")]
        session.add_all(speakers)
        session.flush()
        for i, speaker in enumerate(speakers):
            label = f"SPEAKER_{i:02}"
            session.add(
                DiarizationTurn(
                    pipeline_run_id=run.id,
                    turn_index=i,
                    start_seconds=float(i),
                    end_seconds=float(i + 1),
                    label=label,
                    skip_reason="too_short",
                )
            )
            record_decision(
                session,
                pipeline_run_id=run.id,
                diarization_label=label,
                decision=Decision.ASSIGN,
                speaker_id=speaker.id,
                operator="test",
                idempotency_key=str(uuid.uuid4()),
            )
        seg = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0,
            end_seconds=2,
            diarization_label="SPEAKER_00",
            raw_text=" Hello there",
            words=[
                {"word": " Hello", "start": 0, "end": 1},
                {"word": " there", "start": 1, "end": 2},
            ],
        )
        session.add(seg)
        session.flush()
        if revoke_override:
            ruling = record_decision(
                session,
                pipeline_run_id=run.id,
                diarization_label="SPEAKER_00",
                transcript_segment_id=seg.id,
                decision=Decision.ASSIGN,
                speaker_id=speakers[0].id,
                operator="test",
                idempotency_key=str(uuid.uuid4()),
            )
            coarse = attributed_turns(session, run.id, text=TranscriptText.RAW)
            assert len(coarse) == 1
            assert coarse[0].pieces[0].rule == PieceRule.SEGMENT_OVERRIDE
            session.commit()
            # Segment undo is a compensating INHERIT, not a label-scope REVOKE.
            undo_segment_decision(
                session,
                run.id,
                ruling.id,
                "test",
                undo_key(ruling.id),
                3600,
            )
        result = attributed_turns(session, run.id, text=TranscriptText.RAW)
        assert [t.speaker for t in result] == ["Alex", "Sam"]
        pieces = [p for t in result for p in t.pieces]
        assert all(p.timed and p.rule == PieceRule.WORD_LEVEL for p in pieces)
        assert [(p.start_seconds, p.end_seconds) for p in pieces] == [(0, 1), (1, 2)]
        assert join_pieces(pieces) == seg.raw_text
