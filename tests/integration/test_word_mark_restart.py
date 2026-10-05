"""Effective marks count as editorial loss only when a restart deletes segments."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_translation_view_export import _build_client
from tests.integration.test_word_mark_surfaces import mark, seed_markable
from voxint.adjudication.word_marks import undo_word_mark
from voxint.cli import main
from voxint.db.models import STAGE_ORDER, SegmentWordMark, Stage, TranscriptSegment, WordMarkAction
from voxint.ingest.service import (
    RunRestartEditorialLossError,
    restart_impact,
    restart_run,
    restart_stage_profiles,
)


@pytest.fixture
def restart_client(session_factory: sessionmaker[Session]) -> TestClient:
    return _build_client(session_factory)


def seed_history(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    run_id, segment_id = seed_markable(session)
    mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
    mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)
    mark(session, run_id, segment_id, 2, WordMarkAction.OMIT)
    mark(session, run_id, segment_id, 2, WordMarkAction.CLEAR)
    undone = mark(session, run_id, segment_id, 3, WordMarkAction.OMIT)
    undo_word_mark(
        session, run_id=run_id, mark_id=undone.id, operator="test", user_id=None, grace_seconds=300
    )
    session.commit()
    return run_id, segment_id


def test_effective_counts_and_stage_profiles(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, _ = seed_history(session)
        full = restart_impact(session, run_id)
        assert full.word_marks == 2
        assert full.has_editorial_work and not full.requires_void
        for stage in STAGE_ORDER:
            impact = restart_impact(session, run_id, from_stage=stage)
            expected = 2 if stage in (Stage.ACQUIRE, Stage.PREPARE, Stage.TRANSCRIBE) else 0
            assert impact.word_marks == expected
            assert impact.has_editorial_work is bool(expected)
        profiles = restart_stage_profiles(full)
        assert [p["word_marks"] for p in profiles] == [2, 2, 2, 0, 0, 0]
        assert [p["safe"] for p in profiles] == [False, False, False, True, True, True]


def test_restart_refusal_attributes_and_acknowledged_cascade(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_history(session)
        with pytest.raises(RunRestartEditorialLossError) as caught:
            restart_run(session, run_id)
        error = caught.value
        assert error.word_marks == 2 and error.corrections == error.verifications == 0
        assert "2 clean-up mark(s)" in str(error)
        assert session.get(TranscriptSegment, segment_id) is not None
        restart_run(session, run_id, acknowledge_editorial=True)
        session.commit()
        assert session.get(TranscriptSegment, segment_id) is None
        assert session.scalar(select(func.count()).select_from(SegmentWordMark)) == 0


def test_restart_template_reports_marks_on_deleting_stages(
    session_factory: sessionmaker[Session],
    restart_client: TestClient,
) -> None:
    with session_factory() as session:
        run_id, _ = seed_history(session)
    response = restart_client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    assert 'data-word-marks="2"' in response.text
    assert 'data-word-marks="0"' in response.text
    assert "clean-up mark(s)" in response.text
    assert "opt.dataset.wordMarks" in response.text
    assert "verifications > 0 || wordMarks > 0" in response.text
    from html.parser import HTMLParser

    class Options(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.options: dict[str, dict[str, str | None]] = {}

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            values = dict(attrs)
            if tag == "option" and "data-word-marks" in values:
                self.options[values.get("value") or ""] = values

    parsed = Options()
    parsed.feed(response.text)
    assert parsed.options[""]["data-word-marks"] == "2"
    for stage in STAGE_ORDER:
        expected = "2" if stage in (Stage.ACQUIRE, Stage.PREPARE, Stage.TRANSCRIBE) else "0"
        assert parsed.options[stage.value]["data-word-marks"] == expected


def test_cli_restart_warning_and_bulk_consumers(
    session_factory: sessionmaker[Session],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with session_factory() as session:
        run_id, _ = seed_history(session)
    monkeypatch.setattr("voxint.db.session.build_engine", lambda: engine)
    assert main(["restart", str(run_id), "--yes"]) == 2
    out = capsys.readouterr().out
    assert "2 clean-up mark(s)" in out and "--acknowledge-editorial" in out
    with session_factory() as session:
        assert restart_impact(session, run_id).word_marks == 2
    assert main(["restart", "--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 clean-up marks" in out and "1 editorial-loss" in out
    assert main(["restart", "--all", "--yes"]) == 0
    captured = capsys.readouterr()
    assert "editorial-loss, pass --acknowledge-editorial" in captured.err
    assert "1 editorial-loss" in captured.out


@pytest.mark.parametrize("stage", [Stage.DIARIZE_EMBED, Stage.ENHANCE_MATCH, Stage.FINALIZE])
def test_segment_preserving_restart_keeps_marks_without_acknowledgement(
    session_factory: sessionmaker[Session],
    stage: Stage,
) -> None:
    from voxint.adjudication.word_marks import effective_marks
    from voxint.db.models import StageRun, StageStatus

    with session_factory() as session:
        run_id, segment_id = seed_history(session)
        before = effective_marks(session, run_id)
        previous = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
        session.add(
            StageRun(
                pipeline_run_id=run_id, stage=previous.value, status=StageStatus.COMPLETED.value
            )
        )
        session.commit()
        restart_run(session, run_id, from_stage=stage)
        session.commit()
        assert session.get(TranscriptSegment, segment_id) is not None
        assert effective_marks(session, run_id) == before
