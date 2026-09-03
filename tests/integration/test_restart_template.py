"""Integration tests for restart label-risk template controls.

Verifies that the run-detail and editor restart forms render correctly
based on the RestartImpact state: clean (no checkbox), label-risk
(required checkbox), or blocked (disabled button).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_RESTART, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "restart-template-test-csrf-key"
NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY
    )
    app = create_app(settings=settings, session_factory=session_factory)
    test_client = TestClient(app)
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _seed_completed_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/restart-template.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id,
        status=RunStatus.COMPLETED.value,
        created_at=NOW,
        updated_at=NOW,
    )
    session.add(run)
    session.flush()
    return run.id


def _add_speaker(session: Session) -> uuid.UUID:
    speaker = Speaker(display_name="Test Speaker")
    session.add(speaker)
    session.flush()
    return speaker.id


def _add_label_scope_decision(session: Session, run_id: uuid.UUID) -> None:
    speaker_id = _add_speaker(session)
    session.add(AdjudicationDecision(
        pipeline_run_id=run_id,
        diarization_label="SPEAKER_00",
        decision="assign",
        speaker_id=speaker_id,
        transcript_segment_id=None,
        operator="test",
        idempotency_key=str(uuid.uuid4()),
    ))
    session.flush()


def _add_segment_scope_decision(session: Session, run_id: uuid.UUID) -> None:
    speaker_id = _add_speaker(session)
    seg = TranscriptSegment(
        pipeline_run_id=run_id,
        diarization_label="SPEAKER_00",
        segment_index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        raw_text="hello",
    )
    session.add(seg)
    session.flush()
    session.add(AdjudicationDecision(
        pipeline_run_id=run_id,
        diarization_label="SPEAKER_00",
        decision="assign",
        speaker_id=speaker_id,
        transcript_segment_id=seg.id,
        operator="test",
        idempotency_key=str(uuid.uuid4()),
    ))
    session.flush()


class TestRunDetailRestart:
    def test_clean_run_shows_restart_without_checkbox(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_completed_run(session)
            session.commit()
        resp = client.get(f"/runs/{run_id}", follow_redirects=False)
        assert resp.status_code == 200
        html = resp.text
        assert 'name="acknowledge_label_risk"' not in html
        assert "Run again from the beginning" in html

    def test_label_risk_shows_checkbox(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_completed_run(session)
            _add_label_scope_decision(session, run_id)
            session.commit()
        resp = client.get(f"/runs/{run_id}", follow_redirects=False)
        assert resp.status_code == 200
        html = resp.text
        assert 'name="acknowledge_label_risk"' in html
        assert "required" in html
        assert "speaker ruling" in html

    def test_segment_blocker_disables_button(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_completed_run(session)
            _add_segment_scope_decision(session, run_id)
            session.commit()
        resp = client.get(f"/runs/{run_id}", follow_redirects=False)
        assert resp.status_code == 200
        html = resp.text
        assert "disabled" in html
        assert "Blocked" in html
        assert "Submit the file as a new run instead" in html


class TestRestartPost:
    def test_label_risk_without_ack_returns_409(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_completed_run(session)
            run = session.get(PipelineRun, run_id)
            assert run is not None
            revision = run.revision
            _add_label_scope_decision(session, run_id)
            session.commit()
        resp = client.post(
            f"/runs/{run_id}/restart",
            data={
                "revision": str(revision),
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_RESTART),
            },
            follow_redirects=False,
        )
        assert resp.status_code == 409

    def test_label_risk_with_ack_succeeds(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_completed_run(session)
            run = session.get(PipelineRun, run_id)
            assert run is not None
            revision = run.revision
            _add_label_scope_decision(session, run_id)
            session.commit()
        resp = client.post(
            f"/runs/{run_id}/restart",
            data={
                "revision": str(revision),
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_RESTART),
                "acknowledge_label_risk": "true",
            },
            follow_redirects=False,
        )
        assert resp.status_code in (303, 200)
