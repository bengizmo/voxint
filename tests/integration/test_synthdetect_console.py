"""Integration tests for the synthdetect console layer (PR 3, #145).

Tests the settings section, run-detail panel, report page, and manual
score button. Uses the real synthdetect plugin (no monkeypatching) with
a test database.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_PLUGIN, CSRF_SETTINGS, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    SynthdetectJob,
    SynthdetectJobStatus,
    SynthdetectScore,
)
from voxint.plugins import reset_plugins_cache

CREDS = ("reviewer", "s3cret")
CSRF_SECRET = "synthdetect-console-test-csrf-key"


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset_plugins_cache()
    yield
    reset_plugins_cache()


def _client(
    session_factory: sessionmaker[Session], **overrides: object
) -> TestClient:
    defaults: dict[str, object] = {
        "synthdetect_enabled": False,
        "synthdetect_autogenerate": False,
        "synthdetect_url": "http://localhost:19999",
    }
    defaults.update(overrides)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=CSRF_SECRET,
        **defaults,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _seed_completed_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/source")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        run_id: uuid.UUID = run.id
        session.commit()
        return run_id


def _seed_synthdetect_job(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    *,
    status: str = SynthdetectJobStatus.SUCCEEDED.value,
    scored_turns: int = 5,
    total_turns: int = 6,
    skipped_turns: int = 1,
    mean_risk: float | None = 0.08,
    max_risk: float | None = 0.42,
    error: str | None = None,
) -> uuid.UUID:
    with session_factory() as session:
        job = SynthdetectJob(
            pipeline_run_id=run_id,
            inference_space="w2v2-aasist-df-m2-s0e11",
            calibration_policy_id="platt-m2-s0e11-dev-v1",
            status=status,
            total_turns=total_turns,
            scored_turns=scored_turns,
            skipped_turns=skipped_turns,
            mean_risk=mean_risk,
            max_risk=max_risk,
            error=error,
        )
        session.add(job)
        session.flush()
        job_id: uuid.UUID = job.id
        session.commit()
        return job_id


def _seed_scores(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    job_id: uuid.UUID,
    count: int = 3,
) -> None:
    with session_factory() as session:
        for i in range(count):
            turn = DiarizationTurn(
                pipeline_run_id=run_id,
                turn_index=i,
                label=f"SPEAKER_{i:02d}",
                start_seconds=float(i * 10),
                end_seconds=float(i * 10 + 9),
                skip_reason="test",
            )
            session.add(turn)
            session.flush()
            score = SynthdetectScore(
                synthdetect_job_id=job_id,
                pipeline_run_id=run_id,
                diarization_turn_id=turn.id,
                speaker_label=turn.label,
                raw_logit=float(i) - 1.0,
                calibrated_score=0.05 + i * 0.2,
                window_count=4,
                inference_space="w2v2-aasist-df-m2-s0e11",
                calibration_policy_id="platt-m2-s0e11-dev-v1",
            )
            session.add(score)
        session.commit()


# --------------------------------------------------------------------------- #
# Settings section.
# --------------------------------------------------------------------------- #
class TestSettingsSection:
    def test_settings_page_renders_synthdetect_section(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory)
        resp = client.get("/settings/plugins")
        assert resp.status_code == 200
        assert 'id="synthdetect"' in resp.text
        assert "Synthetic speech detection" in resp.text

    def test_settings_section_shows_tri_state_radios(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory)
        resp = client.get("/settings/plugins")
        assert resp.status_code == 200
        assert 'name="synthdetect_enabled"' in resp.text
        assert 'name="synthdetect_autogenerate"' in resp.text

    def test_settings_post_saves_enabled(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory)
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_SETTINGS)
        resp = client.post(
            "/synthdetect/settings",
            data={
                "synthdetect_enabled": "on",
                "synthdetect_autogenerate": "inherit",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/settings#synthdetect"

    def test_settings_post_rejects_autogenerate_without_enabled(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory, synthdetect_enabled=False)
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_SETTINGS)
        resp = client.post(
            "/synthdetect/settings",
            data={
                "synthdetect_enabled": "off",
                "synthdetect_autogenerate": "on",
                "csrf_token": csrf,
            },
        )
        assert resp.status_code == 200
        assert "Turn synthetic speech detection on" in resp.text

    def test_settings_post_rejects_bad_csrf(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory)
        resp = client.post(
            "/synthdetect/settings",
            data={
                "synthdetect_enabled": "on",
                "synthdetect_autogenerate": "inherit",
                "csrf_token": "bad-token",
            },
        )
        assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# Run-detail panel.
# --------------------------------------------------------------------------- #
class TestRunDetailPanel:
    def test_panel_shows_disabled_when_feature_off(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        client = _client(session_factory, synthdetect_enabled=False)
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert "Synthetic speech detection is off" in resp.text

    def test_panel_shows_not_scored_with_score_button(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert "Not scored" in resp.text
        assert "Score for synthetic speech" in resp.text

    def test_panel_shows_queued_state(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        _seed_synthdetect_job(
            session_factory, run_id, status=SynthdetectJobStatus.QUEUED.value,
            scored_turns=None, total_turns=None, skipped_turns=None,
            mean_risk=None, max_risk=None,
        )
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert "Queued" in resp.text

    def test_panel_shows_succeeded_with_risk_summary(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        _seed_synthdetect_job(session_factory, run_id)
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert "Scored" in resp.text
        assert "5 of 6 turns" in resp.text
        assert "Mean risk" in resp.text
        assert "Full report" in resp.text

    def test_panel_shows_failed_with_retry_button(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        _seed_synthdetect_job(
            session_factory, run_id,
            status=SynthdetectJobStatus.FAILED.value,
            scored_turns=None, total_turns=None, skipped_turns=None,
            mean_risk=None, max_risk=None,
            error="service error: connection refused",
        )
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert "Failed" in resp.text
        assert "Retry scoring" in resp.text


# --------------------------------------------------------------------------- #
# Report page.
# --------------------------------------------------------------------------- #
class TestReportPage:
    def test_report_page_renders_for_succeeded_job(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        job_id = _seed_synthdetect_job(session_factory, run_id)
        _seed_scores(session_factory, run_id, job_id)
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/synthdetect/report/{run_id}")
        assert resp.status_code == 200
        assert "Synthetic speech detection report" in resp.text
        assert "5 of 6" in resp.text
        assert "platt-m2-s0e11-dev-v1" in resp.text
        assert "Known limitations" in resp.text

    def test_report_page_shows_all_scores_table(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        job_id = _seed_synthdetect_job(session_factory, run_id)
        _seed_scores(session_factory, run_id, job_id, count=3)
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/synthdetect/report/{run_id}")
        assert resp.status_code == 200
        assert "SPEAKER_00" in resp.text
        assert "SPEAKER_01" in resp.text
        assert "SPEAKER_02" in resp.text

    def test_report_page_404s_for_unknown_run(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/synthdetect/report/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_report_page_handles_no_job(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.get(f"/synthdetect/report/{run_id}")
        assert resp.status_code == 200
        assert "No synthdetect job found" in resp.text


# --------------------------------------------------------------------------- #
# Manual score button.
# --------------------------------------------------------------------------- #
class TestScoreButton:
    def test_score_post_rejects_bad_csrf(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        client = _client(session_factory, synthdetect_enabled=True)
        resp = client.post(
            f"/synthdetect/score/{run_id}",
            data={"csrf_token": "bad"},
        )
        assert resp.status_code == 403

    def test_score_post_rejects_when_feature_disabled(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        client = _client(session_factory, synthdetect_enabled=False)
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_PLUGIN)
        resp = client.post(
            f"/synthdetect/score/{run_id}",
            data={"csrf_token": csrf},
        )
        assert resp.status_code == 409

    def test_score_post_404s_for_unknown_run(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _client(session_factory, synthdetect_enabled=True)
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_PLUGIN)
        resp = client.post(
            f"/synthdetect/score/{uuid.uuid4()}",
            data={"csrf_token": csrf},
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Admin gate on mutation routes (issue #305).
# --------------------------------------------------------------------------- #
class TestAdminGate:
    """Non-admin users must get 403 on synthdetect mutation routes."""

    def _multiuser_client(
        self,
        session_factory: sessionmaker[Session],
        role: str = "reviewer",
    ) -> TestClient:
        from voxint.api.auth import SESSION_COOKIE, create_session, new_session_token
        from voxint.users import UserRole, create_user

        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            voxint_multi_user=True,
            database_url="postgresql+psycopg://x/x",
            csrf_secret=CSRF_SECRET,
            synthdetect_enabled=True,
            synthdetect_autogenerate=False,
            synthdetect_url="http://localhost:19999",
        )
        app = create_app(settings=settings, session_factory=session_factory)
        seed_onboarded(session_factory)

        with session_factory() as session:
            admin = create_user(
                session, username="admin", password="adminpass", role=UserRole.ADMIN
            )
            session.commit()
            if role == "reviewer":
                user = create_user(
                    session,
                    username="viewer",
                    password="viewerpass",
                    role=UserRole.REVIEWER,
                )
                session.commit()
                user_id = user.id
            else:
                user_id = admin.id

            token = new_session_token()
            create_session(session, user_id=user_id, token=token, ttl_seconds=3600)
            session.commit()

        client = TestClient(app, cookies={SESSION_COOKIE: token})
        return client

    def test_reviewer_cannot_post_settings(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = self._multiuser_client(session_factory, role="reviewer")
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_SETTINGS)
        resp = client.post(
            "/synthdetect/settings",
            data={
                "synthdetect_enabled": "on",
                "synthdetect_autogenerate": "inherit",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        assert resp.status_code == 403

    def test_reviewer_cannot_post_score(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_completed_run(session_factory)
        client = self._multiuser_client(session_factory, role="reviewer")
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_PLUGIN)
        resp = client.post(
            f"/synthdetect/score/{run_id}",
            data={"csrf_token": csrf},
            follow_redirects=False,
        )
        assert resp.status_code == 403

    def test_admin_can_post_settings(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = self._multiuser_client(session_factory, role="admin")
        csrf = mint_csrf_token(CSRF_SECRET, CSRF_SETTINGS)
        resp = client.post(
            "/synthdetect/settings",
            data={
                "synthdetect_enabled": "on",
                "synthdetect_autogenerate": "inherit",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
