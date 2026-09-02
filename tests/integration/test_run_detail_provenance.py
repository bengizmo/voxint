"""Run-detail "Pipeline models" render + the attempt-safety guarantee (B1).

The selection logic is unit-tested in ``tests/unit/test_model_provenance.py``;
this exercises it through the real ``GET /runs/{id}`` route and template against
seeded database rows, and pins the load-bearing case: when a stage is retried, the
page shows the model from the *latest completed* attempt, never a failed or
lease-expired attempt's stamp.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)
from voxint.pipeline.model_identity import METRICS_KEY

CREDS = ("reviewer", "s3cret")
BASE = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(voxint_user=CREDS[0], voxint_password=CREDS[1])
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _identity(**roles: object) -> dict[str, object]:
    return {"v": 1, "observed_before_attempt": True, **roles}


def _stage_run(
    run_id: uuid.UUID,
    stage: Stage,
    *,
    attempt: int,
    status: StageStatus,
    metrics: dict[str, object] | None,
) -> StageRun:
    return StageRun(
        pipeline_run_id=run_id,
        stage=stage.value,
        status=status.value,
        attempt=attempt,
        started_at=BASE + timedelta(minutes=attempt),
        finished_at=BASE + timedelta(minutes=attempt, seconds=30),
        metrics=metrics,
    )


def _seed_run(
    session_factory: sessionmaker[Session], *, stage_runs: list[StageRun] | None
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/source")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            created_at=BASE,
            updated_at=BASE,
        )
        session.add(run)
        session.flush()
        for sr in stage_runs or []:
            sr.pipeline_run_id = run.id
            session.add(sr)
        session.commit()
        return run.id


def test_render_shows_latest_completed_attempt_not_a_failed_retry(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # TRANSCRIBE was retried: attempt 1 completed on large-v2; attempt 2 FAILED
    # after the probe found the service unreachable. The page must show large-v2.
    run_id = _seed_run(
        session_factory,
        stage_runs=[
            _stage_run(
                uuid.uuid4(),
                Stage.TRANSCRIBE,
                attempt=1,
                status=StageStatus.COMPLETED,
                metrics={
                    METRICS_KEY: _identity(
                        asr={
                            "reachable": True,
                            "model": "large-v2",
                            "engine": "ct2-legacy",
                            "revision": "a" * 40,
                        }
                    )
                },
            ),
            _stage_run(
                uuid.uuid4(),
                Stage.TRANSCRIBE,
                attempt=2,
                status=StageStatus.FAILED,
                metrics={
                    METRICS_KEY: _identity(
                        asr={"reachable": False, "detail": "timeout"}
                    )
                },
            ),
        ],
    )
    body = client.get(f"/runs/{run_id}").text
    assert "Pipeline models" in body
    assert "large-v2" in body
    # The failed retry's marker must not be attributed to the shown identity.
    assert "timeout" not in body
    # The completed attempt was attempt 1, so no "from attempt N>1" note appears.
    assert "from attempt" not in body


def test_render_prefers_a_later_completed_retry(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _seed_run(
        session_factory,
        stage_runs=[
            _stage_run(
                uuid.uuid4(),
                Stage.TRANSCRIBE,
                attempt=1,
                status=StageStatus.COMPLETED,
                metrics={METRICS_KEY: _identity(asr={"reachable": True, "model": "large-v2"})},
            ),
            _stage_run(
                uuid.uuid4(),
                Stage.TRANSCRIBE,
                attempt=2,
                status=StageStatus.COMPLETED,
                metrics={METRICS_KEY: _identity(asr={"reachable": True, "model": "large-v3"})},
            ),
        ],
    )
    body = client.get(f"/runs/{run_id}").text
    assert "large-v3" in body
    assert "from attempt 2" in body


def test_render_never_shows_an_older_stamp_over_a_newer_unstamped_completion(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Issue #126: attempt 2 completed but stamped nothing; attempt 1 completed
    # earlier with a stamp. The page must say "Not recorded" for the stage, not
    # attribute attempt 2's result to attempt 1's identity.
    run_id = _seed_run(
        session_factory,
        stage_runs=[
            _stage_run(
                uuid.uuid4(),
                Stage.TRANSCRIBE,
                attempt=1,
                status=StageStatus.COMPLETED,
                metrics={METRICS_KEY: _identity(asr={"reachable": True, "model": "large-v2"})},
            ),
            _stage_run(
                uuid.uuid4(),
                Stage.TRANSCRIBE,
                attempt=2,
                status=StageStatus.COMPLETED,
                metrics=None,
            ),
        ],
    )
    body = client.get(f"/runs/{run_id}").text
    assert "Not recorded" in body
    assert "large-v2" not in body
    assert "from attempt" not in body


def test_render_marks_unrecorded_stage_not_recorded(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A completed diarize stage with no identity metrics (a legacy run) reads
    # "Not recorded", and the transcribe stage with no rows at all does too.
    run_id = _seed_run(
        session_factory,
        stage_runs=[
            _stage_run(
                uuid.uuid4(),
                Stage.DIARIZE_EMBED,
                attempt=1,
                status=StageStatus.COMPLETED,
                metrics=None,
            ),
        ],
    )
    body = client.get(f"/runs/{run_id}").text
    assert "Pipeline models" in body
    assert "Not recorded" in body


def test_render_surfaces_diarizer_and_embedder_identity(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _seed_run(
        session_factory,
        stage_runs=[
            _stage_run(
                uuid.uuid4(),
                Stage.DIARIZE_EMBED,
                attempt=1,
                status=StageStatus.COMPLETED,
                metrics={
                    METRICS_KEY: _identity(
                        diarizer={
                            "reachable": True,
                            "model": "pyannote/speaker-diarization-3.1",
                        },
                        embedder={"reachable": True, "model": "titanet-large-v2"},
                    )
                },
            ),
        ],
    )
    body = client.get(f"/runs/{run_id}").text
    assert "pyannote/speaker-diarization-3.1" in body
    assert "titanet-large-v2" in body


def _seed_run_with_prompt(
    session_factory: sessionmaker[Session], *, initial_prompt: str | None
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/source")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            created_at=BASE,
            updated_at=BASE,
            initial_prompt=initial_prompt,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        session.commit()
        return run_id


def test_render_shows_applied_glossary_prompt(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The "Glossary applied" card shows the exact recorded initial_prompt (issue #123).
    run_id = _seed_run_with_prompt(session_factory, initial_prompt="Zoning Board, NUCA")
    body = client.get(f"/runs/{run_id}").text
    assert "Glossary applied" in body
    assert "Zoning Board, NUCA" in body


def test_render_glossary_empty_state_when_prompt_null(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A run with no recorded prompt shows the honest empty state, not a blank card.
    run_id = _seed_run_with_prompt(session_factory, initial_prompt=None)
    body = client.get(f"/runs/{run_id}").text
    assert "Glossary applied" in body
    # A phrase unique to the glossary empty state (the models card also says
    # "Not recorded"), so this pins the glossary branch specifically.
    assert "began recording the applied glossary" in body


def _seed_run_with_language(
    session_factory: sessionmaker[Session],
    *,
    language: str | None,
    probability: float | None,
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/source")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            created_at=BASE,
            updated_at=BASE,
            detected_language=language,
            detected_language_probability=probability,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        session.commit()
        return run_id


def test_render_shows_detected_language_with_score(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The "Detected language" card (issue #124) shows the labeled language and
    # the detection score with its honest framing.
    run_id = _seed_run_with_language(
        session_factory, language="es", probability=0.9234
    )
    body = client.get(f"/runs/{run_id}").text
    assert "Detected language" in body
    assert "Spanish (es)" in body
    assert "Language detection confidence:" in body
    assert "likely" in body
    assert "not a measure of\ntranscript accuracy" in body


def test_render_detected_language_without_score(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Forced/fallback branches record a language with no score; the card says so
    # rather than fabricating one.
    run_id = _seed_run_with_language(session_factory, language="en", probability=None)
    body = client.get(f"/runs/{run_id}").text
    assert "English (en)" in body
    assert "No detection score was recorded" in body


def test_render_detected_language_empty_state(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _seed_run_with_language(session_factory, language=None, probability=None)
    body = client.get(f"/runs/{run_id}").text
    assert "Detected language" in body
    # A phrase unique to this card's empty state.
    assert "began recording the detected language" in body
