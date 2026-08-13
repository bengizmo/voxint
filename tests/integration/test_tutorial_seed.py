"""`voxint tutorial seed` and its shared fixture builder against real Postgres.

Covers the three-state resolution, assignment-shape constraints, idempotency
(seed-twice, rebuild-after-delete, repair-missing-WAV), and the media-serve +
export paths the tutorial UI drives — all through the committed WAV/fixtures.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.resolver import Resolution, label_states
from voxint.api.app import create_app
from voxint.app_settings import get_app_settings
from voxint.cli import main
from voxint.config import Settings
from voxint.db.models import (
    AudioArtifact,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
    TranscriptSegment,
)
from voxint.db.session import session_scope
from voxint.tutorial import resources
from voxint.tutorial.seed import seed_tutorial_run

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "tutorial-seed-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def settings(media_root: Path) -> Settings:
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        csrf_secret=_CSRF_KEY,
        review_claim_ttl_seconds=600,
    )


def _seed(session_factory: sessionmaker[Session], settings: Settings) -> uuid.UUID:
    with session_scope(session_factory) as session:
        return seed_tutorial_run(
            session, media_root=settings.media_root, settings=settings
        )


def _count(session: Session, model: type) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_seed_creates_three_states(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _seed(session_factory, settings)
    layout = resources.load_layout()
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value
        assert run.current_stage is None
        states = {s.label: s for s in label_states(session, run_id)}

    grounded = states[layout["roster_speaker"]["label"]]
    assert grounded.resolution is Resolution.GROUNDED_COSINE
    assert grounded.speaker_name == layout["roster_speaker"]["display_name"]
    assert grounded.cosine_grounded is True

    heard = states[layout["heard_name"]["label"]]
    assert heard.resolution is Resolution.UNRESOLVED
    assert heard.llm_hint_name == layout["heard_name"]["name"]

    plain = states[layout["unresolved_label"]]
    assert plain.resolution is Resolution.UNRESOLVED
    assert plain.llm_hint_name is None
    assert plain.cosine_speaker_id is None


def test_seed_assignment_shapes(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _seed(session_factory, settings)
    layout = resources.load_layout()
    roster_label = layout["roster_speaker"]["label"]
    heard_label = layout["heard_name"]["label"]
    unresolved_label = layout["unresolved_label"]
    with session_factory() as session:
        rows = (
            session.execute(
                select(SpeakerAssignment).where(
                    SpeakerAssignment.pipeline_run_id == run_id
                )
            )
            .scalars()
            .all()
        )
    by_key = {(r.diarization_label, r.method): r for r in rows}

    cosine = by_key[(roster_label, "cosine")]
    assert cosine.grounded is True
    assert cosine.speaker_id is not None
    assert cosine.proposed_name is None
    assert cosine.confidence == pytest.approx((0.95 + 1.0) / 2.0)

    hint = by_key[(heard_label, "llm_hint")]
    assert hint.grounded is False
    assert hint.speaker_id is None
    assert hint.proposed_name == layout["heard_name"]["name"]
    assert hint.confidence is None

    # The unresolved label carries no proposal at all.
    assert (unresolved_label, "cosine") not in by_key
    assert (unresolved_label, "llm_hint") not in by_key


def test_seed_twice_is_idempotent(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    first = _seed(session_factory, settings)
    second = _seed(session_factory, settings)
    assert first == second
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1
        assert _count(session, MediaItem) == 1
        assert _count(session, DiarizationTurn) == len(
            resources.load_layout()["utterances"]
        )
        assert _count(session, TranscriptSegment) == len(
            resources.load_layout()["utterances"]
        )
        assert _count(session, SpeakerEmbedding) == 1
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id == first


def test_reseed_rebuilds_after_run_deleted(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    first = _seed(session_factory, settings)
    # Simulate a deleted run: remove its child rows in FK-safe order, then the run.
    # The app_settings.tutorial_run_id FK (ON DELETE SET NULL) nulls itself.
    with session_scope(session_factory) as session:
        for model in (
            SpeakerAssignment,
            TranscriptSegment,
            DiarizationTurn,
            AudioArtifact,
        ):
            session.execute(
                delete(model).where(model.pipeline_run_id == first)
            )
        from voxint.db.models import StageRun

        session.execute(delete(StageRun).where(StageRun.pipeline_run_id == first))
        session.execute(delete(PipelineRun).where(PipelineRun.id == first))

    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id is None  # SET NULL fired

    second = _seed(session_factory, settings)
    assert second != first
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1  # old gone, one fresh run
        assert _count(session, MediaItem) == 1  # sentinel MediaItem reused
        assert _count(session, Speaker) == 1  # roster speaker reused
        assert _count(session, SpeakerEmbedding) == 1  # centroid not duplicated
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id == second


def test_reseed_repairs_missing_wav(
    session_factory: sessionmaker[Session], settings: Settings, media_root: Path
) -> None:
    run_id = _seed(session_factory, settings)
    wav = media_root / "artifacts" / str(run_id) / "normalized.wav"
    assert wav.is_file()
    wav.unlink()

    again = _seed(session_factory, settings)
    assert again == run_id  # same run, no rebuild
    assert wav.is_file()
    assert wav.read_bytes() == resources.load_sample_wav_bytes()
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1


# --- media-serve + export through the real API --------------------------------


@pytest.fixture()
def client_and_run(
    session_factory: sessionmaker[Session], settings: Settings
) -> tuple[TestClient, uuid.UUID]:
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    run_id = _seed(session_factory, settings)
    return client, run_id


def test_media_serves_seeded_wav(
    client_and_run: tuple[TestClient, uuid.UUID],
) -> None:
    client, run_id = client_and_run
    size = len(resources.load_sample_wav_bytes())

    head = client.head(f"/media/{run_id}")
    assert head.status_code == 200
    assert head.headers["accept-ranges"] == "bytes"
    assert int(head.headers["content-length"]) == size

    full = client.get(f"/media/{run_id}")
    assert full.status_code == 200
    assert full.content[:4] == b"RIFF"
    assert len(full.content) == size


def test_export_attributes_grounded_but_not_heard_name(
    client_and_run: tuple[TestClient, uuid.UUID],
) -> None:
    client, run_id = client_and_run
    layout = resources.load_layout()
    export = client.get(f"/review/{run_id}/export.txt")
    assert export.status_code == 200
    body = export.text

    # The grounded label is attributed to its roster speaker.
    assert f"] {layout['roster_speaker']['display_name']}:" in body
    # Both unresolved labels render as their raw diarization label.
    assert f"] {layout['heard_name']['label']}:" in body
    assert f"] {layout['unresolved_label']}:" in body
    # The heard name is never promoted to an attribution.
    assert f"] {layout['heard_name']['name']}:" not in body
    # Every utterance's text is present.
    for utt in layout["utterances"]:
        assert utt["text"] in body


# --- CLI end to end -----------------------------------------------------------


def test_cli_tutorial_seed_creates_completed_run(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    assert main(["tutorial", "seed"]) == 0
    run_id = uuid.UUID(capsys.readouterr().out.strip())
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None and run.status == RunStatus.COMPLETED.value
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id == run_id
    assert (tmp_path / "artifacts" / str(run_id) / "normalized.wav").is_file()


def test_cli_tutorial_seed_is_idempotent(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    assert main(["tutorial", "seed"]) == 0
    first = capsys.readouterr().out.strip()
    assert main(["tutorial", "seed"]) == 0
    second = capsys.readouterr().out.strip()
    assert first == second
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1
