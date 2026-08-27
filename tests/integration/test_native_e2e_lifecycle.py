"""DB-backed + HTTP-contract tests for the native E2E lifecycle tool.

Two halves, both against real Postgres (disposable ``voxint_e2e`` via the shared
``session_factory`` fixture) — **no model tier needed**:

* ``check_run_invariants`` over a synthetic COMPLETED run that satisfies all five
  invariants, plus one negative case per invariant proving each is actually gated.
* The HTTP steps (``_onboard`` / ``_submit`` / ``_poll``) against a real
  ``create_app`` TestClient with a seeded completed run — pinning onboard
  idempotency, the submit deferred-guard, and poll status parsing without a
  running Metal pipeline.
"""

from __future__ import annotations

import io
import uuid
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from tools.native_e2e_lifecycle import (
    LaneError,
    NativeConfig,
    _onboard,
    _poll,
    _submit,
    check_run_invariants,
)

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    Stage,
    StageRun,
    StageStatus,
    TranscriptSegment,
)

_CSRF_KEY = "native-e2e-test-csrf-key"
_CREDS = ("admin", "native-test-pw")


def _config() -> NativeConfig:
    """A NativeConfig whose creds/secret match the test app (bypasses state.env)."""
    return NativeConfig(
        pg_port="0",
        redis_port="0",
        api_port="0",
        db_password="unused",
        voxint_password=_CREDS[1],
        csrf_secret=_CSRF_KEY,
        api_user=_CREDS[0],
    )


def _wav_bytes(seconds: float = 0.02) -> bytes:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def _seed_completed_native_run(session: Session) -> uuid.UUID:
    """A COMPLETED run satisfying all five native-run invariants.

    Includes a deliberately-FAILED earlier ``transcribe`` attempt (to prove the
    per-stage group-by tolerates a prior failure) and one honest skip-reason turn
    (embedding NULL) alongside embedded ones — exactly the shape a real run leaves.
    """
    media = MediaItem(source_path=f"e2e/{uuid.uuid4().hex}.wav", duration_seconds=15.0)
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()

    for stage in Stage:
        session.add(
            StageRun(pipeline_run_id=run.id, stage=stage.value, status=StageStatus.COMPLETED.value)
        )
    # A prior failed transcribe attempt: invariant 2 must still pass (a later
    # attempt completed), so it must group by stage, not count rows.
    session.add(
        StageRun(
            pipeline_run_id=run.id,
            stage=Stage.TRANSCRIBE.value,
            status=StageStatus.FAILED.value,
            attempt=2,
        )
    )

    session.add(
        TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=5.0,
            raw_text="Good morning everyone.",
            diarization_label="SPEAKER_00",
        )
    )
    # Two embedded turns (real TitaNet) + one honest skip (embedding XOR skip_reason).
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=0,
            start_seconds=0.0,
            end_seconds=5.0,
            label="SPEAKER_00",
            embedding=[0.01] * EMBEDDING_DIM,
            embedding_space="titanet-large-v1",
        )
    )
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=1,
            start_seconds=5.0,
            end_seconds=10.0,
            label="SPEAKER_01",
            embedding=[0.02] * EMBEDDING_DIM,
            embedding_space="titanet-large-v1",
        )
    )
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=2,
            start_seconds=10.0,
            end_seconds=11.0,
            label="SPEAKER_00",
            skip_reason="too_short",  # embedding NULL, honestly skipped
        )
    )
    session.commit()
    return run.id


# --------------------------------------------------------------------------- #
# check_run_invariants — happy path + one negative per invariant
# --------------------------------------------------------------------------- #
def test_invariants_pass_for_a_well_formed_run(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
    with session_factory() as session:
        assert check_run_invariants(session, run_id) == []


def test_missing_run_is_reported(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        problems = check_run_invariants(session, uuid.uuid4())
    assert len(problems) == 1 and "not found" in problems[0]


def test_invariant_run_status(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        session.get(PipelineRun, run_id).status = RunStatus.RUNNING.value  # type: ignore[union-attr]
        session.commit()
    with session_factory() as session:
        problems = check_run_invariants(session, run_id)
    assert any("status='running'" in p for p in problems)


def test_invariant_stage_incomplete(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        # Drop the only completed finalize attempt → invariant 2 must trip.
        session.query(StageRun).filter(
            StageRun.pipeline_run_id == run_id,
            StageRun.stage == Stage.FINALIZE.value,
        ).delete()
        session.commit()
    with session_factory() as session:
        problems = check_run_invariants(session, run_id)
    assert any("'finalize' has no completed attempt" in p for p in problems)


def test_invariant_blank_transcript(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        for seg in session.query(TranscriptSegment).filter(
            TranscriptSegment.pipeline_run_id == run_id
        ):
            seg.raw_text = "   "  # whitespace-only → not counted
        session.commit()
    with session_factory() as session:
        problems = check_run_invariants(session, run_id)
    assert any("non-empty raw_text" in p for p in problems)


def test_invariant_no_embeddings(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        # Null every embedding (must give each a skip_reason to satisfy the XOR CHECK).
        for turn in session.query(DiarizationTurn).filter(
            DiarizationTurn.pipeline_run_id == run_id
        ):
            turn.embedding = None
            turn.embedding_space = None
            turn.skip_reason = "too_short"
        session.commit()
    with session_factory() as session:
        problems = check_run_invariants(session, run_id)
    assert any("no embedded turn" in p for p in problems)


def test_invariant_wrong_embedding_space(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        turn = (
            session.query(DiarizationTurn)
            .filter(DiarizationTurn.pipeline_run_id == run_id, DiarizationTurn.turn_index == 0)
            .one()
        )
        turn.embedding_space = "some-other-space-v9"
        session.commit()
    with session_factory() as session:
        problems = check_run_invariants(session, run_id)
    assert any("expected 'titanet-large-v1'" in p for p in problems)


def test_invariant_enrollment_rows_present(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        session.add(Speaker(display_name="Enrolled Human"))  # operator-only artifact
        session.commit()
    with session_factory() as session:
        problems = check_run_invariants(session, run_id)
    assert any("speakers: 1 rows" in p for p in problems)


# --------------------------------------------------------------------------- #
# HTTP steps against a real create_app TestClient (no model tier)
# --------------------------------------------------------------------------- #
def _client(session_factory: sessionmaker[Session], media_root: Path) -> TestClient:
    settings = Settings(
        voxint_user=_CREDS[0],
        voxint_password=_CREDS[1],
        media_root=media_root,
        csrf_secret=_CSRF_KEY,
    )
    # follow_redirects=False so the 303 from onboard/submit reaches the helper,
    # exactly as a plain httpx.Client (which never follows) would.
    client = TestClient(
        create_app(settings=settings, session_factory=session_factory), follow_redirects=False
    )
    seed_onboarded(session_factory)
    return client


def test_onboard_is_idempotent(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path)
    # Two calls both return 303 with no raise (complete_onboarding is get-or-create).
    _onboard(client, _config())  # type: ignore[arg-type]
    _onboard(client, _config())  # type: ignore[arg-type]


def test_submit_returns_run_id(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxint.ingest.service import SubmissionResult

    monkeypatch.setattr(SubmissionResult, "publish", lambda self: True)
    client = _client(session_factory, tmp_path)
    run_id = _submit(client, _config(), _wav_bytes(), "clip.wav")  # type: ignore[arg-type]
    parsed = uuid.UUID(run_id)
    with session_factory() as session:
        assert session.get(PipelineRun, parsed) is not None


def test_submit_fails_when_enqueue_deferred(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxint.ingest.service import SubmissionResult

    monkeypatch.setattr(SubmissionResult, "publish", lambda self: False)
    client = _client(session_factory, tmp_path)
    with pytest.raises(LaneError, match="deferred"):
        _submit(client, _config(), _wav_bytes(), "clip.wav")  # type: ignore[arg-type]


def test_poll_returns_on_completed(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path)
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
    # Completed on the first read → returns without raising (no sleep).
    _poll(client, _config(), str(run_id), interval=0.01, timeout=5.0)  # type: ignore[arg-type]


def test_poll_fails_on_terminal_failure(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path)
    with session_factory() as session:
        run_id = _seed_completed_native_run(session)
        session.get(PipelineRun, run_id).status = RunStatus.FAILED.value  # type: ignore[union-attr]
        session.commit()
    with pytest.raises(LaneError, match="terminal 'failed'"):
        _poll(client, _config(), str(run_id), interval=0.01, timeout=5.0)  # type: ignore[arg-type]
