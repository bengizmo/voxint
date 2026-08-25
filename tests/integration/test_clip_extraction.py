"""Attributed audio-clip extraction + serving against real Postgres (issue #88).

The pure sample-bound math + WAV frame-copy is unit-tested in
``tests/unit/test_clips.py``; here we pin the DB + filesystem + route contract:
content-addressed generation and adoption, the annotation gates (404 foreign,
409 stale, 422 coarse), the serve route (Content-Disposition attachment, byte
range, HEAD), reclamation status (410), and that the GC sweep ages a clip by its
OWN ``created_at`` rather than the run's ``updated_at``.
"""

import io
import uuid
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.annotations import (
    CaptureEndpoint,
    CapturePayload,
    capture_annotation,
)
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLIP_EXTRACT, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    PipelineRun,
    RunStatus,
    TranscriptSegment,
)
from voxint.media.reclaim import reclaim_expired_intermediates

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "clip-extraction-test-csrf-key"

# seg0 "Hello world there": content_start=[0,6,12]; "world" is offset [6, 11).
# seg1 "how are you":       a second segment, so a cross-segment span is coarse.
_SEG0 = "Hello world there"
_SEG1 = "how are you"


def write_wav(path: Path, seconds: float = 1.0, amplitude: int = 8192) -> int:
    """Write a real mono 16 kHz s16le WAV and return its frame count."""
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(amplitude.to_bytes(2, "little", signed=True) * frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return frames


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


def _ep(seg_id: uuid.UUID, offset: int) -> CaptureEndpoint:
    return CaptureEndpoint(segment_id=seg_id, offset=offset)


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(session_factory: sessionmaker[Session], media_root: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        csrf_secret=_CSRF_KEY,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _seed_run_wav(session: Session, media_root: Path, *, seconds: float = 1.0) -> uuid.UUID:
    """A completed run with a real normalized WAV, its preprocessed_audio row, and
    two word-timed segments."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav", duration_seconds=seconds)
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    audio_rel = f"artifacts/{run.id}/normalized.wav"
    write_wav(media_root / audio_rel, seconds=seconds)
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=audio_rel,
        )
    )
    for index, (raw, lo, hi) in enumerate([(_SEG0, 0.0, 0.5), (_SEG1, 0.5, 0.9)]):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=lo,
                end_seconds=hi,
                raw_text=raw,
                diarization_label="S0",
                words=_tokens(raw, lo, hi),
            )
        )
    session.commit()
    return run.id


def _seg_ids(session: Session, run_id: uuid.UUID) -> list[uuid.UUID]:
    return list(
        session.execute(
            select(TranscriptSegment.id)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        ).scalars()
    )


def _capture(
    session: Session, run_id: uuid.UUID, payload: CapturePayload
) -> uuid.UUID:
    row = capture_annotation(
        session,
        run_id=run_id,
        payload=payload,
        operator="reviewer",
        nonce=uuid.uuid4().hex,
        color_index=0,
    )
    session.commit()
    return row.id


def _word_annotation(session: Session, run_id: uuid.UUID) -> uuid.UUID:
    """A single-word ("world") highlight: precise word timing."""
    segs = _seg_ids(session, run_id)
    return _capture(session, run_id, CapturePayload(_ep(segs[0], 6), _ep(segs[0], 11), "world"))


def _coarse_annotation(session: Session, run_id: uuid.UUID) -> uuid.UUID:
    """A whole-segment highlight: segment_range, only approximate segment timing."""
    segs = _seg_ids(session, run_id)
    return _capture(
        session, run_id, CapturePayload(_ep(segs[0], 0), _ep(segs[0], len(_SEG0)), _SEG0)
    )


def _mint() -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLIP_EXTRACT)}


def _clip_rows(session: Session, run_id: uuid.UUID) -> list[AudioArtifact]:
    return list(
        session.execute(
            select(AudioArtifact).where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.AUDIO_CLIP.value,
            )
        ).scalars()
    )


# --------------------------------------------------------------------------- #
# generate + adopt
# --------------------------------------------------------------------------- #


def test_extract_generates_clip(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)

    resp = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    clip_id = body["clipId"]
    assert body["downloadUrl"] == f"/runs/{run_id}/clips/{clip_id}"
    assert body["filename"].startswith("voxint-") and body["filename"].endswith(".wav")

    with session_factory() as session:
        rows = _clip_rows(session, run_id)
        assert len(rows) == 1
        assert str(rows[0].id) == clip_id
        assert rows[0].idempotency_key is not None
        assert (media_root / rows[0].path).is_file()
        # It is a real, shorter-than-source WAV.
        with wave.open(str(media_root / rows[0].path), "rb") as w:
            assert w.getframerate() == 16000
            assert 0 < w.getnframes() < 16000


def test_extract_replay_adopts_same_clip(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)

    first = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    second = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    assert first.status_code == second.status_code == 201
    assert first.json()["clipId"] == second.json()["clipId"]
    with session_factory() as session:
        assert len(_clip_rows(session, run_id)) == 1


def test_extract_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
    resp = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data={})
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# annotation gates
# --------------------------------------------------------------------------- #


def test_extract_foreign_annotation_404(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
    resp = client.post(f"/review/{run_id}/annotations/{uuid.uuid4()}/clips", data=_mint())
    assert resp.status_code == 404


def test_extract_unknown_run_404(client: TestClient) -> None:
    resp = client.post(
        f"/review/{uuid.uuid4()}/annotations/{uuid.uuid4()}/clips", data=_mint()
    )
    assert resp.status_code == 404


def test_extract_stale_409(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    from voxint.db.models import SegmentReviewState

    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
        # Correct seg0 so its effective text drifts: the captured hash no longer
        # matches, and the read resolver marks the highlight stale.
        seg0 = _seg_ids(session, run_id)[0]
        session.add(
            SegmentReviewState(
                transcript_segment_id=seg0,
                pipeline_run_id=run_id,
                corrected_text="Goodbye planet entirely",
                corrected_at=datetime.now(UTC),
            )
        )
        session.commit()

    resp = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    assert resp.status_code == 409
    assert resp.headers.get("X-Voxint-Conflict") == "stale"


def test_extract_coarse_422(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _coarse_annotation(session, run_id)
    resp = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    assert resp.status_code == 422


def test_extract_source_reclaimed_409(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
        src = session.execute(
            select(AudioArtifact).where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
            )
        ).scalar_one()
        src.reclaimed_at = datetime.now(UTC)
        src.reclaimed_bytes = 0
        session.commit()
    resp = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    assert resp.status_code == 409
    assert resp.headers.get("X-Voxint-Conflict") != "stale"


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


def _generate(
    client: TestClient, run_id: uuid.UUID, ann_id: uuid.UUID
) -> tuple[str, str]:
    resp = client.post(f"/review/{run_id}/annotations/{ann_id}/clips", data=_mint())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["clipId"], body["downloadUrl"]


def test_serve_clip_attachment(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
    _clip_id, url = _generate(client, run_id, ann_id)

    resp = client.get(url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["accept-ranges"] == "bytes"
    disp = resp.headers["content-disposition"]
    assert disp.startswith('attachment; filename="voxint-') and disp.endswith('.wav"')
    assert int(resp.headers["content-length"]) == len(resp.content)
    assert resp.content[:4] == b"RIFF"


def test_serve_clip_range(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
    _clip_id, url = _generate(client, run_id, ann_id)

    full = client.get(url).content
    resp = client.get(url, headers={"Range": "bytes=0-9"})
    assert resp.status_code == 206
    assert resp.headers["content-range"].startswith("bytes 0-9/")
    assert resp.content == full[:10]


def test_serve_clip_head(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
    _clip_id, url = _generate(client, run_id, ann_id)

    resp = client.head(url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.content == b""


def test_serve_missing_404(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
    assert client.get(f"/runs/{run_id}/clips/{uuid.uuid4()}").status_code == 404


def test_serve_reclaimed_410(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
    clip_id, url = _generate(client, run_id, ann_id)
    with session_factory() as session:
        row = session.get(AudioArtifact, uuid.UUID(clip_id))
        assert row is not None
        row.reclaimed_at = datetime.now(UTC)
        row.reclaimed_bytes = 0
        session.commit()
    assert client.get(url).status_code == 410


# --------------------------------------------------------------------------- #
# GC
# --------------------------------------------------------------------------- #


def test_gc_ages_clip_by_created_at(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    """A clip freshly generated on an OLD terminal run survives one sweep: the
    sweep ages a clip by the clip's own created_at, not the run's updated_at."""
    with session_factory() as session:
        run_id = _seed_run_wav(session, media_root)
        ann_id = _word_annotation(session, run_id)
    clip_id, _url = _generate(client, run_id, ann_id)

    # Push the RUN far into the past; the clip row stays freshly created.
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.updated_at = datetime.now(UTC) - timedelta(days=30)
        session.commit()

    cutoff = datetime.now(UTC) - timedelta(days=7)
    with session_factory() as session:
        summary = reclaim_expired_intermediates(
            session,
            media_root=media_root,
            cutoff=cutoff,
            batch_limit=50,
            tutorial_run_id=None,
        )
        # The old normalized.wav is eligible; the fresh clip is not.
        clip = session.get(AudioArtifact, uuid.UUID(clip_id))
        assert clip is not None
        assert clip.reclaimed_at is None
    assert summary.reclaimed >= 1
