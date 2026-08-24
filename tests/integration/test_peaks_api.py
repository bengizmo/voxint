"""GET /media/{run_id}/peaks (issue #57): lazy compute, cache trust, lifecycle.

Real Postgres, real WAVs on disk, the real route. The pure reducer math lives
in ``tests/unit/test_waveform_peaks.py``; here we pin the caching contract —
fingerprint verification, ETag/304, reclamation survival, fail-closed errors —
and the invalidation hooks (prepare re-run, media delete).
"""

import io
import os
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    PipelineRun,
    RunStatus,
    TranscriptSegment,
)
from voxint.ingest.service import delete_run_derived_media
from voxint.media.peaks import (
    PEAK_BUCKETS,
    SourceFingerprint,
    compute_peaks,
    store_peaks,
)
from voxint.pipeline.stages import prepare

CREDS = ("reviewer", "s3cret")


def write_wav(path: Path, seconds: float = 0.5, amplitude: int = 8192) -> None:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(amplitude.to_bytes(2, "little", signed=True) * frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(session_factory: sessionmaker[Session], media_root: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def seed_run(session: Session, media_root: Path) -> uuid.UUID:
    """A completed run with a real normalized WAV and one transcript segment."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav", duration_seconds=0.5)
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    audio_rel = f"artifacts/{run.id}/normalized.wav"
    write_wav(media_root / audio_rel)
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=audio_rel,
        )
    )
    session.add(
        TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=0.4,
            raw_text="hello",
            diarization_label="S0",
        )
    )
    session.commit()
    return run.id


def peaks_rows(session: Session, run_id: uuid.UUID) -> list[AudioArtifact]:
    return list(
        session.execute(
            select(AudioArtifact).where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.WAVEFORM_PEAKS.value,
            )
        ).scalars()
    )


def test_unknown_run_404(client: TestClient) -> None:
    assert client.get(f"/media/{uuid.uuid4()}/peaks").status_code == 404


def test_first_request_computes_caches_and_serves(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    resp = client.get(f"/media/{run_id}/peaks")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["cache-control"] == "private, no-cache"
    body = resp.json()
    assert body["version"] == 1
    assert body["duration"] == 0.5
    assert body["sampleRate"] == 16000
    assert body["frameCount"] == 8000
    assert len(body["peaks"]) == PEAK_BUCKETS
    assert body["peaks"][0] == round(8192 / 32768.0, 3)

    # Cache file is byte-identical to the response; row carries the fingerprint.
    cache_file = media_root / "artifacts" / str(run_id) / "peaks.json"
    assert cache_file.read_bytes() == resp.content
    with session_factory() as session:
        rows = peaks_rows(session, run_id)
        assert len(rows) == 1
        fp = (rows[0].meta or {})["source_fingerprint"]
        wav_stat = (media_root / "artifacts" / str(run_id) / "normalized.wav").stat()
        assert fp == {"size": wav_stat.st_size, "mtime_ns": wav_stat.st_mtime_ns}
        # Quoted strong ETag = the canonical row's UUID.
        assert resp.headers["etag"] == f'"{rows[0].id}"'


def test_second_request_serves_cache_without_recompute(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    first = client.get(f"/media/{run_id}/peaks")
    assert first.status_code == 200

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("cache hit must not recompute")

    monkeypatch.setattr("voxint.api.routers.legacy_runs.compute_peaks", boom)
    second = client.get(f"/media/{run_id}/peaks")
    assert second.status_code == 200
    assert second.content == first.content
    assert second.headers["etag"] == first.headers["etag"]


def test_if_none_match_returns_304(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    first = client.get(f"/media/{run_id}/peaks")
    etag = first.headers["etag"]

    revalidated = client.get(f"/media/{run_id}/peaks", headers={"If-None-Match": etag})
    assert revalidated.status_code == 304
    assert revalidated.headers["etag"] == etag
    assert revalidated.content == b""

    mismatched = client.get(
        f"/media/{run_id}/peaks", headers={"If-None-Match": '"not-the-etag"'}
    )
    assert mismatched.status_code == 200


def test_stale_fingerprint_recomputes_with_new_etag(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    first = client.get(f"/media/{run_id}/peaks")
    assert first.json()["peaks"][0] == round(8192 / 32768.0, 3)

    # Replace the WAV (louder, longer) WITHOUT touching the peaks row — the
    # crash-window scenario the fingerprint exists for.
    wav = media_root / "artifacts" / str(run_id) / "normalized.wav"
    write_wav(wav, seconds=1.0, amplitude=16384)
    os.utime(wav, ns=(wav.stat().st_atime_ns, wav.stat().st_mtime_ns + 1_000_000))

    second = client.get(f"/media/{run_id}/peaks")
    assert second.status_code == 200
    assert second.json()["duration"] == 1.0
    assert second.json()["peaks"][0] == round(16384 / 32768.0, 3)
    assert second.headers["etag"] != first.headers["etag"]
    with session_factory() as session:
        assert len(peaks_rows(session, run_id)) == 1  # replaced, not duplicated


def test_reclaimed_without_cache_410(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        artifact = session.execute(select(AudioArtifact)).scalars().one()
        (media_root / artifact.path).unlink()
        artifact.reclaimed_at = datetime.now(tz=UTC)
        artifact.reclaimed_bytes = 4096
        session.commit()
    assert client.get(f"/media/{run_id}/peaks").status_code == 410


def test_reclaimed_with_cache_still_serves(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Peaks computed while the WAV was live survive its reclamation: /media is
    # 410 but the static waveform still renders (derived evidence).
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    first = client.get(f"/media/{run_id}/peaks")
    assert first.status_code == 200

    with session_factory() as session:
        artifact = session.execute(
            select(AudioArtifact).where(
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
            )
        ).scalars().one()
        (media_root / artifact.path).unlink()
        artifact.reclaimed_at = datetime.now(tz=UTC)
        artifact.reclaimed_bytes = 4096
        session.commit()

    assert client.get(f"/media/{run_id}").status_code == 410
    survived = client.get(f"/media/{run_id}/peaks")
    assert survived.status_code == 200
    assert survived.content == first.content


def test_no_media_artifact_404(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        media = MediaItem(source_path="incoming/none.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.commit()
        run_id = run.id
    assert client.get(f"/media/{run_id}/peaks").status_code == 404


def test_cached_but_wav_missing_without_reclaim_is_untrusted(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Fail-closed cache trust (review fix): a cached envelope is served without
    # fingerprint verification ONLY after formal reclamation. If the WAV simply
    # goes missing with no reclaim stamp, the cache is untrusted and the route
    # falls through to the honest 404 — never an unverified serve.
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    assert client.get(f"/media/{run_id}/peaks").status_code == 200  # populate cache

    with session_factory() as session:
        artifact = session.execute(
            select(AudioArtifact).where(
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
            )
        ).scalars().one()
        (media_root / artifact.path).unlink()  # WAV gone, NO reclaim stamp
        session.commit()

    assert (media_root / "artifacts" / str(run_id) / "peaks.json").exists()
    assert client.get(f"/media/{run_id}/peaks").status_code == 404


def test_corrupt_wav_404_and_nothing_cached(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        artifact = session.execute(select(AudioArtifact)).scalars().one()
        # Stereo violates the prepare invariant; MediaGate's ffprobe will still
        # pass it (it IS audio), so the reducer's own guard is what fails.
        frames = b"\x00\x00\x00\x00" * 1000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(frames)
        (media_root / artifact.path).write_bytes(buf.getvalue())
        session.commit()

    resp = client.get(f"/media/{run_id}/peaks")
    assert resp.status_code == 404
    assert "waveform unavailable" in resp.json()["detail"]
    with session_factory() as session:
        assert peaks_rows(session, run_id) == []
    assert not (media_root / "artifacts" / str(run_id) / "peaks.json").exists()


def test_store_peaks_twice_keeps_one_canonical_row(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        wav = media_root / "artifacts" / str(run_id) / "normalized.wav"
        with wav.open("rb") as fh:
            fingerprint = SourceFingerprint.of_descriptor(fh)
            payload = compute_peaks(fh)
        first_id = store_peaks(session, run_id, media_root, payload, fingerprint)
        second_id = store_peaks(session, run_id, media_root, payload, fingerprint)
        session.commit()
        assert first_id == second_id
        assert len(peaks_rows(session, run_id)) == 1


def test_media_delete_removes_peaks_row_and_file(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    assert client.get(f"/media/{run_id}/peaks").status_code == 200
    cache_file = media_root / "artifacts" / str(run_id) / "peaks.json"
    assert cache_file.exists()

    with session_factory() as session:
        plan = delete_run_derived_media(session, run_id, media_root=media_root)
        session.commit()
    assert cache_file in plan.paths
    with session_factory() as session:
        assert peaks_rows(session, run_id) == []


def test_prepare_rerun_deletes_stale_peaks_row(
    session_factory: sessionmaker[Session],
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A requeued run re-enters prepare, which rewrites normalized.wav; the same
    # statement that clears the stale preprocessed row must clear the peaks row.
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        wav = media_root / "artifacts" / str(run_id) / "normalized.wav"
        with wav.open("rb") as fh:
            fingerprint = SourceFingerprint.of_descriptor(fh)
            payload = compute_peaks(fh)
        store_peaks(session, run_id, media_root, payload, fingerprint)
        session.commit()
        source_rel = session.get(
            MediaItem, session.get(PipelineRun, run_id).media_item_id
        ).source_path
        write_wav(media_root / source_rel)

    def fake_normalize(source: Path, dest: Path, **kwargs: object) -> SimpleNamespace:
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_wav(dest)
        return SimpleNamespace(
            duration_seconds=0.5, sample_rate=16000, channels=1, codec="pcm_s16le"
        )

    monkeypatch.setattr(prepare, "normalize_to_wav", fake_normalize)
    ctx = SimpleNamespace(media_root=media_root, ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe")
    with session_factory() as session:
        prepare.run(ctx, session, run_id)  # type: ignore[arg-type]
        session.commit()
        assert peaks_rows(session, run_id) == []
        remaining = session.execute(
            select(AudioArtifact.kind).where(AudioArtifact.pipeline_run_id == run_id)
        ).scalars().all()
        assert remaining == [ArtifactKind.PREPROCESSED_AUDIO.value]


def test_partial_unique_index_rejects_second_row(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    from sqlalchemy.exc import IntegrityError

    with session_factory() as session:
        run_id = seed_run(session, media_root)
        session.add(
            AudioArtifact(
                pipeline_run_id=run_id,
                kind=ArtifactKind.WAVEFORM_PEAKS.value,
                path=f"artifacts/{run_id}/peaks.json",
            )
        )
        session.commit()
        session.add(
            AudioArtifact(
                pipeline_run_id=run_id,
                kind=ArtifactKind.WAVEFORM_PEAKS.value,
                path=f"artifacts/{run_id}/peaks.json",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
