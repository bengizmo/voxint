"""Cross-recording samples use real attribution, lifecycle and distinguishable PCM."""

import io
import struct
import uuid
import wave
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.ledger import record_decision
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    TranscriptSegment,
)
from voxint.media.executor import execute_operation, plan_trash
from voxint.media.purge import execute_purge, plan_purge
from voxint.media.reclaim import reclaim_expired_intermediates
from voxint.speakers.roster import archive_speaker, merge_speakers

CREDS = ("reviewer", "s3cret")


@dataclass
class Recording:
    media_id: uuid.UUID
    run_id: uuid.UUID
    path: Path
    pcm: bytes


def seed(
    session: Session,
    root: Path,
    speaker: uuid.UUID,
    *,
    start: float = 1.0,
    end: float = 6.0,
    age: int = 1,
    human: bool = True,
    method: str = "cosine",
    overlap: bool = False,
) -> Recording:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav", duration_seconds=25)
    session.add(media)
    session.flush()
    media.created_at = datetime.now(UTC) - timedelta(days=age)
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    run.updated_at = media.created_at
    rel = f"artifacts/{run.id}/normalized.wav"
    path = root / rel
    path.parent.mkdir(parents=True)
    pcm = b"".join(struct.pack("<h", (i + age * 1000) % 32768) for i in range(400000))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    source = root / media.source_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(path.read_bytes())
    session.add(
        AudioArtifact(pipeline_run_id=run.id, kind=ArtifactKind.PREPROCESSED_AUDIO.value, path=rel)
    )
    session.add(
        TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=start,
            end_seconds=end,
            raw_text="Confirmed voice",
            diarization_label="S0",
        )
    )
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=0,
            start_seconds=start,
            end_seconds=end,
            label="S0",
            skip_reason="too_short",
            overlap=overlap,
        )
    )
    if overlap:
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=1,
                start_seconds=start,
                end_seconds=start + 1.5,
                label="S1",
                skip_reason="too_short",
                overlap=True,
            )
        )
    session.flush()
    if human or method == "auto_enroll":
        record_decision(
            session,
            pipeline_run_id=run.id,
            diarization_label="S0",
            decision=Decision.ASSIGN if human else Decision.AUTO_ENROLL,
            operator=CREDS[0],
            idempotency_key=str(uuid.uuid4()),
            speaker_id=speaker,
        )
    else:
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run.id,
                diarization_label="S0",
                speaker_id=speaker,
                method=method,
                grounded=True,
                confidence=0.95,
            )
        )
    session.commit()
    return Recording(media.id, run.id, path, pcm)


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    seed_onboarded(session_factory)
    result = TestClient(
        create_app(
            settings=Settings(
                voxint_user=CREDS[0],
                voxint_password=CREDS[1],
                media_root=tmp_path,
                csrf_secret="voice-sample-test-key",
            ),
            session_factory=session_factory,
        )
    )
    result.auth = CREDS
    return result


def identities(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    current = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    speaker = Speaker(display_name="Dana")
    session.add_all([current, speaker])
    session.commit()
    return current.id, speaker.id


def url(current: uuid.UUID, speaker: uuid.UUID) -> str:
    return f"/media/{current}/editor/voice-sample/{speaker}"


def audio(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return wav.readframes(wav.getnframes())


def reclaim(session: Session, root: Path) -> None:
    summary = reclaim_expired_intermediates(
        session,
        media_root=root,
        cutoff=datetime.now(UTC) - timedelta(hours=12),
        batch_limit=100,
        tutorial_run_id=None,
    )
    assert summary.reclaimed >= 1


def trash(session: Session, root: Path, media_id: uuid.UUID, *, purge: bool = False) -> None:
    operation = plan_trash(session, media_id, "worker")
    session.commit()
    execute_operation(session, root, operation, "worker")
    media = session.get(MediaItem, media_id)
    assert media is not None and media.trashed_at is not None
    if purge:
        operation = plan_purge(session, media_id, "worker")
        session.commit()
        execute_purge(session, root, operation, "worker")
        session.refresh(media)
        assert media.purged_at is not None


@pytest.mark.parametrize(
    ("start", "end", "overlap", "lo", "hi"),
    [
        (1.0, 6.0, False, 16000, 96000),
        (1.0, 6.0, True, 40000, 96000),
        (0.00003, 20.00003, False, 0, 160000),
    ],
)
def test_exact_frames_and_no_leak(
    client: TestClient,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    start: float,
    end: float,
    overlap: bool,
    lo: int,
    hi: int,
) -> None:
    with session_factory() as session:
        current, speaker = identities(session)
        source = seed(session, tmp_path, speaker, start=start, end=end, overlap=overlap)
    response = client.get(url(current, speaker))
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == "inline"
    assert response.headers["cache-control"] == "no-store"
    assert audio(response.content) == source.pcm[lo * 2 : hi * 2]
    assert client.get(url(current, speaker)).content == response.content
    assert (
        client.get(url(current, speaker), headers={"Range": "bytes=0-3"}).content
        == response.content
    )
    assert client.head(url(current, speaker)).status_code == 405
    for private in (str(source.media_id), str(source.run_id), str(source.path), "Confirmed voice"):
        assert private.encode() not in response.content
        assert private not in str(response.headers)
    chunks = []
    offset = 12
    while offset < len(response.content):
        name, length = struct.unpack_from("<4sI", response.content, offset)
        chunks.append(name)
        offset += 8 + length + length % 2
    assert chunks == [b"fmt ", b"data"]
    assert client.get(f"/media/{current}/editor/voice-samples").json() == {
        "speakerIds": [str(speaker)]
    }


@pytest.mark.parametrize(
    "state",
    [
        "machine",
        "auto_enroll",
        "trash",
        "purge",
        "reclaim",
        "gate",
        "missing",
        "duplicate",
        "current",
        "archived",
        "short",
        "unknown",
        "archived_run",
    ],
)
def test_ineligible_and_gone(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path, state: str
) -> None:
    with session_factory() as session:
        current, speaker = identities(session)
        source = seed(
            session,
            tmp_path,
            speaker,
            human=state not in {"machine", "auto_enroll"},
            method="auto_enroll" if state == "auto_enroll" else "cosine",
            end=2.99 if state == "short" else 6,
        )
        if state in {"trash", "purge"}:
            trash(session, tmp_path, source.media_id, purge=state == "purge")
        elif state == "reclaim":
            reclaim(session, tmp_path)
        elif state == "gate":
            source.path.unlink()
        elif state == "missing":
            session.query(AudioArtifact).filter_by(pipeline_run_id=source.run_id).delete()
        elif state == "duplicate":
            session.add(
                AudioArtifact(
                    pipeline_run_id=source.run_id,
                    kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                    path="duplicate.wav",
                )
            )
        elif state == "current":
            current = source.media_id
        elif state == "archived":
            archive_speaker(session, speaker)
        elif state == "unknown":
            speaker = uuid.uuid4()
        elif state == "archived_run":
            run = session.get(PipelineRun, source.run_id)
            assert run is not None
            run.archived_at = datetime.now(UTC)
        session.commit()
    response = client.get(url(current, speaker))
    gone = state in {"reclaim", "gate", "missing", "duplicate"}
    assert response.status_code == (410 if gone else 404)
    assert response.json() == {"detail": "voice_sample_gone" if gone else "no_voice_sample"}
    assert response.headers["cache-control"] == "no-store"
    listed = client.get(f"/media/{current}/editor/voice-samples").json()["speakerIds"]
    assert (str(speaker) in listed) == gone


@pytest.mark.parametrize("failure", ["reclaim", "gate", "bounds", "pcm", "floor"])
def test_fallback_and_floor_before_recency(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path, failure: str
) -> None:
    with session_factory() as session:
        current, speaker = identities(session)
        newest = seed(session, tmp_path, speaker, age=1, end=2.2 if failure == "floor" else 6)
        if failure == "reclaim":
            reclaim(session, tmp_path)
        elif failure == "gate":
            newest.path.unlink()
        elif failure in {"bounds", "pcm"}:
            with wave.open(str(newest.path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(1 if failure == "pcm" else 2)
                wav.setframerate(16000)
                wav.writeframes(b"\x01" * (16000 if failure == "bounds" else 400000))
        older = seed(session, tmp_path, speaker, age=2, end=7)
    response = client.get(url(current, speaker))
    assert response.status_code == 200
    assert audio(response.content) == older.pcm[32000:224000]


def test_merges_and_archived_survivor(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        current, alias = identities(session)
        source = seed(session, tmp_path, alias)
        survivor = Speaker(display_name="Survivor")
        session.add(survivor)
        session.flush()
        merge_speakers(session, alias, survivor.id)
        session.commit()
        survivor_id = survivor.id
    assert audio(client.get(url(current, alias)).content) == source.pcm[32000:192000]
    assert client.get(url(current, survivor_id)).content == client.get(url(current, alias)).content
    assert client.get(f"/media/{current}/editor/voice-samples").json() == {
        "speakerIds": [str(survivor_id)]
    }
    with session_factory() as session:
        archive_speaker(session, survivor_id)
        session.commit()
    assert client.get(url(current, alias)).status_code == 404


def test_auth_unknown_media_and_viewer(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    from voxint.api.auth import SESSION_COOKIE, create_session, new_session_token
    from voxint.db.models import UserRole
    from voxint.users import create_user

    with session_factory() as session:
        current, speaker = identities(session)
        seed(session, tmp_path, speaker)
        user = create_user(session, username="viewer", password="viewerpass", role=UserRole.VIEWER)
        session.commit()
        cookie = new_session_token()
        create_session(session, user_id=user.id, token=cookie, ttl_seconds=3600)
        session.commit()
    for path in (url(current, speaker), f"/media/{current}/editor/voice-samples"):
        response = client.get(path, auth=None)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"
    for path in (url(uuid.uuid4(), speaker), f"/media/{uuid.uuid4()}/editor/voice-samples"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"detail": "not found"}
        assert response.headers["cache-control"] == "no-store"
    viewer = TestClient(
        create_app(
            settings=Settings(
                voxint_multi_user=True, media_root=tmp_path, csrf_secret="voice-sample-test-key"
            ),
            session_factory=session_factory,
        ),
        cookies={SESSION_COOKIE: cookie},
    )
    assert viewer.get(url(current, speaker)).status_code == 200
    assert viewer.get(f"/media/{current}/editor/voice-samples").status_code == 200


def test_mixed_corpus_list_agreement(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    speakers = []
    with session_factory() as session:
        current, _ = identities(session)
        for state in ("confirmed", "machine", "trash", "reclaim", "current", "archived"):
            speaker = Speaker(display_name=state)
            session.add(speaker)
            session.flush()
            speakers.append(speaker.id)
            source = seed(session, tmp_path, speaker.id, human=state != "machine", age=0)
            if state == "trash":
                trash(session, tmp_path, source.media_id)
            elif state == "reclaim":
                run = session.get(PipelineRun, source.run_id)
                assert run is not None
                run.updated_at = datetime.now(UTC) - timedelta(days=2)
                session.commit()
                reclaim(session, tmp_path)
            elif state == "current":
                run = session.get(PipelineRun, source.run_id)
                assert run is not None
                run.media_item_id = current
            elif state == "archived":
                archive_speaker(session, speaker.id)
            session.commit()
    listed = client.get(f"/media/{current}/editor/voice-samples").json()["speakerIds"]
    assert listed == sorted(listed)
    statuses = [client.get(url(current, speaker)).status_code for speaker in speakers]
    assert statuses == [200, 404, 404, 410, 404, 404]
    assert set(listed) == {
        str(speaker)
        for speaker, status in zip(speakers, statuses, strict=True)
        if status in {200, 410}
    }


@pytest.mark.parametrize("failure", [None, "bounds", "source", "unexpected"])
def test_source_handle_closed(
    client: TestClient,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    from typing import BinaryIO

    from voxint.api.routers import editor
    from voxint.media.clips import ClipBounds, ClipBoundsError, ClipSourceError, read_clip_frames

    with session_factory() as session:
        current, speaker = identities(session)
        seed(session, tmp_path, speaker)
    original = read_clip_frames
    handles: list[BinaryIO] = []

    def read(source: BinaryIO, bounds: ClipBounds) -> bytes:
        handles.append(source)
        if failure == "bounds":
            raise ClipBoundsError("bad bounds")
        if failure == "source":
            raise ClipSourceError("short read")
        if failure == "unexpected":
            raise RuntimeError("unexpected")
        return original(source, bounds)

    monkeypatch.setattr(editor, "read_clip_frames", read)
    if failure == "unexpected":
        with pytest.raises(RuntimeError, match="unexpected"):
            client.get(url(current, speaker))
    else:
        assert client.get(url(current, speaker)).status_code == (410 if failure else 200)
    assert len(handles) == 1
    assert handles[0].closed


def test_word_range_override_uses_only_child_frames(
    client: TestClient,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    from sqlalchemy import select

    from voxint.db.models import SegmentSplitBoundary

    with session_factory() as session:
        current, speaker = identities(session)
        source = seed(session, tmp_path, speaker, start=0, end=8)
        child_speaker = Speaker(display_name="Child")
        session.add(child_speaker)
        segment = session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == source.run_id)
        ).one()
        segment.raw_text = "One two three four"
        segment.words = [
            {"word": word, "start": i * 2, "end": (i + 1) * 2}
            for i, word in enumerate(["One ", "two ", "three ", "four"])
        ]
        session.flush()
        session.add(
            SegmentSplitBoundary(
                pipeline_run_id=source.run_id,
                parent_segment_id=segment.id,
                word_index=2,
                operator=CREDS[0],
            )
        )
        record_decision(
            session,
            pipeline_run_id=source.run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator=CREDS[0],
            idempotency_key=str(uuid.uuid4()),
            speaker_id=child_speaker.id,
            transcript_segment_id=segment.id,
            start_word_index=0,
            end_word_index=2,
        )
        session.commit()
        child_id = child_speaker.id
    assert audio(client.get(url(current, child_id)).content) == source.pcm[:128000]
    assert audio(client.get(url(current, speaker)).content) == source.pcm[128000:256000]


def test_seeded_library_latency(
    client: TestClient,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """Measure both routes, including a clip that must walk 200 canonical runs."""
    from time import perf_counter

    from sqlalchemy import select

    with session_factory() as session:
        current, speaker = identities(session)
        oldest = seed(session, tmp_path, speaker, age=201)
        artifact = session.scalars(
            select(AudioArtifact).where(AudioArtifact.pipeline_run_id == oldest.run_id)
        ).one()
        for index in range(199):
            media = MediaItem(
                source_path=f"incoming/benchmark-{index}.wav",
                created_at=datetime.now(UTC) - timedelta(days=index),
            )
            session.add(media)
            session.flush()
            run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
            session.add(run)
            session.flush()
            session.add_all(
                [
                    AudioArtifact(
                        pipeline_run_id=run.id,
                        kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                        path=artifact.path,
                    ),
                    TranscriptSegment(
                        pipeline_run_id=run.id,
                        segment_index=0,
                        start_seconds=1,
                        end_seconds=6,
                        raw_text="Machine only",
                        diarization_label="S0",
                    ),
                    DiarizationTurn(
                        pipeline_run_id=run.id,
                        turn_index=0,
                        start_seconds=1,
                        end_seconds=6,
                        label="S0",
                        skip_reason="too_short",
                    ),
                    SpeakerAssignment(
                        pipeline_run_id=run.id,
                        diarization_label="S0",
                        speaker_id=speaker,
                        method="cosine",
                        confidence=0.95,
                        grounded=True,
                    ),
                ]
            )
        session.commit()
    start = perf_counter()
    listing = client.get(f"/media/{current}/editor/voice-samples")
    list_seconds = perf_counter() - start
    start = perf_counter()
    clip = client.get(url(current, speaker))
    clip_seconds = perf_counter() - start
    assert listing.json() == {"speakerIds": [str(speaker)]}
    assert clip.status_code == 200
    assert audio(clip.content) == oldest.pcm[32000:192000]
    print(f"200-run latency: list={list_seconds:.3f}s; worst-case clip={clip_seconds:.3f}s")
