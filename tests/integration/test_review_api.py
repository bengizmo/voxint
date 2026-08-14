"""The review console end to end: queue → claim → rulings → enrollment → export → media.

Real Postgres (migrated), real templates, real ffprobe on a real WAV.
"""

import io
import uuid
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    ArtifactKind,
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

CREDS = ("reviewer", "s3cret")
SPACE = "titanet-large-v1"
_CSRF_KEY = "review-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def write_wav(path: Path, seconds: float = 0.5) -> None:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    path.write_bytes(buf.getvalue())


def unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim] = 1.0
    return vector


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], media_root: Path
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        review_claim_ttl_seconds=600,
        csrf_secret=_CSRF_KEY,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def seed_run(session: Session, media_root: Path) -> uuid.UUID:
    """A completed run: two labels, segments, one grounded proposal + one hint."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()

    audio_rel = f"artifacts/{run.id}/normalized.wav"
    audio_abs = media_root / audio_rel
    audio_abs.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_abs)
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=audio_rel,
        )
    )

    for index, label in enumerate(["S0", "S0", "S1", "S1"]):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                label=label,
                embedding=unit(0 if label == "S0" else 1),
                embedding_space=SPACE,
            )
        )
    for index, (label, text) in enumerate(
        [("S0", "hello there"), ("S1", "hi back"), ("S0", "how are you")]
    ):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                raw_text=text,
                diarization_label=label,
            )
        )
    known = Speaker(display_name="Known Voice")
    session.add(known)
    session.flush()
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run.id,
            diarization_label="S0",
            speaker_id=known.id,
            method="cosine",
            confidence=0.92,
            grounded=True,
        )
    )
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run.id,
            diarization_label="S1",
            method="llm_hint",
            proposed_name="Norma Newvoice",
            grounded=False,
        )
    )
    session.commit()
    return run.id


def claim_token(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers["location"]
    return location.split("token=")[1]


def test_claim_rejected_without_csrf_token(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # /review/{id}/claim mints the run's claim token, so it has no unguessable
    # token of its own — a CSRF token gates a forged cross-site claim. No token ⇒
    # 403 before claim_run touches the DB (the run stays unclaimed).
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    resp = client.post(f"/review/{run_id}/claim", follow_redirects=False)
    assert resp.status_code == 403


def test_full_review_flow(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    # The run is queued (S1 unresolved), S0 already grounded.
    queue = client.get("/review")
    assert queue.status_code == 200
    assert str(run_id) in queue.text

    token = claim_token(client, run_id)
    page = client.get(f"/review/{run_id}", params={"token": token})
    assert page.status_code == 200
    assert "needs ruling" in page.text  # S1
    assert "Known Voice" in page.text  # S0 machine identity
    assert "Norma Newvoice" in page.text  # hint shown as evidence

    # Enroll the unmatched voice under the hinted name (htmx fragment refresh).
    enroll = client.post(
        f"/review/{run_id}/labels/S1/enroll",
        data={"token": token, "nonce": uuid.uuid4().hex, "display_name": "Norma Newvoice"},
        headers={"HX-Request": "true"},
    )
    assert enroll.status_code == 200
    assert "assigned: Norma Newvoice" in enroll.text
    assert "needs ruling" not in enroll.text

    with session_factory() as session:
        embedding = session.execute(select(SpeakerEmbedding)).scalars().one()
        assert embedding.source_diarization_label == "S1"
        assert embedding.embedding_space == SPACE

    # The queue is now empty and the export resolves names through the resolver.
    assert str(run_id) not in client.get("/review").text
    export = client.get(f"/review/{run_id}/export.txt")
    assert export.status_code == 200
    assert "Known Voice: hello there" in export.text
    assert "Norma Newvoice: hi back" in export.text

    # Release and confirm the slot is free.
    release = client.post(
        f"/review/{run_id}/release", data={"token": token}, follow_redirects=False
    )
    assert release.status_code == 303
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None and run.review_claim_token is None


def test_export_formats_content_types_and_payloads(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Structured/subtitle exports share the CLI formatters; here we prove the
    # routes wire the right media type and the attributed data through. S0 is
    # grounded ("Known Voice"); S1 stays a raw label until ruled on.
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    srt = client.get(f"/review/{run_id}/export.srt")
    assert srt.status_code == 200
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert "1\n00:00:00,000 --> 00:00:08,000\nKnown Voice:\nhello there\n" in srt.text

    vtt = client.get(f"/review/{run_id}/export.vtt")
    assert vtt.status_code == 200
    assert vtt.headers["content-type"].startswith("text/vtt")
    assert vtt.text.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:08.000" in vtt.text

    resp = client.get(f"/review/{run_id}/export.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    rows = resp.json()
    assert [r["speaker"] for r in rows] == ["Known Voice", "S1", "Known Voice"]
    assert rows[0] == {
        "start_seconds": 0.0,
        "end_seconds": 8.0,
        "speaker": "Known Voice",
        "text": "hello there",
    }

    rttm = client.get(f"/review/{run_id}/export.rttm")
    assert rttm.status_code == 200
    assert rttm.headers["content-type"].startswith("text/plain")
    # RTTM carries the four raw diarization turns (labels, never resolved names).
    lines = rttm.text.splitlines()
    assert len(lines) == 4
    assert lines[0] == f"SPEAKER {run_id} 1 0.000 8.000 <NA> <NA> S0 <NA> <NA>"
    assert "Known Voice" not in rttm.text


def test_export_text_variant_and_errors(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    # ?text=raw yields the immutable ASR text (here identical to enhanced, which
    # is NULL in the seed) and is accepted; an unknown variant is a 422.
    assert client.get(f"/review/{run_id}/export.srt", params={"text": "raw"}).status_code == 200
    assert (
        client.get(f"/review/{run_id}/export.srt", params={"text": "bogus"}).status_code == 422
    )

    # An unknown run 404s before any formatting.
    assert client.get(f"/review/{uuid.uuid4()}/export.json").status_code == 404
    assert client.get(f"/review/{uuid.uuid4()}/export.rttm").status_code == 404


def test_decision_correction_and_stale_token(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    token = claim_token(client, run_id)
    exclude = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
        headers={"HX-Request": "true"},
    )
    assert exclude.status_code == 200
    assert "excluded" in exclude.text

    # Correction: a later ruling supersedes at read time.
    unknown = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "unknown"},
        headers={"HX-Request": "true"},
    )
    assert unknown.status_code == 200
    assert "Unknown (S1)" in client.get(f"/review/{run_id}/export.txt").text

    # A rotated claim kills the old token: 409, and no ledger row is written.
    fresh = claim_token(client, run_id)
    assert fresh != token
    stale = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
    )
    assert stale.status_code == 409

    # Unknown action and mismatched assign shape are rejected.
    assert (
        client.post(
            f"/review/{run_id}/labels/S1/decision",
            data={"token": fresh, "nonce": uuid.uuid4().hex, "action": "promote"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/review/{run_id}/labels/S1/decision",
            data={"token": fresh, "nonce": uuid.uuid4().hex, "action": "assign"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/review/{run_id}/labels/NOPE/decision",
            data={"token": fresh, "nonce": uuid.uuid4().hex, "action": "exclude"},
        ).status_code
        == 404
    )
    # A well-formed but nonexistent speaker_id is a clean 422, not a raw
    # IntegrityError masquerading as a replay.
    assert (
        client.post(
            f"/review/{run_id}/labels/S1/decision",
            data={
                "token": fresh,
                "nonce": uuid.uuid4().hex,
                "action": "assign",
                "speaker_id": str(uuid.uuid4()),
            },
        ).status_code
        == 422
    )


def test_assign_to_existing_speaker(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known = session.execute(select(Speaker)).scalars().one()
        known_id = known.id

    token = claim_token(client, run_id)
    assign = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "action": "assign",
            "speaker_id": str(known_id),
        },
        headers={"HX-Request": "true"},
    )
    assert assign.status_code == 200
    assert "assigned: Known Voice" in assign.text
    # Assign-to-existing must NOT create an enrollment centroid.
    with session_factory() as session:
        assert session.execute(select(SpeakerEmbedding)).scalars().all() == []


def test_media_range_serving(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    size = (media_root / "artifacts" / str(run_id) / "normalized.wav").stat().st_size

    head = client.head(f"/media/{run_id}")
    assert head.status_code == 200
    assert head.headers["accept-ranges"] == "bytes"
    assert int(head.headers["content-length"]) == size

    full = client.get(f"/media/{run_id}")
    assert full.status_code == 200
    assert len(full.content) == size
    assert full.content[:4] == b"RIFF"

    partial = client.get(f"/media/{run_id}", headers={"Range": "bytes=4-7"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == f"bytes 4-7/{size}"
    assert len(partial.content) == 4

    suffix = client.get(f"/media/{run_id}", headers={"Range": "bytes=-16"})
    assert suffix.status_code == 206
    assert len(suffix.content) == 16
    assert suffix.headers["content-range"] == f"bytes {size - 16}-{size - 1}/{size}"

    beyond = client.get(f"/media/{run_id}", headers={"Range": f"bytes={size}-"})
    assert beyond.status_code == 416
    assert beyond.headers["content-range"] == f"bytes */{size}"

    missing = client.get(f"/media/{uuid.uuid4()}")
    assert missing.status_code == 404


def test_media_rejects_non_audio_artifact(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        artifact = session.execute(select(AudioArtifact)).scalars().one()
        (media_root / artifact.path).write_bytes(b"definitely not audio")
        session.commit()
    assert client.get(f"/media/{run_id}").status_code == 404


def test_metrics_endpoint_renders_prometheus(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        seed_run(session, media_root)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    body = resp.text
    # A seeded completed run + one enrolled speaker are reflected in the series.
    assert '# TYPE voxint_runs gauge' in body
    assert 'voxint_runs{status="completed"} 1' in body
    assert 'voxint_runs{status="failed"} 0' in body  # zero-filled, never absent
    assert 'voxint_roster_speakers 1' in body
    assert body.endswith("\n")
