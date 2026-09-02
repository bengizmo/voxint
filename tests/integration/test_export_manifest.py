"""Integration tests for the JSON provenance manifest routes (issue #122).

The pure builder logic is unit-tested in ``tests/unit/test_manifest.py``; here we
pin the route contracts against real Postgres: response shape, Content-Disposition,
clip/stage/tag round-trips, stale 409, tag filtering, auth/onboarding gates.
"""

from __future__ import annotations

import io
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.annotations import (
    CaptureEndpoint,
    CapturePayload,
    capture_annotation,
)
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_ANNOTATION_TAGS,
    CSRF_CLIP_EXTRACT,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
    TranscriptAnnotation,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "manifest-test-csrf-key"

_SEG0 = "Hello world there"
_SEG1 = "how are you"


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((8192).to_bytes(2, "little", signed=True) * frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _seed_run(
    session: Session,
    media_root: Path,
    *,
    seconds: float = 1.0,
    media_sha256: str | None = None,
) -> uuid.UUID:
    media = MediaItem(
        source_path=f"incoming/{uuid.uuid4()}.wav",
        duration_seconds=seconds,
        sha256=media_sha256,
    )
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    audio_rel = f"artifacts/{run.id}/normalized.wav"
    _write_wav(media_root / audio_rel, seconds=seconds)
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


def _capture(session: Session, run_id: uuid.UUID, payload: CapturePayload) -> uuid.UUID:
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


def _word_annotation(session: Session, run_id: uuid.UUID, seg_index: int = 0) -> uuid.UUID:
    segs = _seg_ids(session, run_id)
    if seg_index == 0:
        return _capture(session, run_id, CapturePayload(_ep(segs[0], 6), _ep(segs[0], 11), "world"))
    return _capture(session, run_id, CapturePayload(_ep(segs[1], 4), _ep(segs[1], 7), "are"))


def _seed_stage_runs(session: Session, run_id: uuid.UUID) -> None:
    session.add(
        StageRun(
            pipeline_run_id=run_id,
            stage=Stage.TRANSCRIBE.value,
            status=StageStatus.COMPLETED.value,
            attempt=1,
            finished_at=datetime(2026, 8, 25, 10, 0, 0, tzinfo=UTC),
            metrics={
                "model_identity": {
                    "asr": {
                        "reachable": True,
                        "model": "large-v2",
                        "revision": None,
                        "engine": "ct2-legacy",
                    },
                },
            },
        )
    )
    session.add(
        StageRun(
            pipeline_run_id=run_id,
            stage=Stage.DIARIZE_EMBED.value,
            status=StageStatus.COMPLETED.value,
            attempt=1,
            finished_at=datetime(2026, 8, 25, 10, 5, 0, tzinfo=UTC),
            metrics={
                "model_identity": {
                    "diarizer": {
                        "reachable": True,
                        "model": "speaker-diarization-3.1",
                        "revision": None,
                    },
                    "embedder": {
                        "reachable": True,
                        "model": "titanet-large-v2",
                        "revision": None,
                    },
                },
            },
        )
    )
    session.commit()


def _extract_clip(
    client: TestClient, run_id: uuid.UUID, ann_id: uuid.UUID
) -> str:
    resp = client.post(
        f"/review/{run_id}/annotations/{ann_id}/clips",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLIP_EXTRACT)},
    )
    assert resp.status_code == 201, resp.text
    clip_id: str = resp.json()["clipId"]
    return clip_id


def _make_tag(client: TestClient, name: str) -> uuid.UUID:
    resp = client.post(
        "/annotations/tags",
        data={
            "name": name,
            "color": "0",
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_ANNOTATION_TAGS),
        },
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


# ---------------------------------------------------------------------------
# Single-annotation manifest: GET /review/{run_id}/annotations/{ann_id}/export.json
# ---------------------------------------------------------------------------


class TestSingleManifest:
    def test_golden_path(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        fn = f"voxint-{run_id.hex[:8]}-manifest-{ann_id.hex[:8]}.json"
        assert fn in resp.headers["content-disposition"]

        body = resp.json()
        assert body["schema_version"] == 1
        assert body["kind"] == "quote_provenance"
        assert "exported_at" in body

        q = body["quote"]
        assert len(q["lines"]) >= 1
        assert q["lines"][0]["text"] == "world"
        assert q["timing_precision"] == "word"
        assert q["annotation_id"] == ann_id.hex
        assert q["tags"] == []
        assert q["note"] is None
        assert q["source_text_hash"]

        assert body["clip"] is None

        src = body["source"]
        assert src["run_id"] == run_id.hex
        assert src["media_sha256"] is None

        prov = body["pipeline_provenance"]
        assert "app_version" in prov

    def test_with_clip(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)

        clip_id = _extract_clip(client, run_id, ann_id)

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        clip = resp.json()["clip"]
        assert clip is not None
        assert clip["id"] == clip_id.replace("-", "")
        assert clip["download_url"] == f"/runs/{run_id}/clips/{clip_id}"
        assert clip["filename"].startswith("voxint-") and clip["filename"].endswith(".wav")
        assert len(clip["sha256"]) == 64
        assert clip["sample_rate"] == 16000
        assert clip["channels"] == 1

    def test_with_tags_and_note(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            tag_a = _make_tag(client, "alpha")
            tag_b = _make_tag(client, "beta")
            ann_id = capture_annotation(
                session,
                run_id=run_id,
                payload=CapturePayload(
                    _ep(_seg_ids(session, run_id)[0], 6),
                    _ep(_seg_ids(session, run_id)[0], 11),
                    "world",
                ),
                operator="reviewer",
                nonce=uuid.uuid4().hex,
                color_index=0,
                note="important quote",
                tag_ids=[tag_a, tag_b],
            ).id
            session.commit()

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        q = resp.json()["quote"]
        assert sorted(q["tags"]) == ["alpha", "beta"]
        assert q["note"] == "important quote"

    def test_with_media_sha256(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root, media_sha256="abcd1234" * 8)
            ann_id = _word_annotation(session, run_id)

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        assert resp.json()["source"]["media_sha256"] == "abcd1234" * 8

    def test_with_stage_provenance(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            _seed_stage_runs(session, run_id)
            ann_id = _word_annotation(session, run_id)

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        stages = resp.json()["pipeline_provenance"]["stages"]
        t = stages["transcribe"]
        assert t["attempt"] == 1
        assert t["finished_at"] is not None
        assert t["roles"]["asr"]["model"] == "large-v2"
        assert t["roles"]["asr"]["reachable"] is True
        de = stages["diarize_embed"]
        assert de["roles"]["diarizer"]["model"] == "speaker-diarization-3.1"
        assert de["roles"]["embedder"]["model"] == "titanet-large-v2"

    def test_no_stage_runs(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        stages = resp.json()["pipeline_provenance"]["stages"]
        assert stages["transcribe"]["roles"] == {}
        assert stages["transcribe"]["attempt"] == 0
        assert stages["diarize_embed"]["roles"] == {}
        assert stages["diarize_embed"]["attempt"] == 0

    def test_stale_annotation_409(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
            session.execute(
                update(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
                .values(raw_text="MUTATED TEXT HERE!")
            )
            session.commit()

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 409
        assert resp.headers.get("x-voxint-conflict") == "stale"

    def test_unknown_run_404(self, client: TestClient) -> None:
        resp = client.get(f"/review/{uuid.uuid4()}/annotations/{uuid.uuid4()}/export.json")
        assert resp.status_code == 404

    def test_unknown_annotation_404(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
        resp = client.get(f"/review/{run_id}/annotations/{uuid.uuid4()}/export.json")
        assert resp.status_code == 404

    def test_foreign_annotation_404(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_a = _seed_run(session, media_root)
            run_b = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_a)
        resp = client.get(f"/review/{run_b}/annotations/{ann_id}/export.json")
        assert resp.status_code == 404

    def test_deleted_annotation_404(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
            session.execute(
                update(TranscriptAnnotation)
                .where(TranscriptAnnotation.id == ann_id)
                .values(deleted_at=datetime.now(UTC))
            )
            session.commit()
        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 404

    def test_requires_auth(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
        client.auth = None
        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Bulk manifest: GET /review/{run_id}/annotations/export.json
# ---------------------------------------------------------------------------


class TestBulkManifest:
    def test_golden_path(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            _word_annotation(session, run_id, seg_index=0)
            _word_annotation(session, run_id, seg_index=1)

        resp = client.get(f"/review/{run_id}/annotations/export.json")
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert f"voxint-{run_id.hex[:8]}-manifests.json" in resp.headers["content-disposition"]

        body = resp.json()
        assert body["schema_version"] == 1
        assert body["kind"] == "quote_provenance_bundle"
        assert "exported_at" in body
        assert body["source"]["run_id"] == run_id.hex
        assert "stages" in body["pipeline_provenance"]
        assert len(body["quotes"]) == 2
        for entry in body["quotes"]:
            assert "quote" in entry
            assert "clip" in entry

    def test_empty_run(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)

        resp = client.get(f"/review/{run_id}/annotations/export.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"schema_version": 1, "kind": "quote_provenance_bundle", "quotes": []}

    def test_tag_filter(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            tag_id = _make_tag(client, "important")
            segs = _seg_ids(session, run_id)
            capture_annotation(
                session,
                run_id=run_id,
                payload=CapturePayload(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
                operator="reviewer",
                nonce=uuid.uuid4().hex,
                color_index=0,
                tag_ids=[tag_id],
            )
            session.commit()
            _word_annotation(session, run_id, seg_index=1)

        resp = client.get(f"/review/{run_id}/annotations/export.json", params={"tag": str(tag_id)})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["quotes"]) == 1
        assert "important" in body["quotes"][0]["quote"]["tags"]

    def test_tag_filter_or_union(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            tag_alpha = _make_tag(client, "alpha")
            tag_beta = _make_tag(client, "beta")
            segs = _seg_ids(session, run_id)
            capture_annotation(
                session,
                run_id=run_id,
                payload=CapturePayload(_ep(segs[0], 0), _ep(segs[0], 5), "Hello"),
                operator="reviewer",
                nonce=uuid.uuid4().hex,
                color_index=0,
                tag_ids=[tag_alpha],
            )
            session.commit()
            capture_annotation(
                session,
                run_id=run_id,
                payload=CapturePayload(_ep(segs[0], 6), _ep(segs[0], 11), "world"),
                operator="reviewer",
                nonce=uuid.uuid4().hex,
                color_index=1,
                tag_ids=[tag_beta],
            )
            session.commit()
            _word_annotation(session, run_id, seg_index=1)

        resp = client.get(
            f"/review/{run_id}/annotations/export.json",
            params=[("tag", str(tag_alpha)), ("tag", str(tag_beta))],
        )
        assert resp.status_code == 200
        assert len(resp.json()["quotes"]) == 2

    def test_unknown_filter_tag_404(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
        resp = client.get(
            f"/review/{run_id}/annotations/export.json",
            params={"tag": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

    def test_stale_annotation_aborts_bundle(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            _word_annotation(session, run_id, seg_index=0)
            _word_annotation(session, run_id, seg_index=1)
            session.execute(
                update(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
                .values(raw_text="MUTATED TEXT HERE!")
            )
            session.commit()

        resp = client.get(f"/review/{run_id}/annotations/export.json")
        assert resp.status_code == 409
        assert resp.headers.get("x-voxint-conflict") == "stale"

    def test_with_clips(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann0 = _word_annotation(session, run_id, seg_index=0)
            ann1 = _word_annotation(session, run_id, seg_index=1)

        _extract_clip(client, run_id, ann0)
        _extract_clip(client, run_id, ann1)

        resp = client.get(f"/review/{run_id}/annotations/export.json")
        assert resp.status_code == 200
        for entry in resp.json()["quotes"]:
            assert entry["clip"] is not None
            assert len(entry["clip"]["sha256"]) == 64

    def test_reclaimed_clip_excluded(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)

        _extract_clip(client, run_id, ann_id)

        with session_factory() as session:
            session.execute(
                update(AudioArtifact)
                .where(
                    AudioArtifact.pipeline_run_id == run_id,
                    AudioArtifact.kind == ArtifactKind.AUDIO_CLIP.value,
                )
                .values(reclaimed_at=datetime.now(UTC), reclaimed_bytes=1000)
            )
            session.commit()

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        assert resp.status_code == 200
        assert resp.json()["clip"] is None

    def test_unknown_run_404(self, client: TestClient) -> None:
        resp = client.get(f"/review/{uuid.uuid4()}/annotations/export.json")
        assert resp.status_code == 404

    def test_requires_auth(
        self, client: TestClient, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
        client.auth = None
        resp = client.get(f"/review/{run_id}/annotations/export.json")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Onboarding gate
# ---------------------------------------------------------------------------


def test_manifest_routes_require_onboarding(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        csrf_secret=_CSRF_KEY,
    )
    unonboarded = TestClient(create_app(settings=settings, session_factory=session_factory))
    unonboarded.auth = CREDS

    run_id = uuid.uuid4()
    ann_id = uuid.uuid4()
    for url in [
        f"/review/{run_id}/annotations/{ann_id}/export.json",
        f"/review/{run_id}/annotations/export.json",
    ]:
        resp = unonboarded.get(url, follow_redirects=False)
        assert resp.status_code == 303
        assert "/setup" in resp.headers["location"]
