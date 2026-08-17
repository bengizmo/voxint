"""The word-boundary split route + lazy words endpoint (issue #59, slice 2).

Real Postgres (migrated), real templates. Exercises POST
``/review/{run}/segments/{seg}/split`` and GET ``.../words``: a split expands the
parent into derived children (parent-scoped review target), replays idempotently,
is claim-gated and bounds-checked, refuses an unsplittable or corrected segment,
and blocks correcting a split segment.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus, TranscriptSegment

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "split-route-test-csrf-key"

# raw_text is the exact concatenation of the word strings (faster-whisper keeps a
# leading space on each token) → a splittable segment with three words.
_WORDS = [
    {"start": 0.0, "end": 0.4, "word": "Hello", "confidence": 0.9},
    {"start": 0.5, "end": 0.9, "word": " there", "confidence": 0.9},
    {"start": 1.0, "end": 1.4, "word": " world", "confidence": 0.9},
]
_RAW = "Hello there world"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(session_factory: sessionmaker[Session], media_root: Path) -> TestClient:
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


_DEFAULT_WORDS = object()  # sentinel: distinguish "unset" from an explicit None


def _seed(
    session_factory: sessionmaker[Session],
    *,
    words: object = _DEFAULT_WORDS,
    raw_text: str = _RAW,
    enhanced_text: str | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    stored_words = _WORDS if words is _DEFAULT_WORDS else words
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        seg = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=2.0,
            raw_text=raw_text,
            enhanced_text=enhanced_text,
            diarization_label="S0",
            words=stored_words,
        )
        session.add(seg)
        session.commit()
        return run.id, seg.id


def _claim(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return resp.headers["location"].split("token=")[1]


def test_split_expands_parent_into_children(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    token = _claim(client, run_id)
    resp = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    segments = body["segments"]
    assert len(segments) == 2  # the one parent is now two derived children
    assert [s["text"] for s in segments] == ["Hello there", "world"]
    # Every child points at the immutable parent as its write target...
    assert {s["sourceSegmentId"] for s in segments} == {str(seg_id)}
    assert {s["segmentId"] for s in segments} == {str(seg_id)}
    # ...but exactly one child is the review-queue entry (parent-scoped counting).
    assert [s["reviewTarget"] for s in segments] == [True, False]
    assert body["progress"] == {"verified": 0, "total": 1}


def test_split_is_idempotent(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    token = _claim(client, run_id)
    first = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 2},
    )
    second = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 2},
    )
    assert first.status_code == second.status_code == 200
    # Replaying the same cut is a structural no-op: still exactly two children.
    assert len(second.json()["segments"]) == 2


def test_split_rejects_wrong_claim_token(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    _claim(client, run_id)
    resp = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": str(uuid.uuid4()), "word_index": 2},
    )
    assert resp.status_code == 409


def test_split_rejects_out_of_bounds_index(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    token = _claim(client, run_id)
    for bad in (0, 3, 99):  # 0 and >= word_count(3) are not interior cuts
        resp = client.post(
            f"/review/{run_id}/segments/{seg_id}/split",
            data={"token": token, "word_index": bad},
        )
        assert resp.status_code == 422, bad


def test_split_rejects_unsplittable_segment(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory, words=None)  # no word timings
    token = _claim(client, run_id)
    resp = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 1},
    )
    assert resp.status_code == 422


def test_split_refuses_corrected_segment(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    token = _claim(client, run_id)
    corrected = client.post(
        f"/review/{run_id}/segments/{seg_id}/text",
        data={"token": token, "text": "Hello there, world!"},
    )
    assert corrected.status_code == 200
    resp = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 2},
    )
    assert resp.status_code == 409


def test_correcting_a_split_segment_is_refused(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    token = _claim(client, run_id)
    split = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 2},
    )
    assert split.status_code == 200
    resp = client.post(
        f"/review/{run_id}/segments/{seg_id}/text",
        data={"token": token, "text": "changed"},
    )
    assert resp.status_code == 409


def test_words_endpoint_reports_splittable_tokens(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory)
    _claim(client, run_id)
    resp = client.get(f"/review/{run_id}/segments/{seg_id}/words")
    assert resp.status_code == 200
    body = resp.json()
    assert body["splittable"] is True
    assert [w["word"] for w in body["words"]] == ["Hello", " there", " world"]


def test_export_reflects_split_children(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The split expands through the ONE shared read path, so the transcript export
    # (not just the console) shows the derived children — an intended export change.
    run_id, seg_id = _seed(session_factory)
    token = _claim(client, run_id)
    client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": 2},
    )
    rows = client.get(f"/review/{run_id}/export.json").json()
    assert [row["text"] for row in rows] == ["Hello there", "world"]
    assert {row["speaker"] for row in rows} == {"S0"}  # children inherit parent speaker


def test_words_endpoint_reports_unsplittable_reason(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id = _seed(session_factory, words=None)
    _claim(client, run_id)
    body = client.get(f"/review/{run_id}/segments/{seg_id}/words").json()
    assert body["splittable"] is False
    assert body["reason"]
    assert body["words"] == []
