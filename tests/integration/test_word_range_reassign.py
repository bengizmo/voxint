"""Sub-segment reassignment: reassign a derived split child to a speaker
(issue #59 slice 3).

Real Postgres (migrated), real templates. Exercises the word-range arm of POST
``/review/{run}/segments/{seg}/relabel``: after splitting a segment, one child is
reassigned to a different speaker and ONLY that child's export line changes; the
word-range scope beats the whole-segment scope; ``inherit`` removes it live; the
route rejects a range that is not a current split child and a half-set range; and
a ranged ruling replays idempotently.
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
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "reassign-route-test-csrf-key"

# A splittable four-word segment (tokens reconcatenate to raw_text exactly).
_WORDS = [
    {"start": 0.0, "end": 0.4, "word": "Hello"},
    {"start": 0.5, "end": 0.9, "word": " there"},
    {"start": 1.0, "end": 1.4, "word": " big"},
    {"start": 1.5, "end": 1.9, "word": " world"},
]
_RAW = "Hello there big world"


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


def _seed(
    session_factory: sessionmaker[Session],
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """A completed run with one splittable segment + a spare roster speaker.
    Returns ``(run_id, segment_id, other_speaker_id)``."""
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
            raw_text=_RAW,
            diarization_label="S0",
            words=_WORDS,
        )
        other = Speaker(display_name="Other Person")
        session.add_all([seg, other])
        session.commit()
        return run.id, seg.id, other.id


def _claim(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return resp.headers["location"].split("token=")[1]


def _split(
    client: TestClient, run_id: uuid.UUID, seg_id: uuid.UUID, token: str, at: int
) -> None:
    resp = client.post(
        f"/review/{run_id}/segments/{seg_id}/split",
        data={"token": token, "word_index": at},
    )
    assert resp.status_code == 200, resp.text


def _reassign(
    client: TestClient,
    run_id: uuid.UUID,
    seg_id: uuid.UUID,
    token: str,
    *,
    action: str,
    speaker_id: uuid.UUID | None = None,
    start: int | None = None,
    end: int | None = None,
    nonce: str | None = None,
):
    data: dict[str, str] = {
        "token": token,
        "nonce": nonce or uuid.uuid4().hex,
        "action": action,
    }
    if speaker_id is not None:
        data["speaker_id"] = str(speaker_id)
    if start is not None:
        data["start_word_index"] = str(start)
    if end is not None:
        data["end_word_index"] = str(end)
    return client.post(
        f"/review/{run_id}/segments/{seg_id}/relabel", data=data, headers={"HX-Request": "true"}
    )


def _export(client: TestClient, run_id: uuid.UUID) -> list[dict[str, str]]:
    return client.get(f"/review/{run_id}/export.json").json()


def test_reassign_child_changes_only_that_child(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id, other = _seed(session_factory)
    token = _claim(client, run_id)
    _split(client, run_id, seg_id, token, at=2)  # -> ["Hello there", "big world"]
    resp = _reassign(
        client, run_id, seg_id, token, action="assign", speaker_id=other, start=2, end=4
    )
    assert resp.status_code == 200, resp.text
    rows = _export(client, run_id)
    assert [r["text"] for r in rows] == ["Hello there", "big world"]
    # Only the reassigned child ([2,4)) takes the new speaker; the first child
    # keeps the parent's label.
    assert rows[0]["speaker"] == "S0"
    assert rows[1]["speaker"] == "Other Person"


def test_word_range_beats_whole_segment_scope(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id, other = _seed(session_factory)
    other2_name = "Whole Segment Speaker"
    with session_factory() as session:
        whole = Speaker(display_name=other2_name)
        session.add(whole)
        session.commit()
        whole_id = whole.id
    token = _claim(client, run_id)
    _split(client, run_id, seg_id, token, at=2)
    # Whole-segment override first, then a finer word-range override on child [2,4).
    _reassign(client, run_id, seg_id, token, action="assign", speaker_id=whole_id)
    _reassign(client, run_id, seg_id, token, action="assign", speaker_id=other, start=2, end=4)
    rows = _export(client, run_id)
    # Child [0,2) follows the whole-segment override; child [2,4) the finer range.
    assert rows[0]["speaker"] == other2_name
    assert rows[1]["speaker"] == "Other Person"


def test_inherit_removes_the_word_range_override(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id, other = _seed(session_factory)
    token = _claim(client, run_id)
    _split(client, run_id, seg_id, token, at=2)
    _reassign(client, run_id, seg_id, token, action="assign", speaker_id=other, start=2, end=4)
    # Append-only reset: inherit on the same range makes the child follow its
    # label again (live, not a frozen copy).
    resp = _reassign(client, run_id, seg_id, token, action="inherit", start=2, end=4)
    assert resp.status_code == 200, resp.text
    rows = _export(client, run_id)
    assert [r["speaker"] for r in rows] == ["S0", "S0"]


def test_reassign_rejects_range_that_is_not_a_current_child(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id, other = _seed(session_factory)
    token = _claim(client, run_id)
    _split(client, run_id, seg_id, token, at=2)  # children are [0,2) and [2,4)
    # [1,3) straddles the cut — not a real partition; refuse it (409) rather than
    # write a ledger row the read path would silently ignore.
    resp = _reassign(
        client, run_id, seg_id, token, action="assign", speaker_id=other, start=1, end=3
    )
    assert resp.status_code == 409, resp.text


def test_reassign_rejects_half_set_range(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id, other = _seed(session_factory)
    token = _claim(client, run_id)
    _split(client, run_id, seg_id, token, at=2)
    resp = _reassign(
        client, run_id, seg_id, token, action="assign", speaker_id=other, start=2, end=None
    )
    assert resp.status_code == 422, resp.text


def test_ranged_reassign_replays_idempotently(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, seg_id, other = _seed(session_factory)
    token = _claim(client, run_id)
    _split(client, run_id, seg_id, token, at=2)
    nonce = uuid.uuid4().hex
    args = dict(action="assign", speaker_id=other, start=2, end=4, nonce=nonce)
    first = _reassign(client, run_id, seg_id, token, **args)  # type: ignore[arg-type]
    second = _reassign(client, run_id, seg_id, token, **args)  # type: ignore[arg-type]
    assert first.status_code == second.status_code == 200
    rows = _export(client, run_id)
    assert [r["speaker"] for r in rows] == ["S0", "Other Person"]
