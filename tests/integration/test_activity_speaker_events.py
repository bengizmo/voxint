"""Speaker-identification activity events end to end (issue #162, Console 2.0 P7).

The four adjudication seams (label assign, segment assign, enroll, merge) each
emit exactly one ``speaker_identified`` activity row IN the ruling's transaction
when ``console_activity_enabled`` is on; corrections (exclude / unknown / inherit)
and the flag-off path emit nothing; a merge coalesces to one event; re-asserting a
label's current speaker identifies nothing new; and the Jobs badge (a live-jobs
count) is untouched by an identification.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_merge_api import HX, hidden_fields
from tests.integration.test_review_api import CREDS, claim_token, seed_run
from voxint.api.app import create_app
from voxint.api.jobs_query import jobs_badge_count
from voxint.config import Settings
from voxint.db.models import ActivityEvent, ActivityKind, Speaker, TranscriptSegment

_CSRF_KEY = "review-api-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path):  # type: ignore[no-untyped-def]
    return tmp_path


def _make_client(
    session_factory: sessionmaker[Session], media_root, *, activity: bool
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        review_claim_ttl_seconds=600,
        csrf_secret=_CSRF_KEY,
        console_activity_enabled=activity,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session], media_root) -> TestClient:
    return _make_client(session_factory, media_root, activity=True)


def _speaker_events(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> list[ActivityEvent]:
    with session_factory() as session:
        return list(
            session.query(ActivityEvent)
            .filter_by(
                pipeline_run_id=run_id, kind=ActivityKind.SPEAKER_IDENTIFIED.value
            )
            .order_by(ActivityEvent.id)
        )


def _add_speaker(session_factory: sessionmaker[Session], name: str) -> uuid.UUID:
    with session_factory() as session:
        speaker = Speaker(display_name=name)
        session.add(speaker)
        session.commit()
        return speaker.id


def _decide(
    client: TestClient,
    run_id: uuid.UUID,
    label: str,
    token: str,
    *,
    action: str,
    speaker_id: uuid.UUID | None = None,
):  # type: ignore[no-untyped-def]
    data = {"token": token, "nonce": uuid.uuid4().hex, "action": action}
    if speaker_id is not None:
        data["speaker_id"] = str(speaker_id)
    return client.post(f"/review/{run_id}/labels/{label}/decision", data=data, headers=HX)


def _segment_ids(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> dict[int, uuid.UUID]:
    with session_factory() as session:
        rows = session.execute(
            select(TranscriptSegment.segment_index, TranscriptSegment.id).where(
                TranscriptSegment.pipeline_run_id == run_id
            )
        ).all()
    return {int(idx): sid for idx, sid in rows}


# --- label scope -----------------------------------------------------------


def test_label_assign_emits_one(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    spk = _add_speaker(session_factory, "Alice Anderson")
    token = claim_token(client, run_id)
    assert _decide(client, run_id, "S1", token, action="assign", speaker_id=spk).status_code == 200
    rows = _speaker_events(session_factory, run_id)
    assert len(rows) == 1
    assert rows[0].title == "Alice Anderson"
    assert rows[0].href == f"/jobs/{run_id}"


def test_label_reassert_same_speaker_no_reemit(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    spk = _add_speaker(session_factory, "Alice")
    token = claim_token(client, run_id)
    _decide(client, run_id, "S1", token, action="assign", speaker_id=spk)
    # A fresh nonce re-asserting the SAME speaker changes nothing effective.
    _decide(client, run_id, "S1", token, action="assign", speaker_id=spk)
    assert len(_speaker_events(session_factory, run_id)) == 1


def test_label_reassign_different_speaker_emits_again(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    a = _add_speaker(session_factory, "Alice")
    b = _add_speaker(session_factory, "Bob")
    token = claim_token(client, run_id)
    _decide(client, run_id, "S1", token, action="assign", speaker_id=a)
    _decide(client, run_id, "S1", token, action="assign", speaker_id=b)
    rows = _speaker_events(session_factory, run_id)
    assert [r.title for r in rows] == ["Alice", "Bob"]


def test_label_exclude_and_unknown_are_silent(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    assert _decide(client, run_id, "S1", token, action="exclude").status_code == 200
    assert _decide(client, run_id, "S0", token, action="unknown").status_code == 200
    assert _speaker_events(session_factory, run_id) == []


def test_flag_off_emits_nothing(
    session_factory: sessionmaker[Session], media_root
) -> None:
    off = _make_client(session_factory, media_root, activity=False)
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    spk = _add_speaker(session_factory, "Alice")
    token = claim_token(off, run_id)
    assert _decide(off, run_id, "S1", token, action="assign", speaker_id=spk).status_code == 200
    assert _speaker_events(session_factory, run_id) == []


# --- segment scope ---------------------------------------------------------


def test_segment_assign_emits(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    spk = _add_speaker(session_factory, "Carol")
    seg = _segment_ids(session_factory, run_id)[0]
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/segments/{seg}/relabel",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "action": "assign",
            "speaker_id": str(spk),
        },
        headers=HX,
    )
    assert resp.status_code == 200
    rows = _speaker_events(session_factory, run_id)
    assert len(rows) == 1
    assert rows[0].title == "Carol"


def test_segment_inherit_is_silent(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    seg = _segment_ids(session_factory, run_id)[0]
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/segments/{seg}/relabel",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "inherit"},
        headers=HX,
    )
    assert resp.status_code == 200
    assert _speaker_events(session_factory, run_id) == []


# --- enroll ----------------------------------------------------------------


def test_enroll_emits_roster_name(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/labels/S1/enroll",
        data={"token": token, "nonce": uuid.uuid4().hex, "display_name": "Norma Newvoice"},
        headers=HX,
    )
    assert resp.status_code == 200
    rows = _speaker_events(session_factory, run_id)
    assert len(rows) == 1
    # The title is the authoritative roster name, resolved from the speaker row.
    with session_factory() as session:
        name = session.execute(
            select(Speaker.display_name).where(Speaker.display_name == "Norma Newvoice")
        ).scalar_one()
    assert rows[0].title == name


# --- merge (server-side coalescing) ---------------------------------------


def test_merge_emits_one_event(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()
    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    fields = hidden_fields(preview.text)
    fields["nonce"] = uuid.uuid4().hex
    apply = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert apply.status_code == 200
    rows = _speaker_events(session_factory, run_id)
    assert len(rows) == 1  # ONE event for a two-label merge, not two
    assert rows[0].title == "Known Voice (2 labels)"
    assert rows[0].occurrence_key.startswith("merge:")


def test_merge_replay_stays_one_event(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()
    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    fields = hidden_fields(preview.text)
    fields["nonce"] = uuid.uuid4().hex
    client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    client.post(f"/review/{run_id}/merge", data=fields, headers=HX)  # exact replay
    assert len(_speaker_events(session_factory, run_id)) == 1


# --- badge regression ------------------------------------------------------


def test_identification_leaves_jobs_badge_unchanged(
    client: TestClient, session_factory: sessionmaker[Session], media_root
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        before = jobs_badge_count(session)
    spk = _add_speaker(session_factory, "Alice")
    token = claim_token(client, run_id)
    _decide(client, run_id, "S1", token, action="assign", speaker_id=spk)
    with session_factory() as session:
        assert jobs_badge_count(session) == before  # toasts only; badge is live jobs
