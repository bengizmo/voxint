"""Real claim-gated word marks, shown-text status, undo and structural writes."""

import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration import test_segment_undo
from tests.integration.test_segment_undo import _CSRF_KEY, _claim
from tests.integration.test_word_mark_surfaces import mark, seed_markable
from voxint.adjudication.review_state import set_correction
from voxint.adjudication.word_marks import effective_marks
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.api.transcript_view import _run_island_segments
from voxint.db.models import (
    AppSettings,
    PipelineRun,
    SegmentWordMark,
    TranscriptSegment,
    WordMarkAction,
)

mark_client = test_segment_undo.client


@pytest.fixture
def seeded(session_factory: sessionmaker[Session]) -> tuple[uuid.UUID, uuid.UUID]:
    with session_factory() as session:
        return seed_markable(session)


def get_marks(
    client: TestClient, run_id: uuid.UUID, segment_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    response = client.get(
        f"/review/{run_id}/word-marks",
        params={"segment": str(segment_id)} if segment_id else {},
    )
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


def post_mark(
    client: TestClient, seeded: tuple[uuid.UUID, uuid.UUID], token: str,
    action: str, start: int = 0, end: int = 1, *, nonce: str | None = None,
) -> Response:
    run_id, segment_id = seeded
    return client.post(
        f"/review/{run_id}/segments/{segment_id}/word-marks",
        data={"token": token, "nonce": nonce or uuid.uuid4().hex,
              "action": action, "start": start, "end": end},
    )


def undo_mark(
    client: TestClient, run_id: uuid.UUID, token: str, mark_id: str, *, csrf: bool = True,
) -> Response:
    data = {"token": token, "mark_id": mark_id, "nonce": uuid.uuid4().hex}
    if csrf:
        data["csrf_token"] = mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)
    return client.post(f"/review/{run_id}/undo/word-mark", data=data)


def export(client: TestClient, run_id: uuid.UUID) -> bytes:
    response = client.get(
        f"/review/{run_id}/export.md",
        params={"style": "turns", "fillers": "drop", "timestamps": "false"},
    )
    assert response.status_code == 200, response.text
    return response.content


def units(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {u["start"]: u for e in payload["emissions"] for u in e["units"]}


def test_get_detection_focus_and_version(
    mark_client: TestClient, seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    before = get_marks(mark_client, run_id)
    assert before == get_marks(mark_client, run_id)
    assert before["version"].startswith("0.")
    assert before["fillerListDefault"] is True and before["stale"] == []
    assert list(units(before)) == [0]
    assert units(before)[0] == {
        "start": 0, "end": 1, "from": 0, "to": 3,
        "removed": "filler", "protected": False, "mark": None,
    }
    focused = get_marks(mark_client, run_id, segment_id)
    assert list(units(focused)) == list(range(5))
    assert all(u["removed"] is None for i, u in units(focused).items() if i != 0)
    token = _claim(mark_client, run_id)
    written = post_mark(mark_client, seeded, token, "omit", 1, 2)
    assert written.status_code == 200
    after = get_marks(mark_client, run_id)
    assert before["version"] != after["version"]
    assert after == get_marks(mark_client, run_id)


def test_missing_emission_means_nothing_detected(
    mark_client: TestClient, session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session, body="We start now.")
    assert get_marks(mark_client, run_id)["emissions"] == []
    assert len(units(get_marks(mark_client, run_id, segment_id))) == 3


def test_mark_status_offsets_match_island(
    mark_client: TestClient, session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session, body="😀X Um, basically, we start now.")
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        segment.end_seconds = segment.start_seconds + 6
        set_correction(session, segment=segment, text="😀X UM, Basically, we start now!")
        mark(session, run_id, segment_id, 1, WordMarkAction.KEEP)
        mark(session, run_id, segment_id, 2, WordMarkAction.OMIT)
        island = _run_island_segments(session, run_id)[0]
    payload = get_marks(mark_client, run_id)
    annotated = units(payload)
    assert list(annotated) == [1, 2]
    assert annotated[1]["protected"] is True and annotated[1]["removed"] is None
    assert annotated[1]["mark"] == "keep"
    assert annotated[2]["removed"] == "omit" and annotated[2]["mark"] == "omit"
    assert annotated[1]["from"] == 3  # emoji is one code point
    assert [island["text"][u["from"]:u["to"]] for u in annotated.values()] == [
        "UM,", "Basically,",
    ]


@pytest.mark.parametrize("missing", [False, True])
def test_unmarkable_reason_and_write_refusal(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID], missing: bool,
) -> None:
    run_id, segment_id = seeded
    with session_factory() as session:
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        if missing:
            segment.words = None
        else:
            set_correction(session, segment=segment, text="Entirely changed text.")
        session.commit()
    emission = get_marks(mark_client, run_id)["emissions"][0]
    assert emission["markable"] is False and emission["units"] == []
    assert ("no recorded word timings" if missing else "changed too much") in emission["reason"]
    response = post_mark(mark_client, seeded, _claim(mark_client, run_id), "omit")
    assert response.status_code == 409 and response.json()["detail"] == emission["reason"]


def test_stale_sorted_excluded_and_clearable(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    with session_factory() as session:
        mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)
        mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        set_correction(session, segment=segment, text="Different words.")
        session.commit()
    payload = get_marks(mark_client, run_id)
    assert payload["stale"] == [
        {"segmentId": str(segment_id), "start": 0, "end": 1,
         "action": "keep", "segmentStart": 65.0},
        {"segmentId": str(segment_id), "start": 1, "end": 2,
         "action": "omit", "segmentStart": 65.0},
    ]
    assert units(payload) == {}
    token = _claim(mark_client, run_id)
    for start in (0, 1):
        response = post_mark(mark_client, seeded, token, "clear", start, start + 1)
        assert response.status_code == 200
    assert response.json()["stale"] == []
    assert b"Different words." in export(mark_client, run_id)


def test_keep_omit_clear_and_replays(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    token = _claim(mark_client, run_id)
    nonce = uuid.uuid4().hex
    response = post_mark(mark_client, seeded, token, "keep", nonce=nonce)
    assert response.status_code == 200
    undo = response.json()["undo"]
    assert undo["kind"] == "word-mark"
    with session_factory() as session:
        row = session.get(SegmentWordMark, uuid.UUID(undo["markId"]))
        assert row is not None and row.operator == "reviewer"
        assert datetime.fromisoformat(undo["expiresAt"]) == row.created_at + timedelta(seconds=300)
    replay = post_mark(mark_client, seeded, token, "keep", nonce=nonce)
    assert replay.status_code == 200 and replay.json()["undo"] is None
    assert post_mark(mark_client, seeded, token, "omit", nonce=nonce).status_code == 409
    assert post_mark(mark_client, seeded, token, "keep", 2, 3).status_code == 409
    omitted = post_mark(mark_client, seeded, token, "omit", 2, 3)
    assert omitted.status_code == 200 and units(omitted.json())[2]["removed"] == "omit"
    clear_nonce = uuid.uuid4().hex
    cleared = post_mark(mark_client, seeded, token, "clear", nonce=clear_nonce)
    assert cleared.status_code == 200 and units(cleared.json())[0]["mark"] is None
    assert units(cleared.json())[0]["removed"] == "filler"
    assert post_mark(mark_client, seeded, token, "clear").status_code == 409
    replay = post_mark(mark_client, seeded, token, "clear", nonce=clear_nonce)
    assert replay.status_code == 200 and replay.json()["undo"] is None
    with session_factory() as session:
        rows = session.scalars(select(SegmentWordMark).where(
            SegmentWordMark.pipeline_run_id == run_id,
        )).all()
        assert len(rows) == 3
        assert effective_marks(session, run_id) == {(segment_id, 2, 3): "omit"}


@pytest.mark.parametrize("action,start,end,status", [
    ("omit", 0, 2, 422), ("omit", -1, 1, 422), ("omit", 0, 99, 422),
    ("unknown", 0, 1, 422), ("clear", 0, 1, 409), ("keep", 1, 2, 409),
])
def test_write_validation(
    mark_client: TestClient, seeded: tuple[uuid.UUID, uuid.UUID],
    action: str, start: int, end: int, status: int,
) -> None:
    response = post_mark(mark_client, seeded, _claim(mark_client, seeded[0]), action, start, end)
    assert response.status_code == status
    if action == "omit":
        assert response.json()["detail"] == "choose a whole word"


def test_lost_claim(
    mark_client: TestClient, seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    token = _claim(mark_client, seeded[0])
    _claim(mark_client, seeded[0])
    response = post_mark(mark_client, seeded, token, "omit")
    assert response.status_code == 409 and response.headers["X-Voxint-Conflict"] == "claim"


def test_archived_run_cannot_be_claimed(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    with session_factory() as session:
        run = session.get(PipelineRun, seeded[0])
        assert run is not None
        run.archived_at = datetime.now(UTC)
        session.commit()
    response = mark_client.post(f"/review/{seeded[0]}/claim", data={
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM),
    })
    assert response.status_code == 409 and "archived" in response.json()["detail"]


def test_unknown_run_and_foreign_segment(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    with session_factory() as session:
        _, foreign = seed_markable(session)
    run_id, _ = seeded
    assert mark_client.get(f"/review/{uuid.uuid4()}/word-marks").status_code == 404
    for segment in (foreign, uuid.uuid4()):
        assert mark_client.get(
            f"/review/{run_id}/word-marks", params={"segment": str(segment)},
        ).status_code == 404
        assert post_mark(
            mark_client, (run_id, segment), _claim(mark_client, run_id), "omit",
        ).status_code == 404


def test_undo_restores_export_and_replays_without_nonce(
    mark_client: TestClient, seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    token = _claim(mark_client, run_id)
    before = export(mark_client, run_id)
    keep = post_mark(mark_client, seeded, token, "keep")
    assert keep.status_code == 200 and export(mark_client, run_id) != before
    mark_id = keep.json()["undo"]["markId"]
    response = undo_mark(mark_client, run_id, token, mark_id)
    assert response.status_code == 200 and len(units(response.json())) == 5
    assert export(mark_client, run_id) == before
    response = mark_client.post(f"/review/{run_id}/undo/word-mark", data={
        "token": token, "mark_id": mark_id,
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM),
    })
    assert response.status_code == 200 and export(mark_client, run_id) == before
    assert response.json()["emissions"][0]["segmentId"] == str(segment_id)


@pytest.mark.parametrize("refusal", ["drift", "expired", "csrf", "missing", "foreign"])
def test_undo_refusals(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID], refusal: str,
) -> None:
    run_id, _ = seeded
    token = _claim(mark_client, run_id)
    if refusal == "expired":
        mark_client.app.state.settings.UNDO_GRACE_SECONDS = 1
    keep = post_mark(mark_client, seeded, token, "keep")
    assert keep.status_code == 200
    mark_id = keep.json()["undo"]["markId"]
    if refusal == "drift":
        assert post_mark(mark_client, seeded, token, "omit").status_code == 200
    elif refusal == "expired":
        time.sleep(1.05)
    elif refusal == "missing":
        mark_id = str(uuid.uuid4())
    elif refusal == "foreign":
        with session_factory() as session:
            other_run, other_segment = seed_markable(session)
            mark_id = str(mark(session, other_run, other_segment, 0, WordMarkAction.KEEP).id)
    response = undo_mark(mark_client, run_id, token, mark_id, csrf=refusal != "csrf")
    expected = 422 if refusal == "csrf" else 400 if refusal in ("missing", "foreign") else 409
    assert response.status_code == expected


def test_correction_keeps_mapped_marks_clears_changed_and_never_resurrects(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    token = _claim(mark_client, run_id)
    assert post_mark(mark_client, seeded, token, "keep").status_code == 200
    assert post_mark(mark_client, seeded, token, "omit", 1, 2).status_code == 200
    path = f"/review/{run_id}/segments/{segment_id}/text"
    response = mark_client.post(path, data={"token": token, "text": "UM! Basically, WE start NOW."})
    assert response.status_code == 200 and response.json()["marksCleared"] == 0
    with session_factory() as session:
        assert len(effective_marks(session, run_id)) == 2
    response = mark_client.post(path, data={"token": token, "text": "Completely rewritten."})
    assert response.status_code == 200 and response.json()["marksCleared"] == 2
    assert response.json()["text"] == "Completely rewritten."
    assert b"Completely rewritten." in export(mark_client, run_id)
    with session_factory() as session:
        rows = session.scalars(select(SegmentWordMark).where(
            SegmentWordMark.pipeline_run_id == run_id, SegmentWordMark.action == "clear",
        )).all()
        assert len(rows) == 2
        assert all(
            r.operator == "reviewer" and r.idempotency_key.startswith("text-clear:")
            for r in rows
        )
        assert all(r.user_id == rows[0].user_id for r in rows)
        assert not effective_marks(session, run_id)
    response = mark_client.post(path, data={"token": token, "text": ""})
    assert response.status_code == 200 and response.json()["marksCleared"] == 0
    assert units(get_marks(mark_client, run_id))[0]["mark"] is None
    assert b"Um," not in export(mark_client, run_id)


def test_split_at_boundary_keeps_mark_and_child_offsets(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    token = _claim(mark_client, run_id)
    assert post_mark(mark_client, seeded, token, "keep").status_code == 200
    response = mark_client.post(f"/review/{run_id}/segments/{segment_id}/split", data={
        "token": token, "word_index": 2,
    })
    assert response.status_code == 200 and len(response.json()["segments"]) == 2
    payload = get_marks(mark_client, run_id, segment_id)
    assert [(e["wordStart"], e["wordEnd"]) for e in payload["emissions"]] == [(0, 2), (2, 5)]
    assert payload["stale"] == [] and units(payload)[0]["protected"] is True
    with session_factory() as session:
        island = _run_island_segments(session, run_id)
    assert [
        line["text"][u["from"]:u["to"]]
        for line, emission in zip(island, payload["emissions"], strict=True)
        for u in emission["units"]
    ] == ["Um,", "basically,", "we", "start", "now."]
    assert b"Um, basically, we start now." in export(mark_client, run_id)


def test_split_inside_glued_mark_refused_then_clear_allows(
    mark_client: TestClient, session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session, body="um-hm we start")
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        segment.words = [
            {"word": word, "start": 65 + i, "end": 66 + i}
            for i, word in enumerate(("um", "-hm", " we", " start"))
        ]
        session.commit()
    token = _claim(mark_client, run_id)
    target = (run_id, segment_id)
    assert post_mark(mark_client, target, token, "omit", 0, 1).status_code == 422
    assert post_mark(mark_client, target, token, "omit", 0, 2).status_code == 200
    path = f"/review/{run_id}/segments/{segment_id}/split"
    response = mark_client.post(path, data={"token": token, "word_index": 1})
    assert response.status_code == 409
    assert response.json()["detail"] == "This word has a clean-up mark. Clear it first, then split."
    assert post_mark(mark_client, target, token, "clear", 0, 2).status_code == 200
    assert mark_client.post(path, data={"token": token, "word_index": 1}).status_code == 200


@pytest.mark.parametrize("comma", [False, True])
def test_phrase_after_omit_trace_agrees_with_export(
    mark_client: TestClient, session_factory: sessionmaker[Session], comma: bool,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(
            session, body="you, basically, know, it froze" if comma
            else "you basically, know, it froze",
        )
        saved = session.get(AppSettings, 1)
        assert saved is not None
        saved.fillers_add = {"en": ["you know"]}
        session.commit()
    before = get_marks(mark_client, run_id)
    assert before["fillerListDefault"] is False and units(before) == {}
    token = _claim(mark_client, run_id)
    response = post_mark(mark_client, (run_id, segment_id), token, "omit", 1, 2)
    assert response.status_code == 200
    cause = None if comma else "phrase"
    assert {i: u["removed"] for i, u in units(response.json()).items()} == {
        0: cause, 1: "omit", 2: cause, 3: None, 4: None,
    }
    content = export(mark_client, run_id)
    assert b"basically," not in content and b"words you left out: 1" in content
    if comma:
        # The retained comma prevents the phrase, as in the existing filter tests.
        assert b"you, know, it froze" in content and b"Filler words left out: 0" in content
    else:
        assert b"It froze" in content
        assert b"know," not in content and b"Filler words left out: 2" in content


def test_keep_inside_phrase_protects_all_source_units(
    mark_client: TestClient, session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session, body="you know, it froze")
        saved = session.get(AppSettings, 1)
        assert saved is not None
        saved.fillers_add = {"en": ["you know"]}
        session.commit()
    assert [u["removed"] for u in units(get_marks(mark_client, run_id)).values()] == [
        "phrase", "phrase",
    ]
    response = post_mark(
        mark_client, (run_id, segment_id), _claim(mark_client, run_id), "keep", 1, 2,
    )
    assert response.status_code == 200
    annotated = units(response.json())
    assert annotated[0]["protected"] and annotated[1]["protected"]
    assert annotated[0]["mark"] is None and annotated[1]["mark"] == "keep"
    assert b"you know, it froze" in export(mark_client, run_id)


def test_invalid_filler_list_is_409(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    with session_factory() as session:
        saved = session.get(AppSettings, 1)
        assert saved is not None
        saved.fillers_add = {"en": ["invalid123"]}
        session.commit()
    response = mark_client.get(f"/review/{seeded[0]}/word-marks")
    assert response.status_code == 409
    token = _claim(mark_client, seeded[0])
    assert post_mark(mark_client, seeded, token, "omit").status_code == 409


def test_correction_clears_only_its_parent(
    mark_client: TestClient, session_factory: sessionmaker[Session],
    seeded: tuple[uuid.UUID, uuid.UUID],
) -> None:
    run_id, segment_id = seeded
    with session_factory() as session:
        second = TranscriptSegment(
            pipeline_run_id=run_id, segment_index=1, start_seconds=75, end_seconds=77,
            raw_text="um now", diarization_label="S0", words=[
                {"word": "um", "start": 75, "end": 76},
                {"word": " now", "start": 76, "end": 77},
            ],
        )
        session.add(second)
        session.commit()
        other_id = second.id
        mark(session, run_id, other_id, 0, WordMarkAction.KEEP)
        mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
    response = mark_client.post(f"/review/{run_id}/segments/{segment_id}/text", data={
        "token": _claim(mark_client, run_id), "text": "Rewritten.",
    })
    assert response.status_code == 200 and response.json()["marksCleared"] == 1
    with session_factory() as session:
        assert effective_marks(session, run_id) == {(other_id, 0, 1): "keep"}
