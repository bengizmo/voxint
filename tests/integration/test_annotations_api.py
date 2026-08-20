"""The operator annotation layer HTTP surface (issue #86, Step 3).

Thin route handlers over ``voxint.adjudication.annotations`` against real
Postgres, real templates, and the real review claim. The service's coordinate
math / classification / idempotency / staleness is exercised by
``test_annotations_service.py``; this gate pins the ROUTE contracts: auth vs
claim vs CSRF gating, the ``AnnotationError`` -> HTTP mapping and its
``X-Voxint-Conflict`` markers, form parsing, the discriminated PATCH ops, the
OR-union tag filter, the island-prop shape, and the transcript-evidence
immutability guarantee at the route level.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_review_api import CREDS, claim_token
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_ANNOTATION_TAGS, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    MAX_ANNOTATION_NOTE_CHARS,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    TranscriptAnnotation,
    TranscriptSegment,
)

_CSRF_KEY = "review-api-test-csrf-key"

# seg0 "Hello world there": content_start=[0,6,12], content_end=[5,11,17], len 17.
# seg1 "how are you":        content_start=[0,4,8],  content_end=[3,7,11], len 11.
_SEG0 = "Hello world there"
_SEG1 = "how are you"


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


def seed_word_run(session_factory: sessionmaker[Session]) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """A completed run with two word-timed segments, so all three anchor kinds are
    reachable through the routes. Returns (run_id, [seg0_id, seg1_id])."""
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        seg_ids: list[uuid.UUID] = []
        for index, (raw, (lo, hi)) in enumerate([(_SEG0, (0.0, 3.0)), (_SEG1, (3.0, 6.0))]):
            seg = TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=lo,
                end_seconds=hi,
                raw_text=raw,
                diarization_label="S0",
                words=_tokens(raw, lo, hi),
            )
            session.add(seg)
            session.flush()
            seg_ids.append(seg.id)
        session.commit()
        return run.id, seg_ids


def _create_form(
    token: str,
    seg_id: uuid.UUID,
    start_offset: int,
    end_offset: int,
    quote: str,
    *,
    nonce: str = "nonce-001",
    color_index: int = 2,
    note: str | None = None,
    tags: list[uuid.UUID] | None = None,
) -> dict[str, str | list[str]]:
    form: dict[str, str | list[str]] = {
        "token": token,
        "nonce": nonce,
        "start_segment_id": str(seg_id),
        "start_offset": str(start_offset),
        "end_segment_id": str(seg_id),
        "end_offset": str(end_offset),
        "client_quote": quote,
        "color_index": str(color_index),
    }
    if note is not None:
        form["note"] = note
    if tags is not None:
        form["tags"] = [str(t) for t in tags]
    return form


def _mint_tag_csrf() -> str:
    return mint_csrf_token(_CSRF_KEY, CSRF_ANNOTATION_TAGS)


def _make_tag(client: TestClient, name: str, color: int = 0) -> str:
    resp = client.post(
        "/annotations/tags",
        data={"name": name, "color": str(color), "csrf_token": _mint_tag_csrf()},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


# --------------------------------------------------------------------------- #
# GET list + create
# --------------------------------------------------------------------------- #


def test_list_empty_run(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, _ = seed_word_run(session_factory)
    resp = client.get(f"/review/{run_id}/annotations")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"annotations": [], "tags": []}


def test_list_unknown_run_404(client: TestClient) -> None:
    resp = client.get(f"/review/{uuid.uuid4()}/annotations")
    assert resp.status_code == 404


def test_list_unknown_filter_tag_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A mistyped/forged ?tag= fails closed (404), never a silent empty result.
    run_id, _ = seed_word_run(session_factory)
    resp = client.get(f"/review/{run_id}/annotations", params={"tag": str(uuid.uuid4())})
    assert resp.status_code == 404


def test_write_unknown_run_404_not_claim(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # An unknown run on a write route is a 404 (fail-closed taxonomy), not a claim
    # conflict — the missing run is caught before the claim check.
    missing = uuid.uuid4()
    post = client.post(
        f"/review/{missing}/annotations",
        data=_create_form(str(uuid.uuid4()), uuid.uuid4(), 6, 11, "world"),
    )
    assert post.status_code == 404
    patch = client.patch(
        f"/review/{missing}/annotations/{uuid.uuid4()}",
        data={"token": str(uuid.uuid4()), "op": "refresh"},
    )
    assert patch.status_code == 404
    delete = client.request(
        "DELETE",
        f"/review/{missing}/annotations/{uuid.uuid4()}",
        data={"token": str(uuid.uuid4())},
    )
    assert delete.status_code == 404


def test_create_word_range_shape(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", color_index=3),
    )
    assert resp.status_code == 201, resp.text
    shape = resp.json()
    assert shape["anchorKind"] == "word_range"
    assert shape["colorIndex"] == 3
    assert shape["quote"] == "world"
    assert shape["stale"] is False
    assert shape["timingPrecision"] == "word"
    assert shape["startSeconds"] == pytest.approx(1.0)
    assert shape["endSeconds"] == pytest.approx(2.0)
    assert shape["spans"] == [{"lineIndex": 0, "start": 6, "end": 11}]
    assert shape["speakers"] == ["S0"]
    assert shape["tags"] == []
    assert shape["note"] is None

    # And it now lists (transcript order) with the tag universe.
    listing = client.get(f"/review/{run_id}/annotations").json()
    assert [a["id"] for a in listing["annotations"]] == [shape["id"]]


def test_create_text_range_and_segment_range(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    text = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 7, 11, "orld", nonce="nonce-t1"),
    ).json()
    assert text["anchorKind"] == "text_range"
    assert text["timingPrecision"] == "segment"
    assert text["startSeconds"] is not None  # coarse segment fallback
    whole = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 0, 17, _SEG0, nonce="nonce-s1"),
    ).json()
    assert whole["anchorKind"] == "segment_range"


def test_create_requires_claim(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, segs = seed_word_run(session_factory)
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(str(uuid.uuid4()), segs[0], 6, 11, "world"),
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "claim"


def test_create_stale_client_quote_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "WRONG"),
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "stale"


def test_create_bad_color_422(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", color_index=99),
    )
    assert resp.status_code == 422


def test_create_forged_segment_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _ = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, uuid.uuid4(), 6, 11, "world"),
    )
    assert resp.status_code == 404


def test_create_unknown_tag_404(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", tags=[uuid.uuid4()]),
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #


def test_create_replay_same_payload_returns_original(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    first = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", nonce="nonce-dup"),
    ).json()
    again = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", nonce="nonce-dup"),
    )
    assert again.status_code == 201
    assert again.json()["id"] == first["id"]


def test_create_replay_different_payload_409_idempotency(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", nonce="nonce-dup"),
    )
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 0, 17, _SEG0, nonce="nonce-dup"),
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "idempotency"


# --------------------------------------------------------------------------- #
# PATCH: edit / refresh / reanchor
# --------------------------------------------------------------------------- #


def _create_one(
    client: TestClient, run_id: uuid.UUID, token: str, seg_id: uuid.UUID, **kw: object
) -> str:
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, seg_id, 6, 11, "world", **kw),  # type: ignore[arg-type]
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def test_patch_edit_replaces_metadata(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    tag = _make_tag(client, "Key Point")
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={
            "token": token,
            "op": "edit",
            "color_index": "5",
            "note": "look here",
            "tags": [tag],
        },
    )
    assert resp.status_code == 200, resp.text
    shape = resp.json()
    assert shape["colorIndex"] == 5
    assert shape["note"] == "look here"
    assert [t["id"] for t in shape["tags"]] == [tag]


def test_patch_edit_missing_color_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "edit", "note": "x"},
    )
    assert resp.status_code == 422


def test_patch_refresh_noop_when_unchanged(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "refresh"},
    )
    assert resp.status_code == 200
    assert resp.json()["stale"] is False


def test_patch_refresh_text_range_stale_after_correction_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    # A text_range on seg0, then correct seg0 so the hash changes and the offsets die.
    resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 7, 11, "orld"),
    )
    ann_id = resp.json()["id"]
    with session_factory() as session:
        session.add(
            SegmentReviewState(
                transcript_segment_id=segs[0],
                pipeline_run_id=run_id,
                corrected_text="Totally different words entirely",
                corrected_at=datetime.now(UTC),
            )
        )
        session.commit()
    refreshed = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "refresh"},
    )
    assert refreshed.status_code == 409
    assert refreshed.headers["X-Voxint-Conflict"] == "stale"


def test_patch_reanchor_moves_anchor(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={
            "token": token,
            "op": "reanchor",
            "start_segment_id": str(segs[0]),
            "start_offset": "0",
            "end_segment_id": str(segs[0]),
            "end_offset": "5",
            "client_quote": "Hello",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["quote"] == "Hello"


def test_patch_reanchor_missing_payload_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "reanchor", "start_offset": "0"},
    )
    assert resp.status_code == 422


def test_patch_unknown_op_422(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "bogus", "color_index": "1"},
    )
    assert resp.status_code == 422


def test_patch_unknown_id_404(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, _ = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.patch(
        f"/review/{run_id}/annotations/{uuid.uuid4()}",
        data={"token": token, "op": "refresh"},
    )
    assert resp.status_code == 404


def test_patch_requires_claim(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    resp = client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": str(uuid.uuid4()), "op": "refresh"},
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "claim"


# --------------------------------------------------------------------------- #
# DELETE (idempotent soft delete)
# --------------------------------------------------------------------------- #


def test_delete_is_idempotent_204(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann_id = _create_one(client, run_id, token, segs[0])
    first = client.request(
        "DELETE", f"/review/{run_id}/annotations/{ann_id}", data={"token": token}
    )
    assert first.status_code == 204
    again = client.request(
        "DELETE", f"/review/{run_id}/annotations/{ann_id}", data={"token": token}
    )
    assert again.status_code == 204
    # Excluded from the listing.
    listing = client.get(f"/review/{run_id}/annotations").json()
    assert listing["annotations"] == []


def test_delete_unknown_404(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    run_id, _ = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    resp = client.request(
        "DELETE", f"/review/{run_id}/annotations/{uuid.uuid4()}", data={"token": token}
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Tag CRUD
# --------------------------------------------------------------------------- #


def test_tag_create_and_list(client: TestClient) -> None:
    tag_id = _make_tag(client, "Key Point", color=1)
    listing = client.get("/annotations/tags").json()
    assert listing == {"tags": [{"id": tag_id, "name": "Key Point", "color": 1, "archived": False}]}


def test_tag_create_requires_csrf(client: TestClient) -> None:
    resp = client.post("/annotations/tags", data={"name": "x", "color": "0"})
    assert resp.status_code == 403


def test_tag_create_normalized_duplicate_409(client: TestClient) -> None:
    _make_tag(client, "Key Point")
    resp = client.post(
        "/annotations/tags",
        data={"name": "  key POINT ", "color": "0", "csrf_token": _mint_tag_csrf()},
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "duplicate-tag"


def test_tag_create_blank_name_422(client: TestClient) -> None:
    resp = client.post(
        "/annotations/tags",
        data={"name": "   ", "color": "0", "csrf_token": _mint_tag_csrf()},
    )
    assert resp.status_code == 422


def test_tag_patch_rename_recolor(client: TestClient) -> None:
    tag_id = _make_tag(client, "Key Point", color=0)
    resp = client.patch(
        f"/annotations/tags/{tag_id}",
        data={"name": "Highlight", "color": "4", "csrf_token": _mint_tag_csrf()},
    )
    assert resp.status_code == 200
    assert resp.json() == {"id": tag_id, "name": "Highlight", "color": 4, "archived": False}


def test_tag_patch_rename_to_existing_409(client: TestClient) -> None:
    _make_tag(client, "Alpha")
    beta_id = _make_tag(client, "Beta")
    resp = client.patch(
        f"/annotations/tags/{beta_id}",
        data={"name": "alpha", "csrf_token": _mint_tag_csrf()},
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "duplicate-tag"


def test_tag_patch_archive_then_restore(client: TestClient) -> None:
    tag_id = _make_tag(client, "Key Point")
    archived = client.patch(
        f"/annotations/tags/{tag_id}",
        data={"archived": "true", "csrf_token": _mint_tag_csrf()},
    )
    assert archived.status_code == 200
    assert archived.json()["archived"] is True
    restored = client.patch(
        f"/annotations/tags/{tag_id}",
        data={"archived": "false", "csrf_token": _mint_tag_csrf()},
    )
    assert restored.json()["archived"] is False


def test_tag_patch_unknown_404(client: TestClient) -> None:
    resp = client.patch(
        f"/annotations/tags/{uuid.uuid4()}",
        data={"name": "x", "csrf_token": _mint_tag_csrf()},
    )
    assert resp.status_code == 404


def test_tag_patch_requires_csrf(client: TestClient) -> None:
    tag_id = _make_tag(client, "Key Point")
    resp = client.patch(f"/annotations/tags/{tag_id}", data={"name": "y"})
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# tag OR-union filter
# --------------------------------------------------------------------------- #


def test_list_tag_or_union_filter(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    tag_a = _make_tag(client, "Alpha")
    tag_b = _make_tag(client, "Beta")
    a = _create_one(client, run_id, token, segs[0], nonce="nonce-a1", tags=[uuid.UUID(tag_a)])
    # A second, untagged annotation and one tagged Beta.
    _create_one(client, run_id, token, segs[0], nonce="nonce-a2")
    b_resp = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 0, 17, _SEG0, nonce="nonce-b1", tags=[uuid.UUID(tag_b)]),
    )
    b = b_resp.json()["id"]
    # ?tag=A&tag=B -> the two tagged ones (union), not the untagged.
    filtered = client.get(f"/review/{run_id}/annotations", params={"tag": [tag_a, tag_b]}).json()
    assert {ann["id"] for ann in filtered["annotations"]} == {a, b}


# --------------------------------------------------------------------------- #
# island props + route-level immutability
# --------------------------------------------------------------------------- #


def test_review_transcript_hydrates_annotation_props(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    _create_one(client, run_id, token, segs[0])
    _make_tag(client, "Key Point")
    resp = client.get(f"/review/{run_id}/transcript", params={"token": token})
    assert resp.status_code == 200
    html = resp.text
    # The review-stepper island hydrates the annotation props (shape pinned in
    # the JSON list tests; here we pin that the review page carries them + the caps
    # + a tag-CRUD CSRF token).
    for key in ("annotations", "annotationTags", "annotationLimits", "tagCsrf"):
        assert key in html
    for cap in ("paletteSize", "maxSpanSegments", "maxNoteChars", "maxTagsPerAnnotation"):
        assert cap in html


def _evidence_snapshot(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> list[tuple[str, str | None, str | None]]:
    with session_factory() as session:
        segs = (
            session.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.pipeline_run_id == run_id)
                .order_by(TranscriptSegment.segment_index)
            )
            .scalars()
            .all()
        )
        review = {
            r.transcript_segment_id: r.corrected_text
            for r in session.execute(
                select(SegmentReviewState).where(SegmentReviewState.pipeline_run_id == run_id)
            ).scalars()
        }
        return [(s.raw_text, s.enhanced_text, review.get(s.id)) for s in segs]


def test_routes_never_touch_transcript_evidence(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The #86 route-level analogue of the ledger-trigger immutability test: the
    full capture/patch/refresh/delete lifecycle never writes raw/enhanced/corrected
    transcript text."""
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    before = _evidence_snapshot(session_factory, run_id)

    ann_id = _create_one(client, run_id, token, segs[0])
    client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "edit", "color_index": "1", "note": "n"},
    )
    client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={"token": token, "op": "refresh"},
    )
    client.patch(
        f"/review/{run_id}/annotations/{ann_id}",
        data={
            "token": token,
            "op": "reanchor",
            "start_segment_id": str(segs[0]),
            "start_offset": "0",
            "end_segment_id": str(segs[0]),
            "end_offset": "5",
            "client_quote": "Hello",
        },
    )
    client.request("DELETE", f"/review/{run_id}/annotations/{ann_id}", data={"token": token})

    assert _evidence_snapshot(session_factory, run_id) == before


# --------------------------------------------------------------------------- #
# Pull-quote export (issue #86 Landing 2): per-row, bulk, live
# --------------------------------------------------------------------------- #


def _live_form(
    seg_id: uuid.UUID,
    start_offset: int,
    end_offset: int,
    quote: str,
    *,
    note: str | None = None,
    tags: list[uuid.UUID] | None = None,
) -> dict[str, str | list[str]]:
    form: dict[str, str | list[str]] = {
        "start_segment_id": str(seg_id),
        "start_offset": str(start_offset),
        "end_segment_id": str(seg_id),
        "end_offset": str(end_offset),
        "client_quote": quote,
    }
    if note is not None:
        form["note"] = note
    if tags is not None:
        form["tags"] = [str(t) for t in tags]
    return form


def test_export_single_word_range_markdown(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world"),
    ).json()
    # A read, so no claim/token required.
    resp = client.get(f"/review/{run_id}/annotations/{ann['id']}/export.md")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/markdown")
    body = resp.text
    assert "## S0" in body
    assert "> world" in body  # reading copy: no bracket in the body
    assert "**Source:**" in body
    assert "[00:00:01.000\u201300:00:02.000]" in body  # precise word timing in the citation
    assert "≈" not in body  # word timing is precise, no approximate marker


def test_export_single_includes_tags_and_note(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    tag_id = uuid.UUID(_make_tag(client, "Key Point"))
    ann = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", note="worth a look", tags=[tag_id]),
    ).json()
    body = client.get(f"/review/{run_id}/annotations/{ann['id']}/export.md").text
    assert "**Tags:** Key Point" in body
    assert "**Note:** worth a look" in body


def test_export_single_unknown_id_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _ = seed_word_run(session_factory)
    resp = client.get(f"/review/{run_id}/annotations/{uuid.uuid4()}/export.md")
    assert resp.status_code == 404


def test_export_single_unknown_run_404(client: TestClient) -> None:
    resp = client.get(f"/review/{uuid.uuid4()}/annotations/{uuid.uuid4()}/export.md")
    assert resp.status_code == 404


def test_export_single_stale_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    ann = client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 7, 11, "orld"),
    ).json()
    with session_factory() as session:
        session.add(
            SegmentReviewState(
                transcript_segment_id=segs[0],
                pipeline_run_id=run_id,
                corrected_text="Totally different words entirely",
                corrected_at=datetime.now(UTC),
            )
        )
        session.commit()
    resp = client.get(f"/review/{run_id}/annotations/{ann['id']}/export.md")
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "stale"


def test_export_bulk_order_matches_panel(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    # Create out of transcript order; the panel and the bulk export must agree.
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[1], 0, 3, "how", nonce="nonce-02b"),
    )
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", nonce="nonce-01a"),
    )
    listing = client.get(f"/review/{run_id}/annotations").json()["annotations"]
    panel_quotes = [a["quote"] for a in listing]
    assert panel_quotes == ["world", "how"]  # transcript (line) order, not creation
    bulk = client.get(f"/review/{run_id}/annotations/export.md")
    assert bulk.status_code == 200
    # Each quote body appears, in the same order, separated by the thematic break.
    blocks = bulk.text.split("\n---\n\n")
    assert len(blocks) == 2
    assert "world" in blocks[0] and "how" in blocks[1]


def test_export_bulk_or_tag_filter(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    tag_a = uuid.UUID(_make_tag(client, "aaa"))
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 6, 11, "world", nonce="nonce-01a", tags=[tag_a]),
    )
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[1], 0, 3, "how", nonce="nonce-02b"),
    )
    filtered = client.get(f"/review/{run_id}/annotations/export.md", params={"tag": str(tag_a)})
    assert filtered.status_code == 200
    assert "world" in filtered.text
    assert "how" not in filtered.text
    assert "\n---\n\n" not in filtered.text  # only one block matched


def test_export_bulk_unknown_tag_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _ = seed_word_run(session_factory)
    resp = client.get(
        f"/review/{run_id}/annotations/export.md", params={"tag": str(uuid.uuid4())}
    )
    assert resp.status_code == 404


def test_export_bulk_empty_is_empty_body(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _ = seed_word_run(session_factory)
    resp = client.get(f"/review/{run_id}/annotations/export.md")
    assert resp.status_code == 200
    assert resp.text == ""


def test_export_bulk_stale_fails_atomically(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    token = claim_token(client, run_id)
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[1], 0, 3, "how", nonce="nonce-02b"),
    )
    client.post(
        f"/review/{run_id}/annotations",
        data=_create_form(token, segs[0], 7, 11, "orld", nonce="nonce-01a"),
    )
    with session_factory() as session:
        session.add(
            SegmentReviewState(
                transcript_segment_id=segs[0],
                pipeline_run_id=run_id,
                corrected_text="Totally different words entirely",
                corrected_at=datetime.now(UTC),
            )
        )
        session.commit()
    resp = client.get(f"/review/{run_id}/annotations/export.md")
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "stale"


def _count_annotations(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> int:
    with session_factory() as session:
        return len(
            session.execute(
                select(TranscriptAnnotation).where(
                    TranscriptAnnotation.pipeline_run_id == run_id
                )
            ).scalars().all()
        )


def test_live_pull_quote_persists_nothing_no_claim(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    # No claim, no nonce, no CSRF — a read-shaped POST that writes nothing.
    resp = client.post(
        f"/review/{run_id}/annotations/export/live.md",
        data=_live_form(segs[0], 6, 11, "world", note="scratch"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "> world" in resp.text
    assert "[00:00:01.000\u201300:00:02.000]" in resp.text
    assert "**Note:** scratch" in resp.text
    assert _count_annotations(session_factory, run_id) == 0


def test_live_pull_quote_stale_client_quote_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    resp = client.post(
        f"/review/{run_id}/annotations/export/live.md",
        data=_live_form(segs[0], 6, 11, "WRONG"),
    )
    assert resp.status_code == 409
    assert resp.headers["X-Voxint-Conflict"] == "stale"


def test_live_pull_quote_empty_selection_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segs = seed_word_run(session_factory)
    resp = client.post(
        f"/review/{run_id}/annotations/export/live.md",
        data=_live_form(segs[0], 6, 6, ""),
    )
    assert resp.status_code == 422


def test_live_pull_quote_forged_segment_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _ = seed_word_run(session_factory)
    resp = client.post(
        f"/review/{run_id}/annotations/export/live.md",
        data=_live_form(uuid.uuid4(), 6, 11, "world"),
    )
    assert resp.status_code == 404


def test_live_pull_quote_over_cap_note_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Create-parity: the live route must enforce the same note cap as create
    # (via normalize_note), so an over-cap note is 422 and nothing is written —
    # never a full-length quote returned (issue #86 review fix).
    run_id, segs = seed_word_run(session_factory)
    resp = client.post(
        f"/review/{run_id}/annotations/export/live.md",
        data=_live_form(segs[0], 6, 11, "world", note="x" * (MAX_ANNOTATION_NOTE_CHARS + 1)),
    )
    assert resp.status_code == 422, resp.text
    assert _count_annotations(session_factory, run_id) == 0

