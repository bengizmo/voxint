"""Label-scope decision undo over HTTP (issue #573).

``POST /review/{run}/labels/{label}/decision`` returns an ``undo`` payload for a
fresh ruling, and ``POST /review/{run}/undo/decide`` reverts it with a
compensating REVOKE. Real Postgres, real app.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import voxint.adjudication.undo as undo_module
from tests.integration.conftest import seed_onboarded
from tests.integration.test_review_api import CREDS, claim_token, seed_run
from voxint.adjudication.resolver import effective_decisions
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import AdjudicationDecision, Decision, Speaker

_CSRF_KEY = "review-api-test-csrf-key"
_GRACE = 300


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
        UNDO_GRACE_SECONDS=_GRACE,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _csrf() -> str:
    return mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)


def _decide(
    client: TestClient,
    run_id: uuid.UUID,
    token: str,
    label: str,
    action: str,
    *,
    speaker_id: uuid.UUID | None = None,
    nonce: str | None = None,
) -> dict[str, object]:
    data = {"token": token, "nonce": nonce or uuid.uuid4().hex, "action": action}
    if speaker_id is not None:
        data["speaker_id"] = str(speaker_id)
    resp = client.post(f"/review/{run_id}/labels/{label}/decision", data=data)
    assert resp.status_code == 200, resp.text
    body: dict[str, object] = resp.json()
    return body


def _undo(
    client: TestClient,
    run_id: uuid.UUID,
    token: str,
    decision_id: str,
    *,
    csrf: str | None = None,
    nonce: str | None = None,
) -> object:
    data = {
        "token": token,
        "decision_id": decision_id,
        "nonce": nonce or f"undo:{decision_id}",
    }
    if csrf is not None:
        data["csrf_token"] = csrf
    return client.post(f"/review/{run_id}/undo/decide", data=data)


def _label(body: dict[str, object], label: str) -> dict[str, object]:
    labels = body["labels"]
    assert isinstance(labels, list)
    return next(entry for entry in labels if entry["label"] == label)


def test_fresh_decision_returns_an_undo_payload(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)

    before = datetime.now(UTC)
    body = _decide(client, run_id, token, "S1", "exclude")

    undo = body["undo"]
    assert isinstance(undo, dict)
    assert undo["kind"] == "decide"
    with session_factory() as session:
        row = session.get(AdjudicationDecision, uuid.UUID(undo["decisionId"]))
        assert row is not None
        assert (row.diarization_label, row.decision) == ("S1", Decision.EXCLUDE.value)
    expires = datetime.fromisoformat(undo["expiresAt"])
    assert before + timedelta(seconds=_GRACE - 5) <= expires
    assert expires <= datetime.now(UTC) + timedelta(seconds=_GRACE)


def test_replayed_decision_carries_no_undo_payload(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    nonce = uuid.uuid4().hex

    first = _decide(client, run_id, token, "S1", "exclude", nonce=nonce)
    replay = _decide(client, run_id, token, "S1", "exclude", nonce=nonce)

    assert "undo" in first
    assert "undo" not in replay


def test_undo_restores_the_label_and_returns_label_states(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()
    token = claim_token(client, run_id)
    before = _label(_decide(client, run_id, token, "S1", "unknown"), "S1")["resolution"]

    assigned = _decide(client, run_id, token, "S1", "assign", speaker_id=known_id)
    assert _label(assigned, "S1")["speakerId"] == str(known_id)
    undo = assigned["undo"]
    assert isinstance(undo, dict)

    resp = _undo(client, run_id, token, undo["decisionId"], csrf=_csrf())

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    restored = _label(body, "S1")
    assert restored["resolution"] == before
    assert restored["speakerId"] is None
    assert "undo" not in body
    assert {"labels", "segments", "progress"} <= set(body)
    with session_factory() as session:
        assert effective_decisions(session, run_id)["S1"].decision == Decision.UNKNOWN.value


def test_undo_retry_is_idempotent(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    undo = _decide(client, run_id, token, "S1", "exclude")["undo"]
    assert isinstance(undo, dict)

    first = _undo(client, run_id, token, undo["decisionId"], csrf=_csrf())
    retry = _undo(client, run_id, token, undo["decisionId"], csrf=_csrf())

    assert first.status_code == retry.status_code == 200  # type: ignore[attr-defined]
    with session_factory() as session:
        count = session.execute(
            select(func.count())
            .select_from(AdjudicationDecision)
            .where(AdjudicationDecision.voids_decision_id == uuid.UUID(undo["decisionId"]))
        ).scalar_one()
    assert count == 1


def test_undo_after_a_later_ruling_is_a_409_and_changes_nothing(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    undo = _decide(client, run_id, token, "S1", "exclude")["undo"]
    assert isinstance(undo, dict)
    _decide(client, run_id, token, "S1", "unknown")

    resp = _undo(client, run_id, token, undo["decisionId"], csrf=_csrf())

    assert resp.status_code == 409  # type: ignore[attr-defined]
    # A drift 409 is a state conflict, never a lost claim: the island keeps
    # its claim and shows "too late to undo".
    assert "x-voxint-conflict" not in resp.headers  # type: ignore[attr-defined]
    with session_factory() as session:
        assert effective_decisions(session, run_id)["S1"].decision == Decision.UNKNOWN.value


def test_undo_past_the_grace_window_is_a_409(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    undo = _decide(client, run_id, token, "S1", "exclude")["undo"]
    assert isinstance(undo, dict)

    # The ledger is append-only (a trigger blocks UPDATE), so move the undo
    # module's clock past the window instead of backdating the row.
    class _Later(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> "_Later":
            moved = datetime.now(tz) + timedelta(seconds=_GRACE + 1)
            return cls.fromtimestamp(moved.timestamp(), tz)

    monkeypatch.setattr(undo_module, "datetime", _Later)

    resp = _undo(client, run_id, token, undo["decisionId"], csrf=_csrf())

    assert resp.status_code == 409  # type: ignore[attr-defined]
    assert "grace window" in resp.json()["detail"]  # type: ignore[attr-defined]
    with session_factory() as session:
        assert effective_decisions(session, run_id)["S1"].decision == Decision.EXCLUDE.value


def test_undo_requires_the_claim_csrf_token(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    undo = _decide(client, run_id, token, "S1", "exclude")["undo"]
    assert isinstance(undo, dict)

    missing = _undo(client, run_id, token, undo["decisionId"])
    forged = _undo(client, run_id, token, undo["decisionId"], csrf="forged")

    assert missing.status_code == 422  # type: ignore[attr-defined]
    assert forged.status_code == 403  # type: ignore[attr-defined]
    with session_factory() as session:
        assert effective_decisions(session, run_id)["S1"].decision == Decision.EXCLUDE.value


def test_undo_with_a_stale_claim_is_a_marked_409(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)
    undo = _decide(client, run_id, token, "S1", "exclude")["undo"]
    assert isinstance(undo, dict)

    resp = _undo(client, run_id, str(uuid.uuid4()), undo["decisionId"], csrf=_csrf())

    assert resp.status_code == 409  # type: ignore[attr-defined]
    assert resp.headers["x-voxint-conflict"] == "claim"  # type: ignore[attr-defined]
    with session_factory() as session:
        assert effective_decisions(session, run_id)["S1"].decision == Decision.EXCLUDE.value


def test_undo_of_a_merge_child_is_a_400(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()
    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
    ).json()

    merged = client.post(
        f"/review/{run_id}/merge",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "labels": preview["labels"],
            "speaker_id": str(known_id),
            "expected": json.dumps(preview["expected"]),
        },
    )
    assert merged.status_code == 200, merged.text
    with session_factory() as session:
        child_id = effective_decisions(session, run_id)["S1"].id

    resp = _undo(client, run_id, token, str(child_id), csrf=_csrf())

    assert resp.status_code == 400  # type: ignore[attr-defined]
    assert "merge undo endpoint" in resp.json()["detail"]  # type: ignore[attr-defined]


def test_undo_of_an_unknown_decision_is_a_400(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)

    resp = _undo(client, run_id, token, str(uuid.uuid4()), csrf=_csrf())

    assert resp.status_code == 400  # type: ignore[attr-defined]
