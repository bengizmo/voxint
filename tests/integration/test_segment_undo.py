"""Segment- and word-range-scope relabel undo (issue #573).

A segment-scope ruling cannot carry a REVOKE (the schema keeps REVOKE rows
label-shaped and the segment resolvers reduce newest-wins), so its undo appends
a compensating ruling in the same scope. These tests pin that the compensation
restores exactly what the scope followed before, in the service and over HTTP.
Real Postgres, real app.
"""

import uuid
from datetime import datetime, timedelta, tzinfo
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import voxint.adjudication.undo as undo_module
from tests.integration.conftest import seed_onboarded
from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.adjudication.resolver import (
    effective_decisions,
    segment_states,
    word_range_states,
)
from voxint.adjudication.undo import (
    UndoDriftError,
    UndoError,
    UndoExpiredError,
    undo_segment_decision,
)
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentSplitBoundary,
    Speaker,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "segment-undo-test-csrf-key"
_GRACE = 300

# A splittable four-word segment (tokens reconcatenate to raw_text exactly).
_WORDS = [
    {"start": 0.0, "end": 0.4, "word": "Hello"},
    {"start": 0.5, "end": 0.9, "word": " there"},
    {"start": 1.0, "end": 1.4, "word": " big"},
    {"start": 1.5, "end": 1.9, "word": " world"},
]
_RAW = "Hello there big world"


class _Seeded:
    def __init__(
        self, run_id: uuid.UUID, segment_id: uuid.UUID, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        self.run_id = run_id
        self.segment_id = segment_id
        self.alice = alice
        self.bob = bob


def _cut(run_id: uuid.UUID, segment_id: uuid.UUID, word_index: int) -> SegmentSplitBoundary:
    return SegmentSplitBoundary(
        pipeline_run_id=run_id,
        parent_segment_id=segment_id,
        word_index=word_index,
        operator="ben",
    )


def _seed(session_factory: sessionmaker[Session], *, split_at: int | None = None) -> _Seeded:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        segment = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=2.0,
            raw_text=_RAW,
            diarization_label="S0",
            words=_WORDS,
        )
        tag = uuid.uuid4().hex[:8]
        alice = Speaker(display_name=f"Alice {tag}")
        bob = Speaker(display_name=f"Bob {tag}")
        session.add_all([segment, alice, bob])
        session.flush()
        if split_at is not None:
            session.add(_cut(run.id, segment.id, split_at))
        session.commit()
        return _Seeded(run.id, segment.id, alice.id, bob.id)


def _rule(
    session: Session,
    seeded: _Seeded,
    decision: Decision,
    *,
    speaker_id: uuid.UUID | None = None,
    word_range: tuple[int, int] | None = None,
    key: str | None = None,
) -> AdjudicationDecision:
    row = record_decision(
        session,
        pipeline_run_id=seeded.run_id,
        diarization_label="S0",
        decision=decision,
        operator="ben",
        idempotency_key=key or uuid.uuid4().hex,
        speaker_id=speaker_id,
        transcript_segment_id=seeded.segment_id,
        start_word_index=word_range[0] if word_range else None,
        end_word_index=word_range[1] if word_range else None,
    )
    # One ruling per transaction: created_at is the transaction start, and the
    # resolvers order the scope by it.
    session.commit()
    return row


def _undo(
    session: Session,
    seeded: _Seeded,
    decision_id: uuid.UUID,
    key: str = "undo-segment",
    grace_seconds: float = _GRACE,
) -> dict[str, object]:
    result = undo_segment_decision(
        session,
        run_id=seeded.run_id,
        decision_id=decision_id,
        operator="ben",
        idempotency_key=key,
        grace_seconds=grace_seconds,
    )
    session.commit()
    return result


def _compensation(session: Session, result: dict[str, object]) -> AdjudicationDecision:
    row = session.get(AdjudicationDecision, result["compensating_decision_id"])
    assert row is not None
    return row


def _segment_speaker(session: Session, seeded: _Seeded) -> uuid.UUID | None:
    override = segment_states(session, seeded.run_id).get(seeded.segment_id)
    return override.speaker_id if override else None


def _range_speaker(
    session: Session, seeded: _Seeded, word_range: tuple[int, int]
) -> uuid.UUID | None:
    override = word_range_states(session, seeded.run_id).get(
        (seeded.segment_id, *word_range)
    )
    return override.speaker_id if override else None


# --- service ----------------------------------------------------------------


def test_undo_of_a_first_assign_returns_the_segment_to_its_label(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        assert _segment_speaker(session, seeded) == seeded.alice

        result = _undo(session, seeded, original.id)

        assert result["is_replay"] is False
        assert result["undone_decision_id"] == original.id
        compensation = _compensation(session, result)
        assert compensation.decision == Decision.INHERIT.value
        assert compensation.speaker_id is None
        assert compensation.transcript_segment_id == seeded.segment_id
        assert compensation.diarization_label == "S0"
        assert _segment_speaker(session, seeded) is None
        # Append-only: the undone ruling stays in the ledger, unvoided.
        assert session.get(AdjudicationDecision, original.id) is not None


def test_undo_restores_the_previous_segment_speaker(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)
        assert _segment_speaker(session, seeded) == seeded.bob

        result = _undo(session, seeded, original.id)

        compensation = _compensation(session, result)
        assert compensation.decision == Decision.ASSIGN.value
        assert compensation.speaker_id == seeded.alice
        assert _segment_speaker(session, seeded) == seeded.alice


def test_undo_of_a_reset_restores_the_override_it_cleared(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        reset = _rule(session, seeded, Decision.INHERIT)
        assert _segment_speaker(session, seeded) is None

        _undo(session, seeded, reset.id)

        assert _segment_speaker(session, seeded) == seeded.alice


def test_undo_after_an_earlier_reset_leaves_the_segment_following_its_label(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        _rule(session, seeded, Decision.INHERIT)
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)

        result = _undo(session, seeded, original.id)

        assert _compensation(session, result).decision == Decision.INHERIT.value
        assert _segment_speaker(session, seeded) is None


def test_undo_never_touches_label_scope(session_factory: sessionmaker[Session]) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        label_rule = record_decision(
            session,
            pipeline_run_id=seeded.run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="label-rule",
            speaker_id=seeded.bob,
        )
        session.commit()
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)

        _undo(session, seeded, original.id)

        assert effective_decisions(session, seeded.run_id)["S0"].id == label_rule.id


def test_undo_of_a_word_range_ruling_stays_in_that_range(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory, split_at=2)
    with session_factory() as session:
        whole = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)
        original = _rule(
            session, seeded, Decision.ASSIGN, speaker_id=seeded.alice, word_range=(2, 4)
        )
        assert _range_speaker(session, seeded, (2, 4)) == seeded.alice

        result = _undo(session, seeded, original.id)

        compensation = _compensation(session, result)
        assert compensation.decision == Decision.INHERIT.value
        assert (compensation.start_word_index, compensation.end_word_index) == (2, 4)
        assert _range_speaker(session, seeded, (2, 4)) is None
        # The whole-segment override is a different scope and survives.
        override = segment_states(session, seeded.run_id)[seeded.segment_id]
        assert override.decision.id == whole.id


def test_a_newer_ruling_in_another_scope_is_not_drift(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory, split_at=2)
    with session_factory() as session:
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob, word_range=(0, 2))
        original = _rule(
            session, seeded, Decision.ASSIGN, speaker_id=seeded.alice, word_range=(2, 4)
        )
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice, word_range=(0, 2))

        _undo(session, seeded, original.id)

        assert _range_speaker(session, seeded, (2, 4)) is None
        assert _range_speaker(session, seeded, (0, 2)) == seeded.alice


def test_undo_rejects_a_later_ruling_in_the_same_scope(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)

        with pytest.raises(UndoDriftError):
            _undo(session, seeded, original.id)
        session.rollback()
        assert _segment_speaker(session, seeded) == seeded.bob


def test_undo_rejects_a_range_that_was_split_again(
    session_factory: sessionmaker[Session],
) -> None:
    # The split route refuses a second distinct cut today; this pins the
    # service's own guard in case that ever changes.
    seeded = _seed(session_factory, split_at=2)
    with session_factory() as session:
        original = _rule(
            session, seeded, Decision.ASSIGN, speaker_id=seeded.alice, word_range=(0, 2)
        )
        session.add(_cut(seeded.run_id, seeded.segment_id, 1))
        session.commit()

        with pytest.raises(UndoDriftError, match="split again"):
            _undo(session, seeded, original.id)


def test_undo_enforces_the_grace_window(session_factory: sessionmaker[Session]) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)

        with pytest.raises(UndoExpiredError):
            _undo(session, seeded, original.id, grace_seconds=0)
        session.rollback()
        assert _segment_speaker(session, seeded) == seeded.alice


def test_undo_replays_its_key_even_after_the_window(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        first = _undo(session, seeded, original.id, key="undo-once")

        replay = _undo(session, seeded, original.id, key="undo-once", grace_seconds=0)

        assert replay["is_replay"] is True
        assert replay["compensating_decision_id"] == first["compensating_decision_id"]
        rows = session.execute(
            select(func.count())
            .select_from(AdjudicationDecision)
            .where(AdjudicationDecision.transcript_segment_id == seeded.segment_id)
        ).scalar_one()
        assert rows == 2


def test_undo_refuses_a_key_already_used_elsewhere(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory, split_at=2)
    with session_factory() as session:
        first = _rule(
            session, seeded, Decision.ASSIGN, speaker_id=seeded.alice, word_range=(0, 2)
        )
        second = _rule(
            session, seeded, Decision.ASSIGN, speaker_id=seeded.alice, word_range=(2, 4)
        )
        _undo(session, seeded, first.id, key="shared-key")

        with pytest.raises(ConflictingReplayError):
            _undo(session, seeded, second.id, key="shared-key")
        session.rollback()
        with pytest.raises(ConflictingReplayError):
            _undo(session, seeded, second.id, key=second.idempotency_key)


def test_undo_refuses_a_key_reused_for_another_ruling_in_the_same_scope(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    with session_factory() as session:
        first = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        _undo(session, seeded, first.id, key="same-scope-key")
        second = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)

        with pytest.raises(ConflictingReplayError):
            _undo(session, seeded, second.id, key="same-scope-key")
        session.rollback()
        assert _segment_speaker(session, seeded) == seeded.bob


def test_a_replay_after_a_later_ruling_reports_the_undo_it_already_did(
    session_factory: sessionmaker[Session],
) -> None:
    # A retried request whose first attempt committed is a replay, not a new
    # undo: it must not touch the later ruling.
    seeded = _seed(session_factory)
    with session_factory() as session:
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        first = _undo(session, seeded, original.id, key="retried")
        _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.bob)

        replay = _undo(session, seeded, original.id, key="retried")

        assert replay["is_replay"] is True
        assert replay["compensating_decision_id"] == first["compensating_decision_id"]
        assert _segment_speaker(session, seeded) == seeded.bob


def test_undo_refuses_rows_that_are_not_segment_rulings(
    session_factory: sessionmaker[Session],
) -> None:
    seeded = _seed(session_factory)
    other = _seed(session_factory)
    with session_factory() as session:
        label_rule = record_decision(
            session,
            pipeline_run_id=seeded.run_id,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="label-exclude",
        )
        session.commit()
        foreign = _rule(session, other, Decision.ASSIGN, speaker_id=other.alice)

        with pytest.raises(UndoError, match="no adjudication decision"):
            _undo(session, seeded, uuid.uuid4())
        with pytest.raises(UndoError, match="segment-scope ruling"):
            _undo(session, seeded, label_rule.id)
        with pytest.raises(UndoError, match="segment-scope ruling"):
            _undo(session, seeded, foreign.id)


def test_undo_refuses_a_ruling_whose_segment_was_replaced(
    session_factory: sessionmaker[Session],
) -> None:
    # A restart voids then detaches segment-scope rulings (#507); the segment
    # they applied to is gone, so there is nothing to restore.
    seeded = _seed(session_factory)
    with session_factory() as session:
        original = _rule(session, seeded, Decision.ASSIGN, speaker_id=seeded.alice)
        record_decision(
            session,
            pipeline_run_id=seeded.run_id,
            diarization_label="S0",
            decision=Decision.REVOKE,
            operator="system:restart",
            idempotency_key="restart-void",
            voids_decision_id=original.id,
        )
        segment = session.get(TranscriptSegment, seeded.segment_id)
        session.delete(segment)
        session.commit()
        session.refresh(original)
        assert original.detached_at is not None

        with pytest.raises(UndoDriftError, match="no longer exists"):
            _undo(session, seeded, original.id)


# --- HTTP -------------------------------------------------------------------


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        review_claim_ttl_seconds=600,
        csrf_secret=_CSRF_KEY,
        UNDO_GRACE_SECONDS=_GRACE,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _claim(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return resp.headers["location"].split("token=")[1].split("&")[0]


def _relabel(
    client: TestClient,
    seeded: _Seeded,
    token: str,
    speaker_id: uuid.UUID | None,
    *,
    word_range: tuple[int, int] | None = None,
    nonce: str | None = None,
) -> dict[str, object]:
    data = {
        "token": token,
        "nonce": nonce or uuid.uuid4().hex,
        "action": "inherit" if speaker_id is None else "assign",
    }
    if speaker_id is not None:
        data["speaker_id"] = str(speaker_id)
    if word_range is not None:
        data["start_word_index"] = str(word_range[0])
        data["end_word_index"] = str(word_range[1])
    resp = client.post(
        f"/review/{seeded.run_id}/segments/{seeded.segment_id}/relabel",
        data=data,
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, object] = resp.json()
    return body


def _post_undo(
    client: TestClient,
    seeded: _Seeded,
    token: str,
    decision_id: str,
    *,
    csrf: str | None = "valid",
) -> object:
    data = {"token": token, "decision_id": decision_id, "nonce": f"undo:{decision_id}"}
    if csrf is not None:
        data["csrf_token"] = (
            mint_csrf_token(_CSRF_KEY, CSRF_CLAIM) if csrf == "valid" else csrf
        )
    return client.post(f"/review/{seeded.run_id}/undo/relabel", data=data)


def _name(session_factory: sessionmaker[Session], speaker_id: uuid.UUID) -> str:
    with session_factory() as session:
        speaker = session.get(Speaker, speaker_id)
        assert speaker is not None
        return speaker.display_name


def _segment_speakers(body: dict[str, object]) -> list[object]:
    segments = body["segments"]
    assert isinstance(segments, list)
    return [segment["speaker"] for segment in segments]


def test_fresh_relabel_returns_an_undo_payload_and_a_replay_does_not(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    nonce = uuid.uuid4().hex

    fresh = _relabel(client, seeded, token, seeded.alice, nonce=nonce)
    replay = _relabel(client, seeded, token, seeded.alice, nonce=nonce)

    undo = fresh["undo"]
    assert isinstance(undo, dict)
    assert undo["kind"] == "relabel"
    with session_factory() as session:
        row = session.get(AdjudicationDecision, uuid.UUID(undo["decisionId"]))
        assert row is not None
        assert row.transcript_segment_id == seeded.segment_id
        expires = datetime.fromisoformat(undo["expiresAt"])
        assert expires == row.created_at + timedelta(seconds=_GRACE)
    assert "undo" not in replay


def test_undo_over_http_restores_the_segment_speaker(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    before = _segment_speakers(_relabel(client, seeded, token, seeded.alice))
    changed = _relabel(client, seeded, token, seeded.bob)
    assert _segment_speakers(changed) == [_name(session_factory, seeded.bob)]
    undo = changed["undo"]
    assert isinstance(undo, dict)

    resp = _post_undo(client, seeded, token, undo["decisionId"])

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert {"labels", "segments", "progress"} <= set(body)
    assert "undo" not in body
    assert _segment_speakers(body) == before == [_name(session_factory, seeded.alice)]
    retry = _post_undo(client, seeded, token, undo["decisionId"])
    assert retry.status_code == 200  # type: ignore[attr-defined]
    assert _segment_speakers(retry.json()) == before  # type: ignore[attr-defined]


def test_undo_of_a_split_child_over_http(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    split = client.post(
        f"/review/{seeded.run_id}/segments/{seeded.segment_id}/split",
        data={"token": token, "word_index": 2},
    )
    assert split.status_code == 200, split.text
    changed = _relabel(client, seeded, token, seeded.alice, word_range=(2, 4))
    assert _segment_speakers(changed) == ["S0", _name(session_factory, seeded.alice)]
    undo = changed["undo"]
    assert isinstance(undo, dict)

    resp = _post_undo(client, seeded, token, undo["decisionId"])

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    assert _segment_speakers(resp.json()) == ["S0", "S0"]  # type: ignore[attr-defined]


def test_undo_after_another_change_is_an_unmarked_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    undo = _relabel(client, seeded, token, seeded.alice)["undo"]
    assert isinstance(undo, dict)
    _relabel(client, seeded, token, seeded.bob)

    resp = _post_undo(client, seeded, token, undo["decisionId"])

    assert resp.status_code == 409  # type: ignore[attr-defined]
    assert "x-voxint-conflict" not in resp.headers  # type: ignore[attr-defined]


def test_undo_past_the_window_is_a_409(
    client: TestClient,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    undo = _relabel(client, seeded, token, seeded.alice)["undo"]
    assert isinstance(undo, dict)

    class _Later(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> "_Later":
            moved = datetime.now(tz) + timedelta(seconds=_GRACE + 1)
            return cls.fromtimestamp(moved.timestamp(), tz)

    monkeypatch.setattr(undo_module, "datetime", _Later)

    resp = _post_undo(client, seeded, token, undo["decisionId"])

    assert resp.status_code == 409  # type: ignore[attr-defined]
    assert "grace window" in resp.json()["detail"]  # type: ignore[attr-defined]


def test_undo_gates_on_csrf_and_claim(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    undo = _relabel(client, seeded, token, seeded.alice)["undo"]
    assert isinstance(undo, dict)

    missing = _post_undo(client, seeded, token, undo["decisionId"], csrf=None)
    forged = _post_undo(client, seeded, token, undo["decisionId"], csrf="forged")
    stale = _post_undo(client, seeded, str(uuid.uuid4()), undo["decisionId"])

    assert missing.status_code == 422  # type: ignore[attr-defined]
    assert forged.status_code == 403  # type: ignore[attr-defined]
    assert stale.status_code == 409  # type: ignore[attr-defined]
    assert stale.headers["x-voxint-conflict"] == "claim"  # type: ignore[attr-defined]
    with session_factory() as session:
        assert _segment_speaker(session, seeded) == seeded.alice


def test_undo_of_a_label_ruling_through_the_relabel_route_is_a_400(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seeded = _seed(session_factory)
    token = _claim(client, seeded.run_id)
    with session_factory() as session:
        label_rule = record_decision(
            session,
            pipeline_run_id=seeded.run_id,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key=uuid.uuid4().hex,
        )
        session.commit()
        label_rule_id = str(label_rule.id)

    resp = _post_undo(client, seeded, token, label_rule_id)

    assert resp.status_code == 400  # type: ignore[attr-defined]
