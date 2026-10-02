"""Naming writes and refusals against real Postgres."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_cli_commands import _seed_completed_run
from tests.integration.test_speaker_enrollment import add_turn, make_completed_run
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import label_states
from voxint.adjudication.slots import claim_run
from voxint.adjudication.transcript import attributed_transcript
from voxint.adjudication.undo import undo_decision
from voxint.cli import main
from voxint.db.models import (
    ActivityEvent,
    AdjudicationDecision,
    AutoEnrollEvidence,
    Decision,
    PipelineRun,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
)


@pytest.fixture(autouse=True)
def cli_db(session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "voxint.cli._engine_or_report",
        lambda: (
            create_engine(session_factory.kw["bind"].url),
            0,
        ),
    )
    monkeypatch.setenv("VOXINT_USER", "synthetic-operator")


def snapshot(session: Session) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    return (
        [
            (r.id, r.decision, r.speaker_id, r.operator)
            for r in session.scalars(select(AdjudicationDecision).order_by(AdjudicationDecision.id))
        ],
        [
            (s.id, s.display_name, s.deleted_at, s.merged_into_id)
            for s in session.scalars(select(Speaker).order_by(Speaker.id))
        ],
    )


def rule(
    session: Session,
    run: uuid.UUID,
    speaker: Speaker,
    decision: Decision = Decision.ASSIGN,
    label: str = "SPEAKER_00",
    evidence: str = "created",
) -> None:
    record_decision(
        session,
        pipeline_run_id=run,
        diarization_label=label,
        speaker_id=speaker.id,
        decision=decision,
        operator="system:auto_enroll" if decision == Decision.AUTO_ENROLL else "seed",
        idempotency_key=(
            f"auto_enroll:{run}:{label}" if decision == Decision.AUTO_ENROLL else str(uuid.uuid4())
        ),
    )

    if decision == Decision.AUTO_ENROLL:
        session.add(
            AutoEnrollEvidence(
                pipeline_run_id=run,
                diarization_label=label,
                decision=evidence,
                reason="synthetic",
                top_speaker_id=speaker.id if evidence == "linked" else None,
                similarity=0.99 if evidence == "linked" else None,
            )
        )


def invoke(run: uuid.UUID, voice: str = "SPEAKER_00", name: str = "Sam") -> int:
    return main(["speakers", "name", str(run), voice, name])


def test_list(session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        speaker = Speaker(display_name="Alex")
        s.add(speaker)
        s.flush()
        rule(s, run, speaker)
        s.commit()
        before = snapshot(s)
    assert main(["speakers", "name", str(run)]) == 0
    out = capsys.readouterr().out
    assert "SPEAKER_00  Alex" in out and "human_assign" in out and "0:01" in out
    assert "SPEAKER_01" in out and "unresolved" in out
    missing = uuid.uuid4()
    assert main(["speakers", "name", str(missing)]) == 2
    assert f"error: no run {missing}" in capsys.readouterr().out
    with session_factory() as s:
        assert snapshot(s) == before


@pytest.mark.parametrize(
    "claimant",
    ["synthetic-other", "synthetic-operator", "expired", "incomplete"],
)
def test_claim_guard(
    session_factory: sessionmaker[Session], claimant: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        s.add(Speaker(display_name="Sam"))
        token = claim_run(s, run, reviewer=claimant, ttl_seconds=600)
        row = s.get(PipelineRun, run)
        assert row is not None
        if claimant == "expired":
            row.review_claim_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        if claimant == "incomplete":
            row.status = "queued"
        s.commit()
        before = snapshot(s)
    assert invoke(run) == (0 if claimant == "expired" else 2)
    out = capsys.readouterr().out
    with session_factory() as s:
        row = s.get(PipelineRun, run)
        assert row is not None and row.review_claim_token == token
        after = snapshot(s)
        assert after[1] == before[1]
        if claimant == "expired":
            assert len(after[0]) == len(before[0]) + 1
            assert after[0][0][1:] == ("assign", after[1][0][0], "synthetic-operator")
        else:
            assert after == before
            assert (
                "not completed" in out
                if claimant == "incomplete"
                else (claimant in out and "until " in out)
            )


def test_assign_noop_and_aba(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        s.add_all([Speaker(display_name="Sam"), Speaker(display_name="Alex")])
        s.commit()
        roster = snapshot(s)[1]
    for name, count in [("Sam", 1), ("Sam", 1), ("Alex", 2), ("Sam", 3)]:
        assert invoke(run, name=name) == 0
        with session_factory() as s:
            rows, speakers = snapshot(s)
            assert speakers == roster and len(rows) == count
            assert all(r[1] == "assign" and r[3] == "synthetic-operator" for r in rows)
            assert label_states(s, run)[0].speaker_name == name
            assert attributed_transcript(s, run, text="corrected")[0].speaker == name


@pytest.mark.parametrize(
    "initial,decision",
    [
        ("Voice 3", Decision.AUTO_ENROLL),
        ("Voice 3", Decision.ASSIGN),
        ("Alex", Decision.AUTO_ENROLL),
        ("Voice 3x", Decision.AUTO_ENROLL),
    ],
)
def test_placeholder_detection(
    session_factory: sessionmaker[Session],
    initial: str,
    decision: Decision,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with session_factory() as s:
        run = make_completed_run(s)
        add_turn(s, run, 0, "SPEAKER_00")
        speaker = Speaker(display_name=initial)
        s.add(speaker)
        s.flush()
        old_id = speaker.id
        rule(s, run, speaker, decision)
        other = make_completed_run(s)
        add_turn(s, other, 0, "SPEAKER_00")
        rule(s, other, speaker, Decision.AUTO_ENROLL, evidence="linked")
        s.commit()
    assert invoke(run) == 0
    out = capsys.readouterr().out
    placeholder = initial == "Voice 3" and decision == Decision.AUTO_ENROLL
    with session_factory() as s:
        rows, speakers = snapshot(s)
        assert len(rows) == 3 and len(speakers) == (1 if placeholder else 2)
        current = label_states(s, run)[0]
        assert current.speaker_name == "Sam"
        assert current.effective_decision is not None
        assert current.effective_decision.operator == "synthetic-operator"
        assert current.effective_decision.decision == "assign"
        old = s.get(Speaker, old_id)
        assert old is not None and old.display_name == ("Sam" if placeholder else initial)
        assert (current.speaker_id == old_id) == placeholder
        if placeholder:
            assert 'renamed "Voice 3" -> "Sam" (also appears in 1 other recording)' in out
            decision_id = current.effective_decision.id
            undo_decision(s, run, decision_id, "synthetic-operator", f"undo:{decision_id}", 600)
            s.commit()
            restored = label_states(s, run)[0]
            assert restored.resolution.value == "auto_enroll"
            assert restored.speaker_name == "Sam"
            assert len(snapshot(s)[0]) == 4 and len(snapshot(s)[1]) == 1
        else:
            assert 'enrolled "Sam" as a new speaker' in out


@pytest.mark.parametrize("mode", ["audio", "merged", "archived", "ambiguous", "unknown"])
def test_refusals(
    session_factory: sessionmaker[Session], mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        target = Speaker(display_name="Alex")
        s.add(target)
        s.flush()
        if mode in ("merged", "archived"):
            owner = Speaker(display_name="Sam")
            if mode == "merged":
                owner.merged_into_id = target.id
                owner.merged_at = datetime.now(UTC)
            else:
                owner.deleted_at = datetime.now(UTC)
            s.add(owner)
        if mode == "ambiguous":
            rule(s, run, target)
            rule(s, run, target, label="SPEAKER_01")
        s.commit()
        before = snapshot(s)
    voice = "Alex" if mode == "ambiguous" else "missing" if mode == "unknown" else "SPEAKER_00"
    assert invoke(run, voice) == 2
    out = capsys.readouterr().out
    expected = {
        "audio": "no speaker audio",
        "merged": "was merged",
        "archived": "is archived",
        "ambiguous": "ambiguous",
        "unknown": "unknown",
    }
    assert expected[mode] in out and "\u2014" not in out
    if mode in ("ambiguous", "unknown"):
        assert "SPEAKER_00" in out and "SPEAKER_01" in out
    with session_factory() as s:
        assert snapshot(s) == before


@pytest.mark.parametrize("voice", ["Alex", "aLeX"])
def test_display_name(session_factory: sessionmaker[Session], voice: str) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        a, b = Speaker(display_name="Alex"), Speaker(display_name="Sam")
        s.add_all([a, b])
        s.flush()
        rule(s, run, a)
        s.commit()
        roster = snapshot(s)[1]
    assert invoke(run, voice) == 0
    with session_factory() as s:
        assert snapshot(s)[1] == roster and len(snapshot(s)[0]) == 2
        assert label_states(s, run)[0].speaker_name == "Sam"


def test_rename_rollback(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        speaker = Speaker(display_name="Voice 3")
        s.add(speaker)
        s.flush()
        rule(s, run, speaker, Decision.AUTO_ENROLL)
        s.commit()
        before = snapshot(s)

    def fail(session: Session, **kwargs: object) -> None:
        assert session.scalar(select(Speaker.display_name)) == "Sam"
        raise RuntimeError("synthetic failure after rename")

    monkeypatch.setattr("voxint.adjudication.naming.record_decision", fail)
    assert invoke(run) == 1
    assert "synthetic failure after rename" in capsys.readouterr().out
    with session_factory() as s:
        assert snapshot(s) == before


@pytest.mark.parametrize("enabled", [True, False])
def test_activity(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    from voxint.db.models import ActivityEvent

    monkeypatch.setenv("CONSOLE_ACTIVITY_ENABLED", str(enabled).lower())
    with session_factory() as s:
        run = _seed_completed_run(s)
        s.add(Speaker(display_name="Sam"))
        s.commit()
        roster = snapshot(s)[1]
    assert invoke(run) == 0
    assert invoke(run) == 0
    with session_factory() as s:
        rows, speakers = snapshot(s)
        assert len(rows) == 1 and speakers == roster
        events = list(s.scalars(select(ActivityEvent)))
        assert len(events) == int(enabled)
        if enabled:
            assert events[0].title == "Sam"
            assert events[0].occurrence_key == f"decision:{rows[0][0]}:identified"


def test_other_recording_count(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    from voxint.db.models import SpeakerAssignment

    with session_factory() as s:
        run = _seed_completed_run(s)
        speaker = Speaker(display_name="Voice 9")
        s.add(speaker)
        s.flush()
        rule(s, run, speaker, Decision.AUTO_ENROLL)
        other = make_completed_run(s)
        rule(s, other, speaker)
        rule(s, other, speaker, label="SPEAKER_01")
        s.add(
            SpeakerAssignment(
                pipeline_run_id=other,
                diarization_label="SPEAKER_00",
                speaker_id=speaker.id,
                method="cosine",
                grounded=True,
            )
        )
        machine_only = make_completed_run(s)
        s.add(
            SpeakerAssignment(
                pipeline_run_id=machine_only,
                diarization_label="SPEAKER_00",
                speaker_id=speaker.id,
                method="cosine",
                grounded=True,
            )
        )
        revoked = make_completed_run(s)
        rule(s, revoked, speaker)
        row = s.scalars(
            select(AdjudicationDecision).where(AdjudicationDecision.pipeline_run_id == revoked)
        ).one()
        undo_decision(s, revoked, row.id, "seed", f"undo:{row.id}", 600)
        s.commit()
        before = snapshot(s)
    assert invoke(run) == 0
    assert "2 other recordings" in capsys.readouterr().out
    with session_factory() as s:
        rows, speakers = snapshot(s)
        assert len(rows) == len(before[0]) + 1
        assert len(speakers) == 1 and speakers[0][1] == "Sam"
        assert label_states(s, run)[0].resolution.value == "human_assign"


@pytest.mark.parametrize("name", ["", "   ", "S" * 121])
def test_invalid_name(
    session_factory: sessionmaker[Session],
    name: str,
) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        s.add(Speaker(display_name="Alex"))
        s.commit()
        before = snapshot(s)
    assert invoke(run, name=name) == 2
    with session_factory() as s:
        assert snapshot(s) == before


def test_reserved_operator(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOXINT_USER", "system:auto_enroll")
    with session_factory() as s:
        run = _seed_completed_run(s)
        s.add(Speaker(display_name="Sam"))
        s.commit()
        before = snapshot(s)
    assert invoke(run) == 2
    with session_factory() as s:
        assert snapshot(s) == before


def test_unclaimed_holds_run_lock(session_factory: sessionmaker[Session]) -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from voxint.adjudication.slots import require_unclaimed

    with session_factory() as s:
        run = _seed_completed_run(s)
        before = snapshot(s)
    with session_factory() as naming_session:
        require_unclaimed(naming_session, run)
        with session_factory() as console_session:
            console_session.execute(text("SET LOCAL lock_timeout = '100ms'"))
            with pytest.raises(OperationalError, match="lock timeout"):
                claim_run(console_session, run, reviewer="synthetic-console", ttl_seconds=600)
            console_session.rollback()
        naming_session.commit()
    with session_factory() as s:
        assert snapshot(s) == before
        row = s.get(PipelineRun, run)
        assert row is not None and row.review_claim_token is None


@pytest.mark.parametrize("linked", [False, True])
def test_identity_creation_provenance(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    linked: bool,
) -> None:
    monkeypatch.setenv("CONSOLE_ACTIVITY_ENABLED", "true")
    with session_factory() as s:
        origin = make_completed_run(s)
        run = make_completed_run(s)
        add_turn(s, run, 0, "SPEAKER_00")
        speaker = Speaker(display_name="Voice 7" if linked else "Voice 3")
        s.add(speaker)
        s.flush()
        old_id = speaker.id
        if linked:
            rule(s, run, speaker, Decision.AUTO_ENROLL, evidence="linked")
        else:
            rule(s, origin, speaker, Decision.AUTO_ENROLL)
            s.add(
                SpeakerAssignment(
                    pipeline_run_id=run,
                    diarization_label="SPEAKER_00",
                    speaker_id=old_id,
                    method="cosine",
                    grounded=True,
                )
            )
        s.commit()
        if not linked:
            assert label_states(s, run)[0].effective_decision is None
            assert label_states(s, run)[0].speaker_id == old_id
    assert invoke(run) == 0
    out = capsys.readouterr().out
    assert "a later ruling in the console supersedes this assignment; the rename stays" in out
    with session_factory() as s:
        rows, speakers = snapshot(s)
        assert len(rows) == 2 and len(speakers) == (2 if linked else 1)
        old = s.get(Speaker, old_id)
        assert old is not None and old.display_name == ("Voice 7" if linked else "Sam")
        current = label_states(s, run)[0]
        assert current.speaker_name == "Sam"
        assert (current.speaker_id == old_id) == (not linked)
        assert current.effective_decision is not None
        assert current.effective_decision.decision == "assign"
        events = list(s.scalars(select(ActivityEvent)))
        assert len(events) == 1 and events[0].title == "Sam"
        assert events[0].occurrence_key == f"decision:{current.effective_decision.id}:identified"


@pytest.mark.parametrize("mode", ["superseded", "ungrounded", "alias", "opposed"])
def test_current_impact_count(session_factory: sessionmaker[Session], mode: str) -> None:
    from voxint.adjudication.naming import _other_runs

    with session_factory() as s:
        run, other = make_completed_run(s), make_completed_run(s)
        speaker, bob = Speaker(display_name="Voice 3"), Speaker(display_name="Bob")
        s.add_all([speaker, bob])
        s.flush()
        if mode == "alias":
            alias = Speaker(
                display_name="Alex", merged_into_id=speaker.id, merged_at=datetime.now(UTC)
            )
            s.add(alias)
            s.flush()
            rule(s, other, alias)
        elif mode == "superseded":
            rule(s, other, speaker)
            s.commit()
            rule(s, other, bob)
        else:
            s.add(
                SpeakerAssignment(
                    pipeline_run_id=other,
                    diarization_label="SPEAKER_00",
                    speaker_id=speaker.id,
                    method="cosine",
                    grounded=mode != "ungrounded",
                )
            )
            if mode == "opposed":
                rule(s, other, bob)
        s.commit()
        before = snapshot(s)
        assert _other_runs(s, run, speaker.id) == int(mode == "alias")
        assert snapshot(s) == before


def test_own_current_name_noop(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as s:
        run = _seed_completed_run(s)
        speaker = Speaker(display_name="Voice 3")
        s.add(speaker)
        s.flush()
        rule(s, run, speaker, Decision.AUTO_ENROLL)
        s.commit()
        before = snapshot(s)
    assert invoke(run, voice="Voice 3", name="Voice 3") == 0
    with session_factory() as s:
        assert snapshot(s) == before
        assert list(s.scalars(select(ActivityEvent))) == []


def test_enroll_rollback(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONSOLE_ACTIVITY_ENABLED", "true")
    with session_factory() as s:
        run = make_completed_run(s)
        add_turn(s, run, 0, "SPEAKER_00")
        s.commit()
        before = snapshot(s)

    def fail(session: Session, **kwargs: object) -> None:
        assert session.scalar(select(Speaker.display_name)) == "Sam"
        assert session.scalar(select(SpeakerEmbedding.id)) is not None
        assert session.scalar(select(AdjudicationDecision.id)) is not None
        raise RuntimeError("synthetic failure after enrollment")

    monkeypatch.setattr("voxint.adjudication.naming.record_speaker_identified", fail)
    assert invoke(run) == 1
    with session_factory() as s:
        assert snapshot(s) == before
        assert list(s.scalars(select(SpeakerEmbedding))) == []
        assert list(s.scalars(select(ActivityEvent))) == []
