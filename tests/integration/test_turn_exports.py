"""Real-database parity for opt-in speaker-turn exports and translations."""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_translation_jobs import record_spanish, seed_run
from tests.integration.test_translation_view_export import _build_client, _stale_edit
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.splits import record_split
from voxint.cli import main
from voxint.db.models import Decision, DiarizationTurn, Speaker, TranscriptSegment


def seed_words(session: Session, *, split: bool = False) -> uuid.UUID:
    """Two named labels and one word-aligned segment, optionally operator-split."""
    run_id = seed_run(session, texts=["Hello there."])
    segment = session.scalars(select(TranscriptSegment).where(
        TranscriptSegment.pipeline_run_id == run_id,
    )).one()
    segment.diarization_label = "SPEAKER_00"
    segment.words = [
        {"word": "Hello", "start": 0, "end": 1},
        {"word": " there.", "start": 1, "end": 2},
    ]
    for i, name in enumerate(("Alex", "Sam")):
        speaker = Speaker(display_name=name)
        session.add(speaker)
        session.flush()
        label = f"SPEAKER_{i:02}"
        session.add(DiarizationTurn(
            pipeline_run_id=run_id, turn_index=i, start_seconds=i, end_seconds=i + 1,
            label=label, skip_reason="too_short",
        ))
        record_decision(session, pipeline_run_id=run_id, diarization_label=label,
                        decision=Decision.ASSIGN, speaker_id=speaker.id,
                        operator="test", idempotency_key=str(uuid.uuid4()))
        if split and i == 1:
            record_split(session, parent=segment, word_index=1, operator="test")
            record_decision(
                session, pipeline_run_id=run_id, diarization_label="SPEAKER_00",
                transcript_segment_id=segment.id, start_word_index=1, end_word_index=2,
                decision=Decision.ASSIGN, speaker_id=speaker.id,
                operator="test", idempotency_key=str(uuid.uuid4()),
            )
    session.commit()
    return run_id


@pytest.mark.parametrize("timestamps", [True, False])
def test_three_surface_parity_and_default(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, timestamps: bool,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session)
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    # Use the fixture's disposable DB, including its worker-specific name.
    monkeypatch.setattr("voxint.cli._engine_or_report", lambda: (
        create_engine(session_factory.kw["bind"].url), 0,
    ))
    bodies: dict[str, bytes] = {}
    for style in ("", "blocks", "turns"):
        params = {"timestamps": str(timestamps).lower()}
        args = ["export", str(run_id), "--format", "md", "-o", str(tmp_path / f"{style}.md")]
        if style:
            params["style"] = style
            args += ["--style", style]
        if not timestamps:
            args += ["--no-timestamps"]
        assert main(args) == 0
        cli_body = (tmp_path / f"{style}.md").read_bytes()
        console = client.get(f"/review/{run_id}/export.md", params=params)
        public = client.get(f"/api/v1/runs/{run_id}/transcript", params={"format": "md", **params},
                            auth=None, headers={"Authorization": "Bearer synthetic-api-key"})
        assert console.status_code == public.status_code == 200
        assert cli_body == console.content == public.content
        bodies[style] = cli_body
    assert bodies[""] == bodies["blocks"]
    bracket = "[00:00:00.000\u201300:00:05.000] " if timestamps else ""
    assert bodies[""] == f"## Alex\n\n> {bracket}Hello there.\n".encode()
    a, b = ("[00:00:00] ", "[00:00:01] ") if timestamps else ("", "")
    assert bodies["turns"] == f"{a}**Alex:** Hello\n\n{b}**Sam:** there.\n".encode()


def test_invalid_style_options(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_db() -> None:
        pytest.fail("invalid CLI options must not touch the database")

    monkeypatch.setattr("voxint.cli._engine_or_report", no_db)
    assert main(["export", str(uuid.uuid4()), "--format", "txt", "--style", "turns"]) == 2
    assert "style applies to the md format only" in capsys.readouterr().out
    with session_factory() as session:
        run_id = seed_words(session)
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    for fmt, style in (("txt", "turns"), ("rttm", "blocks"), ("md", "bogus")):
        response = client.get(f"/api/v1/runs/{run_id}/transcript",
                              params={"format": fmt, "style": style}, auth=None,
                              headers={"Authorization": "Bearer synthetic-api-key"})
        assert response.status_code == 422
        console = client.get(f"/review/{run_id}/export.{fmt}", params={"style": style})
        assert console.status_code == 422
    assert client.get(f"/review/{run_id}/export.md?style=bogus").status_code == 422
    # A missing run is a 404 on every console route, whatever the options say.
    for ext in ("md", "txt", "rttm"):
        missing = client.get(f"/review/{uuid.uuid4()}/export.{ext}", params={"style": "bogus"})
        assert missing.status_code == 404


@pytest.mark.parametrize("split", [False, True])
def test_translated_turns_and_staleness(
    session_factory: sessionmaker[Session], split: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session, split=split)
        record_spanish(session, run_id)
        session.commit()
    client = _build_client(session_factory)
    def no_job_lookup(*args: object) -> None:
        pytest.fail("fresh or stale translation must not query job history")

    monkeypatch.setattr(
        "voxint.api.routers.adjudication_api.active_or_last_translation_job", no_job_lookup,
    )
    response = client.get(f"/review/{run_id}/export.md?lang=es&style=turns")
    assert response.status_code == 200
    expected = (
        "[00:00:00] **Alex:** ES:Hello\n\n[00:00:01] **Sam:** ES:there.\n"
        if split else "[00:00:00] **Alex:** ES:Hello there.\n"
    )
    assert response.text == expected
    # Directly alter evidence to exercise the existing stale hash response.
    with session_factory() as session:
        segment = session.scalars(select(TranscriptSegment).where(
            TranscriptSegment.pipeline_run_id == run_id,
        )).one()
        segment.raw_text = "Changed transcript."
        session.commit()
    stale = client.get(f"/review/{run_id}/export.md?lang=es&style=turns")
    blocks = client.get(f"/review/{run_id}/export.md?lang=es&style=blocks")
    assert stale.status_code == blocks.status_code == 409
    assert stale.json() == blocks.json()
    assert "out of date" in stale.json()["detail"]


@pytest.mark.parametrize("style", ["blocks", "turns"])
def test_translation_count_mismatch_is_same_409(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, style: str,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session)
        record_spanish(session, run_id)
        session.commit()
    client = _build_client(session_factory)
    monkeypatch.setattr("voxint.api.routers.adjudication_api.translation_texts", lambda head: [])
    mismatch = client.get(f"/review/{run_id}/export.md?lang=es&style={style}")
    assert mismatch.status_code == 409
    _stale_edit(session_factory, run_id)
    stale = client.get(f"/review/{run_id}/export.md?lang=es&style={style}")
    assert mismatch.json() == stale.json()
