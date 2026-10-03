"""Real-database parity for default speaker-turn exports and translations."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_translation_jobs import record_spanish, seed_run
from tests.integration.test_translation_view_export import _build_client, _stale_edit
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.splits import record_split
from voxint.api.presentation import friendly_media_label
from voxint.cli import main
from voxint.db.models import (
    Decision,
    DiarizationTurn,
    PipelineRun,
    SegmentReviewState,
    Speaker,
    TranscriptSegment,
)


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
        run = session.get(PipelineRun, run_id)
        assert run is not None
        # The production title rule, not an ad hoc basename.
        title = friendly_media_label(None, run.media_item.source_path)
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
    assert bodies[""] == bodies["turns"]
    bracket = "[00:00:00.000\u201300:00:05.000] " if timestamps else ""
    assert bodies["blocks"] == f"## Alex\n\n> {bracket}Hello there.\n".encode()
    a, b = ("[00:00:00] ", "[00:00:01] ") if timestamps else ("", "")
    assert bodies["turns"] == f"# {title}\n\n{a}**Alex:** Hello\n\n{b}**Sam:** there.\n".encode()


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


@pytest.mark.parametrize("timestamps", [True, False])
@pytest.mark.parametrize("split", [False, True])
def test_translated_turns_and_staleness(
    session_factory: sessionmaker[Session], split: bool, monkeypatch: pytest.MonkeyPatch,
    timestamps: bool,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session, split=split)
        record_spanish(session, run_id)
        session.commit()
        run = session.get(PipelineRun, run_id)
        assert run is not None
        # The production title rule, not an ad hoc basename.
        title = friendly_media_label(None, run.media_item.source_path)
    client = _build_client(session_factory)
    def no_job_lookup(*args: object) -> None:
        pytest.fail("fresh or stale translation must not query job history")

    monkeypatch.setattr(
        "voxint.api.routers.adjudication_api.active_or_last_translation_job", no_job_lookup,
    )
    params = {"lang": "es", "timestamps": str(timestamps).lower()}
    response = client.get(f"/review/{run_id}/export.md", params={**params, "style": "turns"})
    default = client.get(f"/review/{run_id}/export.md", params=params)
    assert default.status_code == 200
    assert default.content == response.content
    assert response.status_code == 200
    expected = (
        "[00:00:00] **Alex:** ES:Hello\n\n[00:00:01] **Sam:** ES:there.\n"
        if split else "[00:00:00] **Alex:** ES:Hello there.\n"
    )
    if not timestamps:
        expected = expected.replace("[00:00:00] ", "").replace("[00:00:01] ", "")
    assert response.text == f"# {title}\n\n{expected}"
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


@pytest.mark.parametrize("snapshot, expected_title", [
    ({"title": "  Synthetic title  "}, "Synthetic title"),
    ({"title": "Report ###"}, "Report \\#\\#\\#"),
    ({"title": "   "}, "Synthetic recording.wav"),
    ({"title": 42}, "Synthetic recording.wav"),
])
def test_turn_header_title_selection(
    session_factory: sessionmaker[Session], snapshot: dict[str, object], expected_title: str,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.sidecar = snapshot
        run.media_item.source_path = "incoming/Synthetic%20recording.wav"
        session.commit()
    client = _build_client(session_factory)
    response = client.get(f"/review/{run_id}/export.md", params={"timestamps": "false"})
    assert response.status_code == 200
    assert response.content == (
        f"# {expected_title}\n\n**Alex:** Hello\n\n**Sam:** there.\n"
    ).encode()


@pytest.mark.parametrize("timestamps", [True, False])
def test_fillers_three_surface_golden_and_default(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timestamps: bool,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session)
        segment = session.scalars(
            select(TranscriptSegment).where(
                TranscriptSegment.pipeline_run_id == run_id,
            )
        ).one()
        segment.raw_text = "Hello um, there."
        segment.words = [
            {"word": "Hello", "start": 0, "end": 0.4},
            {"word": " um,", "start": 0.4, "end": 0.8},
            {"word": " there.", "start": 1, "end": 2},
        ]
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.sidecar = {"title": "Synthetic filler example"}
        session.commit()
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    monkeypatch.setattr(
        "voxint.cli._engine_or_report",
        lambda: (
            create_engine(session_factory.kw["bind"].url),
            0,
        ),
    )
    for fillers in (None, "keep", "drop"):
        path = tmp_path / f"{fillers}.md"
        args = ["export", str(run_id), "--format", "md", "-o", str(path)]
        params = {"timestamps": str(timestamps).lower()}
        if fillers:
            params["fillers"] = fillers
        if fillers == "drop":
            args.append("--drop-fillers")
        if not timestamps:
            args.append("--no-timestamps")
        assert main(args) == 0
        console = client.get(f"/review/{run_id}/export.md", params=params)
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": "md", **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert console.status_code == public.status_code == 200
        a, b = ("[00:00:00] ", "[00:00:01] ") if timestamps else ("", "")
        suffix = "" if fillers == "drop" else " um,"
        golden = (
            f"# Synthetic filler example\n\n{a}**Alex:** Hello{suffix}\n\n{b}**Sam:** there.\n"
        ).encode()
        assert path.read_bytes() == console.content == public.content == golden
    with session_factory() as session:
        segment = session.scalars(
            select(TranscriptSegment).where(
                TranscriptSegment.pipeline_run_id == run_id,
            )
        ).one()
        assert segment.raw_text == "Hello um, there."
        assert segment.words is not None and segment.words[1]["word"] == " um,"


def test_fillers_refusal_matrix(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_db() -> None:
        pytest.fail("invalid CLI filler options must not touch the database")

    monkeypatch.setattr("voxint.cli._engine_or_report", no_db)
    for args in (
        ["--format", "txt"],
        ["--format", "rttm"],
        ["--format", "md", "--style", "blocks"],
    ):
        assert main(["export", str(uuid.uuid4()), "--drop-fillers", *args]) == 2
        assert capsys.readouterr().out == "error: fillers applies to the md turns style only\n"
    with session_factory() as session:
        run_id = seed_words(session)
        record_spanish(session, run_id)
        session.commit()
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    cases = [
        (fmt, {"fillers": "drop"}, "fillers applies to the md turns style only")
        for fmt in ("txt", "rttm", "srt", "vtt", "json")
    ] + [
        (
            "md",
            {"fillers": "drop", "style": "blocks"},
            "fillers applies to the md turns style only",
        ),
        ("md", {"fillers": "bogus"}, "unknown fillers value 'bogus'; valid: keep, drop"),
    ]
    # Only the console routes take a translation; /api/v1 never had lang.
    conflict = client.get(f"/review/{run_id}/export.md", params={"fillers": "drop", "lang": "es"})
    assert conflict.status_code == 422
    assert conflict.json()["detail"] == "fillers cannot be combined with a translation"
    for fmt, params, detail in cases:
        console = client.get(f"/review/{run_id}/export.{fmt}", params=params)
        assert console.status_code == 422
        assert console.json()["detail"] == detail
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": fmt, **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert public.status_code == 422
        assert public.json()["error"]["message"] == detail
        missing_console = client.get(f"/review/{uuid.uuid4()}/export.{fmt}", params=params)
        assert missing_console.status_code == 404
        missing_public = client.get(
            f"/api/v1/runs/{uuid.uuid4()}/transcript",
            params={"format": fmt, **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert missing_public.status_code == 422
        assert missing_public.json()["error"]["message"] == detail


_REPEAT_WORDS = [
    ("Go", 0.0), (" to", 0.1), (" the", 0.2), (" the", 0.3), (" store.", 0.4),
    (" We", 0.5), (" um", 0.6), (" we", 0.7), (" left.", 0.8), (" There.", 1.0),
]


@pytest.mark.parametrize("timestamps", [True, False])
def test_repeats_three_surface_golden_and_default(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timestamps: bool,
) -> None:
    with session_factory() as session:
        run_id = seed_words(session)
        segment = session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
        ).one()
        segment.raw_text = "".join(word for word, _ in _REPEAT_WORDS)
        segment.words = [
            {"word": word, "start": start, "end": start + 0.1} for word, start in _REPEAT_WORDS
        ]
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.sidecar = {"title": "Synthetic repeats example"}
        session.commit()
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    monkeypatch.setattr(
        "voxint.cli._engine_or_report",
        lambda: (create_engine(session_factory.kw["bind"].url), 0),
    )
    alex = {
        (None, None): "Go to the the store. We um we left.",
        (None, "keep"): "Go to the the store. We um we left.",
        (None, ""): "Go to the the store. We um we left.",
        ("drop", None): "Go to the the store. We we left.",
        (None, "drop"): "Go to the store. We um we left.",
        # Filler removal first turns "We um we" into a sentence-start repeat.
        ("drop", "drop"): "Go to the store. We left.",
    }
    a, b = ("[00:00:00] ", "[00:00:01] ") if timestamps else ("", "")
    for (fillers, repeats), text in alex.items():
        path = tmp_path / f"{fillers}-{repeats}.md"
        args = ["export", str(run_id), "--format", "md", "-o", str(path)]
        params = {"timestamps": str(timestamps).lower()}
        if fillers:
            params["fillers"] = fillers
            args.append("--drop-fillers")
        if repeats is not None:
            params["repeats"] = repeats
            if repeats == "drop":
                args.append("--drop-repeats")
        if not timestamps:
            args.append("--no-timestamps")
        assert main(args) == 0
        console = client.get(f"/review/{run_id}/export.md", params=params)
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": "md", **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert console.status_code == public.status_code == 200
        golden = f"# Synthetic repeats example\n\n{a}**Alex:** {text}\n\n{b}**Sam:** There.\n"
        assert path.read_bytes() == console.content == public.content == golden.encode()
    with session_factory() as session:
        segment = session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
        ).one()
        assert segment.raw_text == "Go to the the store. We um we left. There."
        assert segment.words == [
            {"word": word, "start": start, "end": start + 0.1} for word, start in _REPEAT_WORDS
        ]


def test_repeats_refusal_matrix(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_db() -> None:
        pytest.fail("invalid CLI repeats options must not touch the database")

    monkeypatch.setattr("voxint.cli._engine_or_report", no_db)
    for args in (
        ["--format", "txt"],
        ["--format", "srt"],
        ["--format", "vtt"],
        ["--format", "json"],
        ["--format", "rttm"],
        ["--format", "md", "--style", "blocks"],
    ):
        assert main(["export", str(uuid.uuid4()), "--drop-repeats", *args]) == 2
        assert capsys.readouterr().out == "error: repeats applies to the md turns style only\n"
    assert main(
        ["export", str(uuid.uuid4()), "--drop-fillers", "--drop-repeats", "--format", "txt"]
    ) == 2
    assert capsys.readouterr().out == "error: fillers applies to the md turns style only\n"
    with session_factory() as session:
        run_id = seed_words(session)
        record_spanish(session, run_id)
        session.commit()
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    only = "repeats applies to the md turns style only"
    cases = [(fmt, {"repeats": value}, only) for fmt in ("txt", "rttm", "srt", "vtt", "json")
             for value in ("drop", "keep")] + [
        ("md", {"repeats": "drop", "style": "blocks"}, only),
        ("md", {"repeats": "bogus"}, "unknown repeats value 'bogus'; valid: keep, drop"),
        # Fillers are validated first when both options are wrong.
        (
            "txt",
            {"fillers": "drop", "repeats": "drop"},
            "fillers applies to the md turns style only",
        ),
    ]
    for lang_params, detail in (
        ({"repeats": "drop", "lang": "es"}, "repeats cannot be combined with a translation"),
        (
            {"fillers": "drop", "repeats": "drop", "lang": "es"},
            "fillers cannot be combined with a translation",
        ),
    ):
        conflict = client.get(f"/review/{run_id}/export.md", params=lang_params)
        assert conflict.status_code == 422
        assert conflict.json()["detail"] == detail
    # keep never conflicts with a translation.
    assert client.get(
        f"/review/{run_id}/export.md", params={"repeats": "keep", "lang": "es"}
    ).status_code == 200
    for fmt, params, detail in cases:
        console = client.get(f"/review/{run_id}/export.{fmt}", params=params)
        assert console.status_code == 422
        assert console.json()["detail"] == detail
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": fmt, **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert public.status_code == 422
        assert public.json()["error"]["message"] == detail
        missing_console = client.get(f"/review/{uuid.uuid4()}/export.{fmt}", params=params)
        assert missing_console.status_code == 404
        missing_public = client.get(
            f"/api/v1/runs/{uuid.uuid4()}/transcript",
            params={"format": fmt, **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert missing_public.status_code == 422
        assert missing_public.json()["error"]["message"] == detail


def test_repeats_follow_the_selected_text_variant(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raw, enhanced and corrected text each get the filter on all three surfaces,
    and the stored evidence, enhancement and correction rows are left untouched."""
    with session_factory() as session:
        run_id = seed_words(session)
        segment = session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
        ).one()
        segment.raw_text = "".join(word for word, _ in _REPEAT_WORDS)
        segment.words = [
            {"word": word, "start": start, "end": start + 0.1} for word, start in _REPEAT_WORDS
        ]
        segment.enhanced_text = "Go to to the shop. We left. There."
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.sidecar = {"title": "Synthetic variants"}
        session.commit()
        segment_id = segment.id

    def snapshot() -> tuple[object, ...]:
        with session_factory() as session:
            stored = session.get(TranscriptSegment, segment_id)
            assert stored is not None
            reviews = session.scalars(
                select(SegmentReviewState).where(SegmentReviewState.pipeline_run_id == run_id)
            ).all()
            return (
                stored.raw_text,
                stored.enhanced_text,
                stored.words,
                [(r.corrected_text, r.corrected_at, r.verified_at) for r in reviews],
            )

    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    monkeypatch.setattr(
        "voxint.cli._engine_or_report",
        lambda: (create_engine(session_factory.kw["bind"].url), 0),
    )

    def export(variant: str) -> bytes:
        path = tmp_path / f"{variant}.md"
        args = ["export", str(run_id), "--format", "md", "--text", variant, "--drop-repeats"]
        assert main([*args, "--no-timestamps", "-o", str(path)]) == 0
        params = {"text": variant, "repeats": "drop", "timestamps": "false"}
        console = client.get(f"/review/{run_id}/export.md", params=params)
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": "md", **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert console.status_code == public.status_code == 200
        assert path.read_bytes() == console.content == public.content
        return console.content

    head = "# Synthetic variants\n\n"
    before = snapshot()
    assert export("raw") == (
        f"{head}**Alex:** Go to the store. We um we left.\n\n**Sam:** There.\n"
    ).encode()
    # Enhanced text no longer maps onto the words, so it stays whole under the
    # segment's speaker (coarse), and the filter still applies.
    assert export("enhanced") == f"{head}**Alex:** Go to the shop. We left. There.\n".encode()
    assert snapshot() == before
    with session_factory() as session:
        session.add(
            SegmentReviewState(
                transcript_segment_id=segment_id,
                pipeline_run_id=run_id,
                corrected_text="Go to the the market. We left. There.",
                corrected_at=datetime.now(UTC),
            )
        )
        session.commit()
    corrected_before = snapshot()
    assert export("corrected") == f"{head}**Alex:** Go to the market. We left. There.\n".encode()
    assert snapshot() == corrected_before


def test_repeats_declined_only_is_byte_identical_and_seams_hold(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only declined candidates means the default bytes, and a filler-only turn
    between two turns of one speaker keeps both copies (Alex / Sam / Alex)."""
    words = [
        ("I", 0.0), (" I", 0.05), (" I", 0.1), (" know", 0.15), (" that", 0.2),
        (" that", 0.25), (" is", 0.3), (" true.", 0.35), (" No,", 0.4), (" no.", 0.45),
        (" We", 0.5), (" said", 0.55), (' "the', 0.6), (' the"', 0.65), (" and", 0.7),
        (" we", 0.75), (" were", 0.8), (" we", 0.85), (" were", 0.9), (" we", 0.95),
        (" the", 0.98), (" um", 1.5), (" the", 2.2), (" end.", 2.5),
    ]
    with session_factory() as session:
        run_id = seed_words(session)
        segment = session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
        ).one()
        segment.raw_text = "".join(word for word, _ in words)
        segment.words = [{"word": w, "start": s, "end": s + 0.04} for w, s in words]
        session.add(DiarizationTurn(
            pipeline_run_id=run_id, turn_index=2, start_seconds=2, end_seconds=3,
            label="SPEAKER_00", skip_reason="too_short",
        ))
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.sidecar = {"title": "Synthetic declines"}
        session.commit()
    client = _build_client(session_factory, voxint_api_key="synthetic-api-key")
    monkeypatch.setattr(
        "voxint.cli._engine_or_report",
        lambda: (create_engine(session_factory.kw["bind"].url), 0),
    )

    def export(params: dict[str, str], flags: list[str], name: str) -> bytes:
        path = tmp_path / f"{name}.md"
        assert main(["export", str(run_id), "--format", "md", *flags, "-o", str(path)]) == 0
        console = client.get(f"/review/{run_id}/export.md", params=params)
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": "md", **params},
            auth=None,
            headers={"Authorization": "Bearer synthetic-api-key"},
        )
        assert console.status_code == public.status_code == 200
        assert path.read_bytes() == console.content == public.content
        return console.content

    for timestamps in (True, False):
        base = {} if timestamps else {"timestamps": "false"}
        base_flags = [] if timestamps else ["--no-timestamps"]
        default = export(base, base_flags, f"default-{timestamps}")
        assert export(
            {**base, "repeats": "drop"}, [*base_flags, "--drop-repeats"], f"drop-{timestamps}"
        ) == default
        fillers = export(
            {**base, "fillers": "drop"}, [*base_flags, "--drop-fillers"], f"f-{timestamps}"
        )
        both = export(
            {**base, "fillers": "drop", "repeats": "drop"},
            [*base_flags, "--drop-fillers", "--drop-repeats"],
            f"both-{timestamps}",
        )
        # Sam's "um" turn disappears and Alex's two turns merge; the seam keeps
        # "the the" apart, so repeats change nothing after fillers.
        assert both == fillers
        assert b"we were we were we the the end." in both
