"""Real marks across the shared renderer, HTTP downloads, read mode and CLI."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_translation_jobs import record_spanish, seed_run
from tests.integration.test_translation_view_export import _build_client
from voxint.adjudication.transcript import TranscriptText
from voxint.adjudication.word_marks import record_word_mark, undo_word_mark
from voxint.cli import main
from voxint.db.models import SegmentWordMark, TranscriptSegment, WordMarkAction
from voxint.export import TranscriptFormat
from voxint.export.filler_lists import DEFAULT_FILLER_LIST
from voxint.export.service import (
    MarkdownStyle,
    WordMarkPlacementError,
    render_run_transcript_report,
)

API_HEADERS = {"Authorization": "Bearer synthetic-api-key"}


def seed_markable(
    session: Session,
    *,
    body: str = "Um, basically, we start now.",
    start: float = 65,
) -> tuple[uuid.UUID, uuid.UUID]:
    run_id = seed_run(session, texts=[body])
    segment = session.scalars(
        select(TranscriptSegment).where(
            TranscriptSegment.pipeline_run_id == run_id,
        )
    ).one()
    segment.start_seconds, segment.end_seconds = start, start + 5
    segment.words = [
        {"word": (" " if index else "") + word, "start": start + index, "end": start + index + 1}
        for index, word in enumerate(body.split())
    ]
    session.commit()
    return run_id, segment.id


def mark(
    session: Session,
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    index: int,
    action: WordMarkAction,
) -> SegmentWordMark:
    row, replay = record_word_mark(
        session,
        run_id=run_id,
        segment_id=segment_id,
        start=index,
        end=index + 1,
        action=action,
        operator="test",
        user_id=None,
        idempotency_key=str(uuid.uuid4()),
    )
    assert not replay
    session.commit()
    return row


@pytest.fixture
def surface_client(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "voxint.cli._engine_or_report",
        lambda: (
            create_engine(session_factory.kw["bind"].url),
            0,
        ),
    )
    with _build_client(session_factory, voxint_api_key="synthetic-api-key") as client:
        yield client


@pytest.mark.parametrize("fmt", ["md", "txt"])
def test_marks_on_every_filtered_surface(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session)
        mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
        mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)
        report = render_run_transcript_report(
            session,
            run_id,
            TranscriptFormat(fmt),
            text=TranscriptText.CORRECTED,
            style=MarkdownStyle.TURNS,
            fillers=DEFAULT_FILLER_LIST,
            timestamps=False,
        )
        assert (report.fillers_removed, report.omitted, report.kept) == (0, 1, 1)
    params = {"fillers": "drop", "style": "turns", "timestamps": "false"}
    console = surface_client.get(f"/review/{run_id}/export.{fmt}", params=params)
    public = surface_client.get(
        f"/api/v1/runs/{run_id}/transcript",
        params={"format": fmt, **params},
        auth=None,
        headers=API_HEADERS,
    )
    path = tmp_path / f"marked.{fmt}"
    assert (
        main(
            [
                "export",
                str(run_id),
                "--format",
                fmt,
                "--style",
                "turns",
                "--drop-fillers",
                "--no-timestamps",
                "-o",
                str(path),
            ]
        )
        == 0
    )
    stderr = capsys.readouterr().err
    assert "words you left out: 1" in stderr and "fillers you kept: 1" in stderr
    assert console.status_code == public.status_code == 200
    assert path.read_bytes() == console.content == public.content == report.content.encode()
    assert "Um, we start now." in report.content
    if fmt == "md":
        assert report.content.endswith(
            "<!-- Filler words left out: 0. Preset en-1; words you left out: 1; "
            "fillers you kept: 1. The saved transcript is unchanged. -->\n"
        )
    else:
        assert "<!--" not in report.content
    read = surface_client.get(f"/runs/{run_id}/transcript", params={"read": "1", **params})
    assert read.status_code == 200
    assert "Um, we start now." in read.text
    assert "Words you left out: 1." in read.text and "Fillers you kept: 1." in read.text


@pytest.mark.parametrize("action", [WordMarkAction.KEEP, WordMarkAction.OMIT])
@pytest.mark.parametrize("start,clock", [(65, "01:05"), (3665, "1:01:05")], ids=["minute", "hour"])
def test_unplaceable_marks_refuse_every_surface(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    action: WordMarkAction,
    start: float,
    clock: str,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session, start=start)
        mark(session, run_id, segment_id, 0, action)
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        # A later enhancement may keep the segment and replace its displayed words.
        segment.enhanced_text = "A rewritten sentence."
        session.commit()
        for fmt in (TranscriptFormat.MARKDOWN, TranscriptFormat.TXT):
            with pytest.raises(WordMarkPlacementError) as caught:
                render_run_transcript_report(
                    session,
                    run_id,
                    fmt,
                    text=TranscriptText.ENHANCED,
                    style=MarkdownStyle.TURNS,
                    fillers=DEFAULT_FILLER_LIST,
                )
            message = str(caught.value)
            assert caught.value.segment_ids == {segment_id}
    assert clock in message
    assert "requested text cannot place the clean-up marks" in message
    assert "Clear the marks in the review console, or export with fillers kept." in message
    assert all(word not in message for word in ("token", "seq", "anchor", "\u2014"))
    for fmt in ("md", "txt"):
        params = {"fillers": "drop", "style": "turns", "text": "enhanced"}
        console = surface_client.get(f"/review/{run_id}/export.{fmt}", params=params)
        public = surface_client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": fmt, **params},
            auth=None,
            headers=API_HEADERS,
        )
        assert console.status_code == public.status_code == 409
        assert console.json()["detail"] == public.json()["error"]["message"] == message
        path = tmp_path / f"refused.{fmt}"
        assert (
            main(
                [
                    "export",
                    str(run_id),
                    "--format",
                    fmt,
                    "--style",
                    "turns",
                    "--text",
                    "enhanced",
                    "--drop-fillers",
                    "-o",
                    str(path),
                ]
            )
            == 2
        )
        assert not path.exists()
        captured = capsys.readouterr()
        assert message in captured.err and captured.out == ""
        # --force must also leave an existing destination untouched on refusal.
        path.write_bytes(b"prior export")
        assert (
            main(
                [
                    "export",
                    str(run_id),
                    "--format",
                    fmt,
                    "--style",
                    "turns",
                    "--text",
                    "enhanced",
                    "--drop-fillers",
                    "--force",
                    "-o",
                    str(path),
                ]
            )
            == 2
        )
        assert path.read_bytes() == b"prior export"
        capsys.readouterr()
    read = surface_client.get(
        f"/runs/{run_id}/transcript",
        params={"read": "1", "text": "enhanced", "fillers": "drop"},
    )
    assert read.status_code == 409 and read.json()["detail"] == message
    # Keeping fillers remains available even when the selected text cannot place a mark.
    kept = surface_client.get(
        f"/review/{run_id}/export.md",
        params={"text": "enhanced", "fillers": "keep"},
    )
    assert kept.status_code == 200 and "A rewritten sentence." in kept.text


# These selections never load or apply marks, including repeats-only and translations.
UNCHANGED = [
    pytest.param("srt", {}, [], id="srt"),
    pytest.param("vtt", {}, [], id="vtt"),
    pytest.param("json", {}, [], id="json"),
    pytest.param("rttm", {}, [], id="rttm"),
    pytest.param("txt", {}, [], id="timed-txt"),
    pytest.param("md", {"style": "blocks"}, ["--style", "blocks"], id="md-blocks"),
    pytest.param("md", {"repeats": "drop"}, ["--drop-repeats"], id="repeats-only"),
    pytest.param("md", {"fillers": "keep"}, [], id="md-keep"),
    pytest.param(
        "txt", {"style": "turns", "fillers": "keep"}, ["--style", "turns"], id="txt-turns-keep"
    ),
]


@pytest.mark.parametrize("fmt,params,flags", UNCHANGED)
def test_unfiltered_bytes_ignore_existing_marks(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
    tmp_path: Path,
    fmt: str,
    params: dict[str, str],
    flags: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session)

    def snapshot(label: str) -> tuple[bytes, bytes, bytes]:
        console = surface_client.get(f"/review/{run_id}/export.{fmt}", params=params)
        public = surface_client.get(
            f"/api/v1/runs/{run_id}/transcript",
            params={"format": fmt, **params},
            auth=None,
            headers=API_HEADERS,
        )
        path = tmp_path / f"{label}.{fmt}"
        assert main(["export", str(run_id), "--format", fmt, *flags, "-o", str(path)]) == 0
        assert console.status_code == public.status_code == 200
        assert path.read_bytes() == console.content == public.content
        return console.content, public.content, path.read_bytes()

    before = snapshot("before")
    with session_factory() as session:
        mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
        mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)

    def forbidden(*args: object) -> None:
        pytest.fail("this selection must not load marks")

    monkeypatch.setattr("voxint.export.service.effective_marks", forbidden)
    assert snapshot("after") == before


@pytest.mark.parametrize("repeats", ["keep", "drop"])
def test_read_keep_is_byte_identical(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    repeats: str,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session)
    url = f"/runs/{run_id}/transcript?read=1&fillers=keep&repeats={repeats}"
    before = surface_client.get(url)
    with session_factory() as session:
        mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
        mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)

    def forbidden(*args: object) -> None:
        pytest.fail("reading with fillers kept must not load marks")

    monkeypatch.setattr("voxint.api.routers.legacy_runs.effective_marks", forbidden)
    after = surface_client.get(url)
    assert before.status_code == after.status_code == 200
    assert before.content == after.content


def test_translated_markdown_is_byte_identical(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session)
        record_spanish(session, run_id)
        session.commit()
    url = f"/review/{run_id}/export.md?lang=es"
    before = surface_client.get(url)
    with session_factory() as session:
        mark(session, run_id, segment_id, 0, WordMarkAction.KEEP)
        mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)

    def forbidden(*args: object) -> None:
        pytest.fail("translation must not load marks")

    monkeypatch.setattr("voxint.export.service.effective_marks", forbidden)
    after = surface_client.get(url)
    assert before.status_code == after.status_code == 200
    assert before.content == after.content


def test_omit_only_read_mode_reports_all_filtered(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session, body="Basically")
        mark(session, run_id, segment_id, 0, WordMarkAction.OMIT)
    read = surface_client.get(f"/runs/{run_id}/transcript?read=1&fillers=drop")
    assert read.status_code == 200
    assert "The filters left out every word" in read.text
    assert "Words you left out: 1." in read.text
    assert "No transcript segments" not in read.text


def test_omit_undo_restores_every_export_bytes(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
    tmp_path: Path,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session)
        record_spanish(session, run_id)
        session.commit()
    selections = [(case.values[0], case.values[1], case.values[2]) for case in UNCHANGED]
    selections += [
        ("md", {"fillers": "drop"}, ["--drop-fillers"]),
        ("txt", {"fillers": "drop", "style": "turns"}, ["--drop-fillers", "--style", "turns"]),
    ]

    def snapshot(label: str) -> list[bytes]:
        result: list[bytes] = []
        for index, (fmt, params, flags) in enumerate(selections):
            console = surface_client.get(f"/review/{run_id}/export.{fmt}", params=params)
            public = surface_client.get(
                f"/api/v1/runs/{run_id}/transcript",
                params={"format": fmt, **params},
                auth=None,
                headers=API_HEADERS,
            )
            assert console.status_code == public.status_code == 200
            path = tmp_path / f"{label}-{index}.{fmt}"
            assert main(["export", str(run_id), "--format", fmt, *flags, "-o", str(path)]) == 0
            assert console.content == public.content == path.read_bytes()
            result.append(console.content)
        for fillers in ("keep", "drop"):
            read = surface_client.get(f"/runs/{run_id}/transcript?read=1&fillers={fillers}")
            assert read.status_code == 200
            result.append(read.content)
        translated = surface_client.get(f"/review/{run_id}/export.md?lang=es")
        assert translated.status_code == 200
        result.append(translated.content)
        return result

    before = snapshot("before")
    with session_factory() as session:
        row = mark(session, run_id, segment_id, 1, WordMarkAction.OMIT)
        mark_id = row.id
    during = surface_client.get(f"/review/{run_id}/export.md?fillers=drop")
    assert "words you left out: 1" in during.text
    with session_factory() as session:
        undo_word_mark(
            session,
            run_id=run_id,
            mark_id=mark_id,
            operator="test",
            user_id=None,
            grace_seconds=300,
        )
        session.commit()
    assert snapshot("after") == before


def test_default_list_comment_requires_a_mark_to_apply(
    session_factory: sessionmaker[Session],
    surface_client: TestClient,
) -> None:
    with session_factory() as session:
        run_id, segment_id = seed_markable(session)
    url = f"/review/{run_id}/export.md?fillers=drop"
    before = surface_client.get(url)
    assert before.status_code == 200 and "<!--" not in before.text
    with session_factory() as session:
        mark(session, run_id, segment_id, 3, WordMarkAction.KEEP)
        report = render_run_transcript_report(
            session,
            run_id,
            TranscriptFormat.MARKDOWN,
            text=TranscriptText.CORRECTED,
            fillers=DEFAULT_FILLER_LIST,
        )
        assert (report.omitted, report.kept) == (0, 0)
    after = surface_client.get(url)
    assert after.status_code == 200 and after.content == before.content


def test_placement_error_names_all_affected_segment_times(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, first_id = seed_markable(session)
        second = TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=1,
            start_seconds=3665,
            end_seconds=3666,
            raw_text="Um",
            enhanced_text="Changed words",
            diarization_label="S0",
            words=[{"word": "Um", "start": 3665, "end": 3666}],
        )
        session.add(second)
        session.flush()
        first = session.get(TranscriptSegment, first_id)
        assert first is not None
        first.enhanced_text = "Changed words"
        mark(session, run_id, first_id, 0, WordMarkAction.KEEP)
        mark(session, run_id, second.id, 0, WordMarkAction.OMIT)
        with pytest.raises(WordMarkPlacementError) as caught:
            render_run_transcript_report(
                session,
                run_id,
                TranscriptFormat.MARKDOWN,
                text=TranscriptText.ENHANCED,
                fillers=DEFAULT_FILLER_LIST,
            )
        assert "segments at 01:05, 1:01:05." in str(caught.value)
        assert caught.value.segment_ids == {first_id, second.id}
