"""The ``text=cleaned`` console export (#758 slice 4) over the real app + Postgres.

Generations come from the real job executor with a scripted fake model, so
freshness and staleness follow the genuine source hash. Covers the refusals,
segment-format substitution, the D5 turns path, the one-snapshot read, the
byte-identical default matrix and the API/CLI negatives.
"""

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy import text as sql
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_cleanup_jobs import FakeCleanupLLM
from tests.integration.test_cleanup_writer import seed
from tests.integration.test_translation_jobs import make_settings, record_spanish, seed_run
from tests.integration.test_translation_view_export import _build_client
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.review_state import set_correction
from voxint.adjudication.splits import record_split
from voxint.adjudication.transcript import TranscriptText
from voxint.adjudication.word_marks import record_word_mark
from voxint.api.routers import adjudication_api
from voxint.cli import main
from voxint.db.models import (
    AppSettings,
    Decision,
    DiarizationTurn,
    RunCleanup,
    Speaker,
    TranscriptSegment,
    WordMarkAction,
)
from voxint.enrichment.cleanup_jobs import create_job, execute_job
from voxint.enrichment.translations import TranslationError
from voxint.export import TranscriptFormat
from voxint.export.filler_lists import DEFAULT_FILLER_LIST
from voxint.export.service import (
    CleanedVariant,
    ExportOptionError,
    TranslationMismatchError,
    render_run_transcript,
)

SEGMENT_FORMATS = ("txt", "srt", "vtt", "json")
CLEANED_LINES = (
    "I think it's fine.",
    "So we went",
    "to Paris, yesterday.",
    "Okay then.",
    "So -- we go.",
)
API_KEY = "synthetic-api-key"


def _generate(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    proposals: dict[int, str] | None = None,
) -> None:
    with session_factory() as session:
        job, already = create_job(session, pipeline_run_id=run_id, settings=make_settings())
        assert job is not None and not already
        job_id = job.id
        session.commit()
    execute_job(
        session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM(proposals)
    )


def _queue(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as session:
        job, _ = create_job(session, pipeline_run_id=run_id, settings=make_settings())
        assert job is not None
        session.commit()


def _segment(session: Session, run_id: uuid.UUID, index: int) -> TranscriptSegment:
    return session.execute(
        select(TranscriptSegment).where(
            TranscriptSegment.pipeline_run_id == run_id,
            TranscriptSegment.segment_index == index,
        )
    ).scalar_one()


def _edit(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as session:
        set_correction(session, segment=_segment(session, run_id, 0), text="Edited afterwards.")
        session.commit()


def _split(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as session:
        record_split(session, parent=_segment(session, run_id, 0), word_index=2, operator="test")
        session.commit()


@pytest.fixture
def run_id(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        return seed(session)


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> TestClient:
    return _build_client(session_factory, voxint_api_key=API_KEY)


def _cleaned(client: TestClient, run_id: uuid.UUID, fmt: str, **params: str) -> str:
    response = client.get(
        f"/review/{run_id}/export.{fmt}", params={"text": "cleaned", **params}
    )
    assert response.status_code == 200, (fmt, params, response.text)
    return str(response.text)


def _detail(client: TestClient, run_id: uuid.UUID, fmt: str, status: int, **params: str) -> str:
    response = client.get(
        f"/review/{run_id}/export.{fmt}", params={"text": "cleaned", **params}
    )
    assert response.status_code == status, (fmt, params, response.text)
    return str(response.json()["detail"])


# ------------------------------------------------------------------ content


def test_segment_formats_substitute_the_stored_lines(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    for fmt in SEGMENT_FORMATS:
        body = _cleaned(client, run_id, fmt)
        for line in CLEANED_LINES:
            assert line in body, (fmt, line)
        assert "I mean" not in body and "Um," not in body and "uh," not in body, fmt
    blocks = _cleaned(client, run_id, "md", style="blocks")
    assert "I think it's fine." in blocks and "I mean" not in blocks
    # Cue timing is the reviewed export's: only the text changes.
    reviewed = client.get(f"/review/{run_id}/export.srt").text
    assert [row for row in reviewed.splitlines() if "-->" in row] == [
        row for row in _cleaned(client, run_id, "srt").splitlines() if "-->" in row
    ]


@pytest.mark.parametrize("timestamps", [True, False])
def test_turns_apply_the_deletions_as_omits(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    client: TestClient,
    timestamps: bool,
) -> None:
    _generate(session_factory, run_id)
    flag = str(timestamps).lower()
    a, b, c = ("[00:00:00] ", "[00:00:05] ", "[00:00:10] ") if timestamps else ("", "", "")
    body = (
        f"{a}**S0:** I think it's fine.\n\n"
        f"{b}So we went to Paris, yesterday.\n\n"
        f"{c}Okay then. So -- we go.\n"
    )
    md = _cleaned(client, run_id, "md", timestamps=flag)
    assert md.split("\n\n", 1)[1] == body
    txt = _cleaned(client, run_id, "txt", style="turns", timestamps=flag)
    assert txt == body.replace("**S0:**", "S0:")


@pytest.mark.parametrize(
    "fmt, params", [("md", {}), ("txt", {"style": "turns"}), ("md", {"style": "blocks"}),
                    ("txt", {}), ("srt", {}), ("json", {})],
)
def test_a_generation_without_deletions_equals_the_reviewed_export(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient,
    fmt: str, params: dict[str, str],
) -> None:
    _generate(session_factory, run_id, {})
    with session_factory() as session:
        head = session.scalars(select(RunCleanup).where(RunCleanup.pipeline_run_id == run_id)).one()
        assert head.counts["lines_changed"] == 0
    reviewed = client.get(f"/review/{run_id}/export.{fmt}", params=params)
    assert _cleaned(client, run_id, fmt, **params).encode() == reviewed.content


def _seed_two_speakers(session: Session) -> uuid.UUID:
    """One word-timed segment whose words split between two named speakers."""
    run_id = seed_run(session, texts=["Um, hello there, you know, my friend."])
    segment = _segment(session, run_id, 0)
    segment.diarization_label = "SPEAKER_00"
    words = segment.raw_text.split()
    segment.words = [
        {"word": (" " if i else "") + word, "start": i * 0.5, "end": i * 0.5 + 0.5}
        for i, word in enumerate(words)
    ]
    for i, name in enumerate(("Alex", "Sam")):
        speaker = Speaker(display_name=name)
        session.add(speaker)
        session.flush()
        label = f"SPEAKER_{i:02}"
        # Alex speaks 0-1.5 s (Um, hello there,), Sam the rest.
        start, end = (0.0, 1.5) if i == 0 else (1.5, 5.0)
        session.add(DiarizationTurn(
            pipeline_run_id=run_id, turn_index=i, start_seconds=start, end_seconds=end,
            label=label, skip_reason="too_short",
        ))
        record_decision(
            session, pipeline_run_id=run_id, diarization_label=label,
            decision=Decision.ASSIGN, speaker_id=speaker.id,
            operator="test", idempotency_key=str(uuid.uuid4()),
        )
    session.commit()
    return run_id


@pytest.mark.parametrize("fmt, params", [("md", {}), ("txt", {"style": "turns"})])
def test_turns_attribution_matches_the_reviewed_turns(
    session_factory: sessionmaker[Session], client: TestClient,
    fmt: str, params: dict[str, str],
) -> None:
    with session_factory() as session:
        run_id = _seed_two_speakers(session)
    _generate(session_factory, run_id, {0: "hello there, my friend."})
    reviewed = client.get(f"/review/{run_id}/export.{fmt}", params=params).text
    cleaned = _cleaned(client, run_id, fmt, **params)
    tag = "**{}:**" if fmt == "md" else "{}:"
    alex, sam = tag.format("Alex"), tag.format("Sam")
    # The same speaker grouping, minus the deleted words. Sam's turn starts at
    # its first kept word, exactly as a #757 omit mark would move it.
    assert reviewed.endswith(
        f"[00:00:00] {alex} Um, hello there,\n\n[00:00:01] {sam} you know, my friend.\n"
    )
    assert cleaned.endswith(
        f"[00:00:00] {alex} Hello there,\n\n[00:00:02] {sam} My friend.\n"
    )


# ------------------------------------------------------------------ refusals


def test_no_generation_is_409(run_id: uuid.UUID, client: TestClient) -> None:
    for fmt in (*SEGMENT_FORMATS, "md"):
        assert "generate one from the Cleaned page" in _detail(client, run_id, fmt, 409)


def test_a_running_first_generation_is_409(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _queue(session_factory, run_id)
    for fmt in (*SEGMENT_FORMATS, "md"):
        assert "still being generated" in _detail(client, run_id, fmt, 409)


def test_a_current_generation_stays_exportable_while_another_runs(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    _queue(session_factory, run_id)
    assert "I think it's fine." in _cleaned(client, run_id, "txt")


def test_an_edit_stales_the_variant(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    _edit(session_factory, run_id)
    for fmt in (*SEGMENT_FORMATS, "md"):
        detail = _detail(client, run_id, fmt, 409)
        assert detail == adjudication_api._CLEANED_STALE_DETAIL
        assert "out of date" in detail and "regenerate it from the Cleaned page" in detail
    # A regeneration in flight says so instead of calling it stale.
    _queue(session_factory, run_id)
    assert "still being generated" in _detail(client, run_id, "txt", 409)


def test_a_split_stales_the_variant(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    _split(session_factory, run_id)
    assert "out of date" in _detail(client, run_id, "md", 409)


def test_a_speaker_rename_does_not_stale(
    session_factory: sessionmaker[Session], client: TestClient
) -> None:
    with session_factory() as session:
        run_id = _seed_two_speakers(session)
    _generate(session_factory, run_id, {0: "hello there, my friend."})
    with session_factory() as session:
        speaker = session.scalars(select(Speaker).where(Speaker.display_name == "Alex")).one()
        speaker.display_name = "Robin"
        session.commit()
    md = _cleaned(client, run_id, "md")
    assert "**Robin:** Hello there," in md and "Alex" not in md


def test_a_filler_list_or_mark_change_does_not_stale(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    before = _cleaned(client, run_id, "md")
    dropped = client.get(f"/review/{run_id}/export.md", params={"fillers": "drop"}).text
    assert "Paris" in dropped and "I mean, I think" in dropped
    with session_factory() as session:
        provenance = current_cleanup_config(session, run_id)
        row = session.get(AppSettings, 1) or AppSettings(id=1)
        row.fillers_add = {"en": ["paris"]}
        session.add(row)
        record_word_mark(
            session, run_id=run_id, segment_id=_segment(session, run_id, 0).id,
            start=2, end=3, action=WordMarkAction.OMIT, operator="test", user_id=None,
            idempotency_key=str(uuid.uuid4()),
        )
        session.commit()
    # The reviewed export with fillers dropped now uses the list and the mark...
    dropped = client.get(f"/review/{run_id}/export.md", params={"fillers": "drop"}).text
    assert "Paris" not in dropped and "I mean, think" in dropped
    # ...while the cleaned variant ignores both and keeps its provenance.
    assert _cleaned(client, run_id, "md") == before
    with session_factory() as session:
        assert current_cleanup_config(session, run_id) == provenance


def current_cleanup_config(session: Session, run_id: uuid.UUID) -> object:
    return session.scalars(
        select(RunCleanup.config).where(
            RunCleanup.pipeline_run_id == run_id, RunCleanup.superseded_by_cleanup_id.is_(None),
        )
    ).one()


def test_the_snapshot_isolation_does_not_outlive_the_request(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    engine = session_factory.kw["bind"]
    pool_size = engine.pool.size()
    for _ in range(pool_size + 1):
        _cleaned(client, run_id, "md")
    # Every pooled connection is back at the server default.
    connections = [engine.connect() for _ in range(pool_size)]
    try:
        for connection in connections:
            level = connection.execute(sql("SHOW transaction_isolation")).scalar_one()
            assert level == "read committed"
    finally:
        for connection in connections:
            connection.close()


def test_a_service_option_refusal_is_422(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate(session_factory, run_id)

    def refuse(*args: object, **kwargs: object) -> str:
        raise ExportOptionError("the cleaned variant renders the reviewed text alone")

    monkeypatch.setattr(adjudication_api, "render_run_transcript", refuse)
    assert "reviewed text alone" in _detail(client, run_id, "md", 422)


def test_an_unreadable_source_is_409(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate(session_factory, run_id)

    def refuse(session: Session, pipeline_run_id: uuid.UUID) -> object:
        raise TranslationError("run has too many lines")

    monkeypatch.setattr(adjudication_api, "load_translation_source", refuse)
    assert _detail(client, run_id, "txt", 409) == "run has too many lines"


def test_a_render_mismatch_maps_to_the_stale_detail(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate(session_factory, run_id)

    def mismatch(*args: object, **kwargs: object) -> str:
        raise TranslationMismatchError

    monkeypatch.setattr(adjudication_api, "render_run_transcript", mismatch)
    assert _detail(client, run_id, "md", 409) == adjudication_api._CLEANED_STALE_DETAIL


@pytest.mark.parametrize(
    "fmt, params, needle",
    [
        ("txt", {"lang": "es"}, "a translation cannot be combined with the cleaned variant"),
        ("srt", {"lang": "es"}, "a translation cannot be combined with the cleaned variant"),
        ("md", {"fillers": "drop"}, "fillers cannot be combined with the cleaned variant"),
        ("txt", {"style": "turns", "repeats": "drop"},
         "repeats cannot be combined with the cleaned variant"),
        ("srt", {"fillers": "drop"}, "fillers applies to md turns and txt turns only"),
        ("md", {"style": "bogus"}, "unknown style"),
    ],
)
def test_incompatible_options_are_422_before_any_lookup(
    run_id: uuid.UUID, client: TestClient, fmt: str, params: dict[str, str], needle: str,
) -> None:
    # 422 wins even with no generation: the options are checked first.
    assert needle in _detail(client, run_id, fmt, 422, **params)


def test_kept_fillers_and_repeats_are_allowed(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    _generate(session_factory, run_id)
    assert "I think it's fine." in _cleaned(client, run_id, "md", fillers="keep", repeats="keep")


def test_a_missing_run_is_404(client: TestClient) -> None:
    response = client.get(f"/review/{uuid.uuid4()}/export.md", params={"text": "cleaned"})
    assert response.status_code == 404


def test_api_and_cli_refuse_cleaned(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _generate(session_factory, run_id)
    public = client.get(
        f"/api/v1/runs/{run_id}/transcript", params={"format": "txt", "text": "cleaned"},
        auth=None, headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert public.status_code == 422
    assert "unknown transcript text 'cleaned'" in public.text
    monkeypatch.setattr("voxint.cli._engine_or_report", lambda: (
        create_engine(session_factory.kw["bind"].url), 0,
    ))
    with pytest.raises(SystemExit) as exc:
        main(["export", str(run_id), "--text", "cleaned", "-o", str(tmp_path / "x.txt")])
    assert exc.value.code == 2
    assert "invalid choice: 'cleaned'" in capsys.readouterr().err


# ------------------------------------------------------------------ snapshot


# An edit leaks into the turns render; a split changes the segment line count.
@pytest.mark.parametrize("fmt, mutate", [("md", _edit), ("txt", _split)])
def test_the_freshness_check_and_render_share_one_snapshot(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient,
    monkeypatch: pytest.MonkeyPatch, fmt: str,
    mutate: Callable[[sessionmaker[Session], uuid.UUID], None],
) -> None:
    """A change committing between the check and the render cannot leak in."""
    _generate(session_factory, run_id)
    render = render_run_transcript

    def change_then_render(*args: object, **kwargs: object) -> str:
        mutate(session_factory, run_id)
        return render(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(adjudication_api, "render_run_transcript", change_then_render)
    body = _cleaned(client, run_id, fmt)
    assert body.count("I think it's fine.") == 1 and "Edited afterwards." not in body
    monkeypatch.undo()
    # The change did commit: the next export sees it and refuses.
    assert "out of date" in _detail(client, run_id, fmt, 409)


# ------------------------------------------------------------- service seam


def test_the_service_refuses_cleaned_with_other_options(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    variant = CleanedVariant(CLEANED_LINES, {})
    with session_factory() as session:
        for kwargs in (
            {"text": TranscriptText.RAW},
            {"text": TranscriptText.CORRECTED, "translated_texts": CLEANED_LINES},
            {"text": TranscriptText.CORRECTED, "drop_repeats": True},
            {"text": TranscriptText.CORRECTED, "fillers": DEFAULT_FILLER_LIST},
        ):
            with pytest.raises(ExportOptionError, match="cleaned variant"):
                render_run_transcript(
                    session, run_id, TranscriptFormat.MARKDOWN, cleaned=variant,
                    **kwargs,
                )


def test_the_service_refuses_lines_or_omits_that_no_longer_fit(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    with session_factory() as session:
        with pytest.raises(TranslationMismatchError):
            render_run_transcript(
                session, run_id, TranscriptFormat.TXT, text=TranscriptText.CORRECTED,
                cleaned=CleanedVariant(CLEANED_LINES[:2], {}),
            )
        segment_id = _segment(session, run_id, 0).id
        with pytest.raises(TranslationMismatchError):
            render_run_transcript(
                session, run_id, TranscriptFormat.MARKDOWN, text=TranscriptText.CORRECTED,
                cleaned=CleanedVariant(CLEANED_LINES, {(segment_id, 90, 91): "omit"}),
            )


# --------------------------------------------------- byte-identical defaults


def _default_matrix() -> Iterator[tuple[str, dict[str, str]]]:
    for text in (None, "corrected", "enhanced", "raw"):
        base = {"text": text} if text else {}
        for fmt in ("srt", "vtt", "json"):
            yield fmt, base
        for timestamps in ("true", "false"):
            timed = {**base, "timestamps": timestamps}
            for style in ("blocks", "turns"):
                yield "md", {**timed, "style": style}
            yield "md", timed
            yield "txt", timed
            yield "txt", {**timed, "style": "turns"}
        if text in (None, "corrected"):
            for timestamps in ("true", "false"):
                filtered = {**base, "timestamps": timestamps, "fillers": "drop", "repeats": "drop"}
                yield "md", filtered
                yield "txt", {**filtered, "style": "turns"}


def _snapshot(client: TestClient, run_id: uuid.UUID) -> list[bytes]:
    bodies = []
    for fmt, params in _default_matrix():
        response = client.get(f"/review/{run_id}/export.{fmt}", params=params)
        assert response.status_code == 200, (fmt, params, response.text)
        bodies.append(response.content)
    bodies.append(client.get(f"/review/{run_id}/export.rttm").content)
    for fmt in ("txt", "md", "srt", "json"):
        public = client.get(
            f"/api/v1/runs/{run_id}/transcript", params={"format": fmt},
            auth=None, headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert public.status_code == 200
        bodies.append(public.content)
    return bodies


def test_defaults_are_byte_identical_with_or_without_a_cleaned_variant(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, client: TestClient
) -> None:
    # Distinct raw, enhanced and corrected text, so a variant mix-up shows.
    with session_factory() as session:
        _segment(session, run_id, 3).enhanced_text = "So, we go."
        set_correction(session, segment=_segment(session, run_id, 2), text="Okay, then.")
        session.commit()
    raw, enhanced, corrected = (
        client.get(f"/review/{run_id}/export.txt", params={"text": variant}).text
        for variant in ("raw", "enhanced", "corrected")
    )
    assert "So -- we go." in raw and "So, we go." not in raw and "Okay, then." not in raw
    assert "So, we go." in enhanced and "Okay then." in enhanced
    assert "So, we go." in corrected and "Okay, then." in corrected
    baseline = _snapshot(client, run_id)
    _generate(session_factory, run_id)
    assert _snapshot(client, run_id) == baseline, "current generation"
    _generate(session_factory, run_id, {0: "I think it's fine.", 1: "Um, so we went"})
    with session_factory() as session:
        rows = session.scalars(select(RunCleanup).where(RunCleanup.pipeline_run_id == run_id)).all()
        assert sum(row.superseded_by_cleanup_id is not None for row in rows) == 1
    assert _snapshot(client, run_id) == baseline, "superseded generation"
    with session_factory() as session:
        record_spanish(session, run_id)
        session.commit()
    assert _snapshot(client, run_id) == baseline, "coexisting translation"
    # The coexisting translation and the cleaned variant stay independent.
    assert client.get(f"/review/{run_id}/export.txt", params={"lang": "es"}).status_code == 200
    assert "I think it's fine." in _cleaned(client, run_id, "txt")
