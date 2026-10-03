"""Recording dates through real PREPARE, exports, and maintenance sweeps."""

import shutil
import subprocess
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from tests.integration.test_cli_commands import _seed_completed_run
from voxint.adjudication.transcript import TranscriptText
from voxint.cli import main
from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.domain_packs.base import load_default
from voxint.export import TranscriptFormat
from voxint.export.service import MarkdownStyle, render_run_rttm, render_run_transcript
from voxint.media import recorded_date
from voxint.pipeline.engine import execute_run, submit
from voxint.pipeline.stages import prepare
from voxint.pipeline.stages.context import StageContext, build_stage_fns

_DATE = date(2026, 10, 2)
_MEDIA_TOOLS = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _source(root: Path, name: str, *, tagged: bool) -> Path:
    path = root / name
    cmd = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1",
           "-metadata", "creation_time=2026-10-03T05:30:00Z"]
    if tagged:
        cmd += ["-metadata", "com.apple.quicktime.creationdate=2026-10-02T23:30:00-0600"]
    subprocess.run([*cmd, "-movflags", "use_metadata_tags", str(path)],
                   check=True, capture_output=True)
    return path


@_MEDIA_TOOLS
@pytest.mark.parametrize("tagged, preset", [(True, None), (False, None),
                                           (True, date(2000, 1, 1))])
def test_prepare_recorded_date(
    session_factory: sessionmaker[Session], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, tagged: bool, preset: date | None,
) -> None:
    _source(tmp_path, "source.m4a", tagged=tagged)
    ctx = StageContext(asr=FakeASR(), diarizer=FakeDiarizer(), embedder=FakeEmbedder(),
                       llm=FakeLLM(), media_root=tmp_path)
    with session_factory() as session:
        media = MediaItem(source_path="source.m4a", recorded_on=preset)
        session.add(media)
        session.flush()
        run_id = submit(session, media.id, domain_pack=load_default().to_mapping()).id
        media_id = media.id
        session.commit()
    if preset is not None:
        def unexpected_probe(*args: object, **kwargs: object) -> None:
            raise AssertionError("an existing date must not be probed")
        monkeypatch.setattr(prepare, "probe_recorded_on", unexpected_probe)
    final = execute_run(session_factory, run_id, build_stage_fns(ctx))
    assert final.status is RunStatus.COMPLETED
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None
        assert media.recorded_on == (preset or (_DATE if tagged else None))
        prepare.run(ctx, session, run_id)
        session.commit()
        assert media.recorded_on == (preset or (_DATE if tagged else None))


@pytest.mark.parametrize("snapshot, expected", [
    ({"recorded": "1999-12-31"}, "31 Dec 1999"),
    ({"recorded": "invalid"}, "2 Oct 2026"), ({}, "2 Oct 2026"),
])
def test_recorded_date_export(
    session_factory: sessionmaker[Session], snapshot: dict[str, str], expected: str,
) -> None:
    with session_factory() as session:
        run_id = _seed_completed_run(session)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.sidecar = {"title": "Planning | call"}

        def render(fmt: TranscriptFormat, style: MarkdownStyle | None = None) -> str:
            return render_run_transcript(session, run_id, fmt, text=TranscriptText.RAW, style=style)

        before = {fmt: render(fmt) for fmt in TranscriptFormat if fmt != TranscriptFormat.MARKDOWN}
        blocks = render(TranscriptFormat.MARKDOWN, MarkdownStyle.BLOCKS)
        rttm = render_run_rttm(session, run_id)
        undated = render(TranscriptFormat.MARKDOWN)
        assert undated.startswith("# Planning \\| call\n")
        run.media_item.recorded_on = _DATE
        run.sidecar = {"title": "Planning | call", **snapshot}
        session.flush()
        assert render(TranscriptFormat.MARKDOWN) == undated.replace(
            "# Planning \\| call\n", f"# Planning \\| call \\| {expected}\n", 1,
        )
        assert render(TranscriptFormat.MARKDOWN, MarkdownStyle.BLOCKS) == blocks
        assert render_run_rttm(session, run_id) == rttm
        for fmt, original in before.items():
            assert render(fmt) == original
        assert run.media_item.recorded_on == _DATE


@_MEDIA_TOOLS
def test_backfill_recorded_dates(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source(tmp_path, "tagged.m4a", tagged=True)
    _source(tmp_path, "untagged.m4a", tagged=False)
    with session_factory() as session:
        rows = [MediaItem(source_path="old.m4a", current_path="tagged.m4a"),
                MediaItem(source_path="untagged.m4a"), MediaItem(source_path="missing.m4a"),
                MediaItem(source_path="preset.m4a", recorded_on=date(1999, 1, 1)),
                MediaItem(source_path="purged.m4a", purged_at=datetime.now(UTC))]
        session.add_all(rows)
        session.commit()
        found_id = rows[0].id
        real_probe = recorded_date.probe_recorded_on
        probes: list[str] = []

        def probe(path: Path, *, ffprobe_bin: str) -> date | None:
            probes.append(path.name)
            return real_probe(path, ffprobe_bin=ffprobe_bin)

        monkeypatch.setattr(recorded_date, "probe_recorded_on", probe)
        found: list[tuple[uuid.UUID, date]] = []

        def on_found(media_id: uuid.UUID, value: date) -> None:
            found.append((media_id, value))

        def forbidden_commit() -> None:
            raise AssertionError("dry run must not commit")

        with monkeypatch.context() as dry_patch:
            dry_patch.setattr(session, "commit", forbidden_commit)
            result = recorded_date.backfill_recorded_dates(
                session, tmp_path, ffprobe_bin="ffprobe", dry_run=True, on_found=on_found,
            )
        assert result.dated == ((found_id, _DATE),)
        assert result.no_tag == 1
        assert result.skipped_missing == ("missing.m4a",)
        assert found == [(found_id, _DATE)]
        session.expire_all()
        assert rows[0].recorded_on is None
        found.clear()
        result = recorded_date.backfill_recorded_dates(
            session, tmp_path, ffprobe_bin="ffprobe", on_found=on_found,
        )
        assert found == [(found_id, _DATE)]
        assert result.dated == ((found_id, _DATE),)
        assert rows[0].recorded_on == _DATE
        assert rows[3].recorded_on == date(1999, 1, 1)
        assert rows[4].recorded_on is None
        assert sorted(probes) == ["tagged.m4a", "tagged.m4a", "untagged.m4a", "untagged.m4a"]
        probes.clear()
        recorded_date.backfill_recorded_dates(session, tmp_path, ffprobe_bin="ffprobe")
        assert probes == ["untagged.m4a"]


@_MEDIA_TOOLS
def test_cli_backfill_recorded_dates_dry_run(
    session_factory: sessionmaker[Session], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _source(tmp_path, "tagged.m4a", tagged=True)
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    with session_factory() as session:
        media = MediaItem(source_path="tagged.m4a")
        session.add(media)
        session.commit()
        media_id = media.id
    assert main(["media", "backfill-recorded-dates", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert f"{media_id}: would set 2026-10-02" in output
    assert "would set the recording date on 1 media row(s) (dry run, nothing written)" in output
    assert "0 row(s) have no creation date tag (left empty)" in output
    assert "tagged.m4a" not in output
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None and media.recorded_on is None
    assert main(["media", "backfill-recorded-dates"]) == 0
    assert "set the recording date on 1 media row(s)" in capsys.readouterr().out
    assert main(["media", "backfill-recorded-dates"]) == 0
    output = capsys.readouterr().out
    assert "nothing to backfill: every media row already has a recording date" in output


def test_backfill_recorded_dates_concurrent_first_write(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A date committed during the probe wins over the backfill's stale NULL."""
    (tmp_path / "source.m4a").touch()
    with session_factory() as session:
        media = MediaItem(source_path="source.m4a")
        session.add(media)
        session.commit()
        media_id = media.id

        def concurrent_probe(path: Path, *, ffprobe_bin: str) -> date:
            with session_factory() as other:
                concurrent = other.get(MediaItem, media_id)
                assert concurrent is not None
                concurrent.recorded_on = date(1999, 1, 1)
                other.commit()
            return _DATE

        monkeypatch.setattr(recorded_date, "probe_recorded_on", concurrent_probe)
        result = recorded_date.backfill_recorded_dates(session, tmp_path, ffprobe_bin="ffprobe")
        assert result.dated == ()
        session.expire_all()
        assert media.recorded_on == date(1999, 1, 1)


def test_backfill_recorded_dates_interrupted_progress(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption after reporting a row preserves its committed date."""
    for name in ("first.m4a", "second.m4a"):
        (tmp_path / name).touch()
    with session_factory() as session:
        rows = [MediaItem(source_path="first.m4a", created_at=datetime(2020, 1, 1, tzinfo=UTC)),
                MediaItem(source_path="second.m4a", created_at=datetime(2020, 1, 2, tzinfo=UTC))]
        session.add_all(rows)
        session.commit()
        first_id, second_id = (row.id for row in rows)

    def probe(path: Path, *, ffprobe_bin: str) -> date:
        return _DATE

    def interrupt(media_id: uuid.UUID, value: date) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(recorded_date, "probe_recorded_on", probe)
    with session_factory() as session, pytest.raises(RuntimeError, match="interrupted"):
        recorded_date.backfill_recorded_dates(
            session, tmp_path, ffprobe_bin="ffprobe", on_found=interrupt,
        )
    with session_factory() as session:
        first = session.get(MediaItem, first_id)
        second = session.get(MediaItem, second_id)
        assert first is not None and first.recorded_on == _DATE
        assert second is not None and second.recorded_on is None
        result = recorded_date.backfill_recorded_dates(session, tmp_path, ffprobe_bin="ffprobe")
        assert result.dated == ((second_id, _DATE),)
