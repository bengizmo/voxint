"""Upload/URL settings-folder membership (Console 2.0 P2b, ADR 0002 addendum).

``submit_upload``/``submit_url`` gain an OPTIONAL ``media_folder_id``: the bytes
still land under ``incoming/`` (never moved into the folder), but the run freezes
against that folder's — and its project's — vocabulary/corrections. Default ``None``
keeps the pre-P2b global baseline, byte-identical. The first submission's pick wins
on an idempotent replay: a re-POST with a different pick returns the original run
and never re-homes the row.

Service-level (the HTTP routes thread the picker in a later P2b commit); needs the
real Postgres test DB, so it is skipped without ``VOXINT_TEST_DATABASE_URL``.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import yaml
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import MediaFolder, MediaItem, PipelineRun, Project
from voxint.ingest import submit_upload, submit_url

PROJECT_VOCAB = ["projfolder-term"]
BASE_VOCAB = ["base-term"]


def _make_settings(tmp_path: Path) -> Settings:
    """A default pack on disk + a media root; upload cap comfortably large."""
    pack_dir = tmp_path / "base"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"name": "base", "vocabulary": BASE_VOCAB, "corrections": []})
    )
    media_root = tmp_path / "media"
    media_root.mkdir()
    return Settings(
        _env_file=None,
        domain_packs_dir=tmp_path,
        domain_pack_path=pack_dir,
        media_root=media_root,
        upload_max_bytes=10_000_000,
    )


def _seed_folder(
    session_factory: sessionmaker[Session], *, path: str, project_vocab: list[str]
) -> uuid.UUID:
    """A registered folder joined to a project carrying distinctive vocabulary."""
    with session_factory() as session:
        project = Project(name=f"P {path}", vocabulary=project_vocab, corrections=None)
        session.add(project)
        session.flush()
        folder = MediaFolder(path=path, project_id=project.id, domain_pack=None)
        session.add(folder)
        session.commit()
        return folder.id


def test_upload_freezes_against_picked_folder(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)
    folder_id = _seed_folder(
        session_factory, path="research", project_vocab=list(PROJECT_VOCAB)
    )
    with session_factory() as session:
        run = submit_upload(
            session,
            stream=io.BytesIO(b"audio-bytes"),
            filename="clip.wav",
            submission_id=uuid.uuid4().hex,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
            settings=settings,
            media_folder_id=folder_id,
        )
        session.commit()
        run_id, media_id = run.id, run.media_item_id
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None
        # Tagged with the folder, but the bytes still live under incoming/ — no move.
        assert media.media_folder_id == folder_id
        assert media.source_path.startswith("incoming/")
        snapshot = session.get(PipelineRun, run_id).domain_pack  # type: ignore[union-attr]
        assert snapshot is not None
        # The project's vocabulary won — proof the freeze resolved off the folder.
        assert snapshot["vocabulary"] == PROJECT_VOCAB


def test_upload_default_is_global_baseline(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """No folder pick ⇒ media_folder_id NULL and the pre-P2b global baseline."""
    settings = _make_settings(tmp_path)
    with session_factory() as session:
        run = submit_upload(
            session,
            stream=io.BytesIO(b"audio-bytes"),
            filename="clip.wav",
            submission_id=uuid.uuid4().hex,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
            settings=settings,
        )
        session.commit()
        run_id, media_id = run.id, run.media_item_id
    with session_factory() as session:
        assert session.get(MediaItem, media_id).media_folder_id is None  # type: ignore[union-attr]
        snapshot = session.get(PipelineRun, run_id).domain_pack  # type: ignore[union-attr]
        assert snapshot["vocabulary"] == BASE_VOCAB  # type: ignore[index]


def test_upload_replay_keeps_first_membership(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Same submission_id + same bytes, different folder pick: the original run comes
    back and its membership is NOT re-homed (first write wins, no re-freeze)."""
    settings = _make_settings(tmp_path)
    first = _seed_folder(session_factory, path="first", project_vocab=["first-term"])
    second = _seed_folder(session_factory, path="second", project_vocab=["second-term"])
    sub = uuid.uuid4().hex

    with session_factory() as session:
        run = submit_upload(
            session,
            stream=io.BytesIO(b"same-bytes"),
            filename="a.wav",
            submission_id=sub,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
            settings=settings,
            media_folder_id=first,
        )
        session.commit()
        original_run_id, media_id = run.id, run.media_item_id

    with session_factory() as session:
        replay = submit_upload(
            session,
            stream=io.BytesIO(b"same-bytes"),
            filename="a.wav",
            submission_id=sub,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
            settings=settings,
            media_folder_id=second,  # a different pick — must be ignored on replay
        )
        session.commit()
        assert replay.id == original_run_id  # no duplicate run

    with session_factory() as session:
        assert session.get(MediaItem, media_id).media_folder_id == first  # type: ignore[union-attr]


def test_url_freezes_against_picked_folder(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)
    folder_id = _seed_folder(
        session_factory, path="urls", project_vocab=list(PROJECT_VOCAB)
    )
    with session_factory() as session:
        run = submit_url(
            session,
            url="https://example.com/audio.mp3",
            submission_id=uuid.uuid4().hex,
            settings=settings,
            media_folder_id=folder_id,
        )
        session.commit()
        run_id, media_id = run.id, run.media_item_id
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None
        assert media.media_folder_id == folder_id
        assert media.source_path.startswith("incoming/")
        snapshot = session.get(PipelineRun, run_id).domain_pack  # type: ignore[union-attr]
        assert snapshot["vocabulary"] == PROJECT_VOCAB  # type: ignore[index]
