"""The per-field config-resolution freeze (issue #153, ADR 0002 addendum).

The numerics-sensitive core of P2a: at submit a run's effective ``vocabulary`` and
``corrections`` are frozen into ``pipeline_runs.domain_pack`` by PER-FIELD
REPLACEMENT — each field independently takes its first present layer in the order
explicit per-run pack override (CLI/sidecar) > project field > folder pack field >
global baseline (the default pack unioned with the operator's ``app_settings``
glossary/corrections). A project field is nullable: NULL inherits the layer below,
an empty list is "explicitly none" and wins. The snapshot is stamped
``config_resolution_version: 2`` so the worker uses the frozen vocabulary as-is
rather than re-unioning the live glossary; every pre-#153 run keeps its live-union
path (asserted in tests/unit/test_run_preferences.py).

These tests set a DISTINCT sentinel value in every layer, so a resolution that
leaked a masked lower layer into the frozen result would fail — the whole point of
freezing the effective config rather than reasoning about it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import yaml
from sqlalchemy.orm import Session, sessionmaker

from voxint.app_settings import get_or_create
from voxint.config import Settings
from voxint.db.models import MediaFolder, PipelineRun, Project
from voxint.ingest import preview_effective_config, submit_media_item
from voxint.ingest.sidecar import parse_sidecar

# One distinctive value per layer, per field. If any resolution leaked a masked
# layer, the frozen list would carry the wrong sentinel and the assertion fails.
EXPLICIT_PACK = "explicitpack"
FOLDER_PACK = "folderpack"
BASE_PACK = "base"  # the configured DEFAULT pack (global baseline's pack half)

EXPLICIT_VOCAB = ["explicit-term"]
FOLDER_VOCAB = ["folder-term"]
BASE_VOCAB = ["base-term"]
PROJECT_VOCAB = ["project-term"]
GLOSSARY_VOCAB = ["glossary-term"]  # app_settings.vocabulary (global)
# Global baseline vocab = default pack words unioned with the operator glossary.
GLOBAL_VOCAB = ["base-term", "glossary-term"]


def _corr(rule_id: str, match: str) -> dict[str, object]:
    return {
        "id": rule_id,
        "match": match,
        "replace": rule_id.upper(),
        "case_sensitive": True,
        "whole_word": True,
    }


EXPLICIT_CORR = [_corr("ex", "exmatch")]
FOLDER_CORR = [_corr("fo", "fomatch")]
BASE_CORR = [_corr("ba", "bamatch")]
PROJECT_CORR = [_corr("pj", "pjmatch")]
GLOSSARY_CORR = [_corr("gl", "glmatch")]  # app_settings.corrections (global)
GLOBAL_CORR_IDS = ["ba", "gl"]  # default pack rules then operator rules


def _write_pack(root: Path, name: str, vocab: list[str], corr: list[dict[str, object]]) -> None:
    pack_dir = root / name
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"name": name, "vocabulary": vocab, "corrections": corr})
    )


def _make_settings(tmp_path: Path) -> Settings:
    """Three packs on disk; ``base`` is the configured default (global baseline)."""
    _write_pack(tmp_path, BASE_PACK, BASE_VOCAB, BASE_CORR)
    _write_pack(tmp_path, FOLDER_PACK, FOLDER_VOCAB, FOLDER_CORR)
    _write_pack(tmp_path, EXPLICIT_PACK, EXPLICIT_VOCAB, EXPLICIT_CORR)
    return Settings(
        _env_file=None,
        domain_packs_dir=tmp_path,
        domain_pack_path=tmp_path / BASE_PACK,
    )


def _seed_global(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.vocabulary = list(GLOSSARY_VOCAB)
        row.corrections = list(GLOSSARY_CORR)
        session.commit()


def _freeze(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    suffix: str,
    explicit: str | None = None,
    project_vocab: list[str] | None = None,
    project_corr: list[dict[str, object]] | None = None,
    folder_pack: str | None = None,
    sidecar_pack: str | None = None,
    under_folder: bool = True,
) -> dict[str, object]:
    """Set up project+folder+media for one case, submit, return the frozen snapshot.

    ``under_folder=False`` places the media under NO registered folder
    (media_folder_id is None) — the uploads/URLs path, which always resolves to the
    global baseline unless an explicit pack is named.
    """
    folder_path = f"proj{suffix}"
    source_path = f"{folder_path}/a.wav" if under_folder else f"loose{suffix}/a.wav"
    with session_factory() as session:
        project = Project(
            name=f"Project {suffix}",
            vocabulary=project_vocab,
            corrections=project_corr,
        )
        session.add(project)
        session.flush()
        if under_folder:
            session.add(
                MediaFolder(
                    path=folder_path, project_id=project.id, domain_pack=folder_pack
                )
            )
        session.commit()
    sidecar = (
        parse_sidecar(f"domain_pack: {sidecar_pack}\n", source_name="a.wav.yaml")
        if sidecar_pack is not None
        else None
    )
    with session_factory() as session:
        run = submit_media_item(
            session,
            source_path,
            settings=settings,
            domain_pack_name=explicit,
            sidecar=sidecar,
        )
        session.commit()
        run_id = run.run_id
    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.domain_pack is not None
        return stored.domain_pack


def _corr_ids(snapshot: dict[str, object]) -> list[str]:
    return [rule["id"] for rule in snapshot["corrections"]]  # type: ignore[index,union-attr]


# The 2^3 presence matrix over (explicit, project, folder), for BOTH fields at once
# (a present project sets both fields). Global is the unconditional fallback, so it
# is not a fourth bit. First present wins: explicit > project > folder > global.
_MATRIX = [
    # (explicit, project, folder, expected_vocab, expected_corr_ids)
    (1, 1, 1, EXPLICIT_VOCAB, ["ex"]),
    (1, 1, 0, EXPLICIT_VOCAB, ["ex"]),
    (1, 0, 1, EXPLICIT_VOCAB, ["ex"]),
    (1, 0, 0, EXPLICIT_VOCAB, ["ex"]),
    (0, 1, 1, PROJECT_VOCAB, ["pj"]),
    (0, 1, 0, PROJECT_VOCAB, ["pj"]),
    (0, 0, 1, FOLDER_VOCAB, ["fo"]),
    (0, 0, 0, GLOBAL_VOCAB, GLOBAL_CORR_IDS),
]


@pytest.mark.parametrize("e,p,f,exp_vocab,exp_corr", _MATRIX)
def test_freeze_precedence_matrix(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    e: int,
    p: int,
    f: int,
    exp_vocab: list[str],
    exp_corr: list[str],
) -> None:
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)
    snapshot = _freeze(
        session_factory,
        settings,
        suffix=f"{e}{p}{f}",
        explicit=EXPLICIT_PACK if e else None,
        project_vocab=list(PROJECT_VOCAB) if p else None,
        project_corr=[dict(r) for r in PROJECT_CORR] if p else None,
        folder_pack=FOLDER_PACK if f else None,
    )
    assert snapshot["config_resolution_version"] == 2
    assert snapshot["vocabulary"] == exp_vocab
    assert _corr_ids(snapshot) == exp_corr


def test_freeze_fields_resolve_independently(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """Cross-field asymmetry: a project that sets ONLY vocabulary keeps its
    vocabulary but inherits corrections from the folder pack, and vice versa. The
    two fields never move together."""
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)

    vocab_only = _freeze(
        session_factory,
        settings,
        suffix="vonly",
        project_vocab=list(PROJECT_VOCAB),
        project_corr=None,
        folder_pack=FOLDER_PACK,
    )
    assert vocab_only["vocabulary"] == PROJECT_VOCAB
    assert _corr_ids(vocab_only) == ["fo"]

    corr_only = _freeze(
        session_factory,
        settings,
        suffix="conly",
        project_vocab=None,
        project_corr=[dict(r) for r in PROJECT_CORR],
        folder_pack=FOLDER_PACK,
    )
    assert corr_only["vocabulary"] == FOLDER_VOCAB
    assert _corr_ids(corr_only) == ["pj"]


def test_freeze_project_empty_list_is_explicitly_none(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """An empty project list is "explicitly none" and WINS over the folder pack —
    distinct from NULL (inherit). Proven against a folder pack that would otherwise
    supply values."""
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)
    snapshot = _freeze(
        session_factory,
        settings,
        suffix="empty",
        project_vocab=[],
        project_corr=[],
        folder_pack=FOLDER_PACK,
    )
    assert snapshot["vocabulary"] == []
    assert snapshot["corrections"] == []


def test_freeze_caller_pack_beats_sidecar_pack(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """A caller-supplied explicit pack wins over the sidecar's, which wins over the
    project/folder/global layers (both are the layer-1 override)."""
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)
    # Caller names explicitpack; sidecar names folderpack; project + folder are set
    # too. The caller's explicit pack is layer 1 and wins outright.
    caller = _freeze(
        session_factory,
        settings,
        suffix="caller",
        explicit=EXPLICIT_PACK,
        sidecar_pack=FOLDER_PACK,
        project_vocab=list(PROJECT_VOCAB),
        project_corr=[dict(r) for r in PROJECT_CORR],
        folder_pack=FOLDER_PACK,
    )
    assert caller["name"] == EXPLICIT_PACK
    assert caller["vocabulary"] == EXPLICIT_VOCAB
    assert _corr_ids(caller) == ["ex"]

    # Sidecar-only (no caller pack): the sidecar's pack is the explicit override and
    # still beats the project/folder/global layers.
    sidecar = _freeze(
        session_factory,
        settings,
        suffix="sidecar",
        sidecar_pack=EXPLICIT_PACK,
        project_vocab=list(PROJECT_VOCAB),
        folder_pack=FOLDER_PACK,
    )
    assert sidecar["name"] == EXPLICIT_PACK
    assert sidecar["vocabulary"] == EXPLICIT_VOCAB


def test_freeze_no_folder_resolves_to_global_baseline(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """Media under no registered folder (the uploads/URLs shape, media_folder_id is
    None) resolves to the global baseline: the default pack unioned with the
    operator glossary/corrections — unchanged from the pre-#153 behavior."""
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)
    snapshot = _freeze(
        session_factory,
        settings,
        suffix="loose",
        under_folder=False,
    )
    assert snapshot["config_resolution_version"] == 2
    assert snapshot["vocabulary"] == GLOBAL_VOCAB
    assert _corr_ids(snapshot) == GLOBAL_CORR_IDS


# --- The read-only re-run preview (Console 2.0 P2b) ---------------------------
#
# preview_effective_config runs the SAME resolution walk the freeze does, so a
# preview matches exactly what a re-run against the same DB state would freeze —
# the numerics/honesty invariant behind "show the effective config before dispatch".


def _seed_project_folder(
    session_factory: sessionmaker[Session],
    *,
    suffix: str,
    project_vocab: list[str] | None,
    project_corr: list[dict[str, object]] | None,
    folder_pack: str | None,
) -> tuple[str, str]:
    """Seed one project + folder; return ``(folder_path, media_folder_id)``."""
    folder_path = f"proj{suffix}"
    with session_factory() as session:
        project = Project(
            name=f"Project {suffix}",
            vocabulary=project_vocab,
            corrections=project_corr,
        )
        session.add(project)
        session.flush()
        folder = MediaFolder(
            path=folder_path, project_id=project.id, domain_pack=folder_pack
        )
        session.add(folder)
        session.commit()
        return folder_path, str(folder.id)


def test_preview_matches_dispatched_snapshot(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """The preview's pack + counts equal the snapshot a re-run freezes, and its
    per-field source labels name the branch the ONE walk actually took: project
    vocabulary (set) but folder-pack corrections (project corr NULL)."""
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)
    folder_path, folder_id = _seed_project_folder(
        session_factory,
        suffix="prev",
        project_vocab=list(PROJECT_VOCAB),
        project_corr=None,
        folder_pack=FOLDER_PACK,
    )

    with session_factory() as session:
        preview = preview_effective_config(
            session, uuid.UUID(folder_id), settings=settings
        )
    assert preview.pack_name == FOLDER_PACK
    assert preview.vocabulary_source == "project"
    assert preview.corrections_source == "folder"
    assert preview.folder_path == folder_path
    assert preview.project_name == "Project prev"

    # Submit a media item under the same folder; the frozen snapshot must match the
    # preview's shape exactly (same walk, same DB state).
    with session_factory() as session:
        run = submit_media_item(session, f"{folder_path}/a.wav", settings=settings)
        session.commit()
        run_id = run.run_id
    with session_factory() as session:
        snapshot = session.get(PipelineRun, run_id).domain_pack  # type: ignore[union-attr]
    assert snapshot is not None
    assert preview.pack_name == snapshot["name"]
    assert preview.vocabulary_count == len(snapshot["vocabulary"])
    assert preview.corrections_count == len(snapshot["corrections"])
    assert snapshot["vocabulary"] == PROJECT_VOCAB
    assert _corr_ids(snapshot) == ["fo"]


def test_preview_no_folder_is_global_baseline(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    """media_folder_id None (an upload/URL with no folder pick) previews the global
    baseline: both fields sourced "global", no folder/project, counts matching a
    loose submit."""
    settings = _make_settings(tmp_path)
    _seed_global(session_factory)
    with session_factory() as session:
        preview = preview_effective_config(session, None, settings=settings)
    assert preview.pack_name == BASE_PACK
    assert preview.vocabulary_source == "global"
    assert preview.corrections_source == "global"
    assert preview.folder_path is None
    assert preview.project_name is None
    assert preview.vocabulary_count == len(GLOBAL_VOCAB)
    assert preview.corrections_count == len(GLOBAL_CORR_IDS)
