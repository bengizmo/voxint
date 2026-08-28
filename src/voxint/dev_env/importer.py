"""Verify and idempotently apply development-environment bundles."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.api.setup_wizard import MAX_MEDIA_FOLDERS
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import MediaFolder, Project
from voxint.domain_packs.registry import available_domain_packs
from voxint.media.folders import overlapping_registration

_ADVISORY_LOCK_KEY = 0x766F78696E746465
_PROFILES = {"smoke", "realistic", "all"}
_SETTING_DEFAULTS: dict[str, bool | None] = {
    "onboarding_complete": False,
    "llm_enabled": False,
    "watch_folder_enabled": None,
    "enrichment_names_enabled": None,
    "enrichment_names_llm_enabled": None,
    "enrichment_web_research_enabled": None,
    "enrichment_run_assets_enabled": None,
    "enrichment_run_assets_autogenerate": None,
    "semantic_index_enabled": None,
    "semantic_index_autogenerate": None,
}


@dataclass
class ApplyResult:
    settings_created: bool
    projects_created: list[str] = field(default_factory=list)
    folders_created: list[str] = field(default_factory=list)
    folders_skipped: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_seed(bundle_path: Path) -> dict[str, Any]:
    seed_path = bundle_path / "seed" / "dev-seed.yaml"
    if not seed_path.is_file():
        raise ValueError(f"missing seed file: {seed_path}")
    try:
        loaded = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {seed_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{seed_path} must contain a YAML mapping")
    return loaded


def _alembic_head() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    config = Config(str(repository_root / "alembic.ini"))
    config.set_main_option("script_location", str(repository_root / "alembic"))
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise ValueError("could not determine the current Alembic head")
    return head


def _normalized_folder(raw_path: object, media_root: Path) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("media folder path must be a non-empty string")
    posix = PurePosixPath(raw_path)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"media folder path escapes MEDIA_ROOT: {raw_path!r}")
    normalized = posix.as_posix()
    root = media_root.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"media folder path escapes MEDIA_ROOT: {raw_path!r}") from None
    if not candidate.is_dir():
        raise ValueError(f"media folder does not exist under MEDIA_ROOT: {normalized}")
    return normalized


def _manifest_entries(manifest: object) -> list[tuple[str, str]]:
    """Extract common path/checksum pairs without imposing a manifest schema."""
    if not isinstance(manifest, dict):
        raise ValueError("bundle-manifest.json must contain a JSON object")
    raw = manifest.get("checksums", manifest.get("files", manifest.get("sha256")))
    if isinstance(raw, dict):
        entries: list[tuple[str, str]] = []
        for path, value in raw.items():
            checksum = value.get("sha256") if isinstance(value, dict) else value
            if isinstance(path, str) and isinstance(checksum, str):
                entries.append((path, checksum))
        return entries
    if isinstance(raw, list):
        entries = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            path = item.get("path", item.get("file"))
            checksum = item.get("sha256")
            if isinstance(path, str) and isinstance(checksum, str):
                entries.append((path, checksum))
        return entries
    raise ValueError("bundle manifest has no readable checksums or files collection")


def _verify_manifest(bundle_path: Path, errors: list[str]) -> None:
    manifest_path = bundle_path / "bundle-manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = _manifest_entries(manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid bundle-manifest.json: {exc}")
        return
    for relative, expected in entries:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"manifest path escapes bundle: {relative}")
            continue
        target = bundle_path / path.as_posix()
        if not target.is_file():
            errors.append(f"manifest file is missing: {relative}")
            continue
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            errors.append(f"could not read manifest file {relative}: {exc}")
            continue
        normalized_expected = expected.removeprefix("sha256:").lower()
        if actual != normalized_expected:
            errors.append(f"checksum mismatch: {relative}")


def verify_bundle(bundle_path: Path, media_root: Path) -> VerifyResult:
    """Check a bundle's required structure, media, and optional checksums."""
    bundle_path = Path(bundle_path)
    media_root = Path(media_root)
    errors: list[str] = []
    warnings: list[str] = []
    seed_path = bundle_path / "seed" / "dev-seed.yaml"
    packs_path = bundle_path / "domain-packs"
    if not bundle_path.is_dir():
        errors.append(f"bundle directory does not exist: {bundle_path}")
    if not seed_path.is_file():
        errors.append(f"missing seed/dev-seed.yaml: {seed_path}")
    if not packs_path.is_dir():
        errors.append(f"missing domain-packs directory: {packs_path}")

    seed_data: dict[str, Any] | None = None
    if seed_path.is_file():
        try:
            seed_data = _load_seed(bundle_path)
        except ValueError as exc:
            errors.append(str(exc))
    if seed_data is not None:
        folders = seed_data.get("media_folders", [])
        if not isinstance(folders, list):
            errors.append("media_folders must be a list")
        else:
            for item in folders:
                if not isinstance(item, dict):
                    errors.append("each media_folders entry must be a mapping")
                    continue
                try:
                    _normalized_folder(item.get("path"), media_root)
                except ValueError as exc:
                    errors.append(str(exc))

    _verify_manifest(bundle_path, errors)
    if not (bundle_path / "bundle-manifest.json").exists():
        warnings.append("bundle-manifest.json is absent; checksums were not verified")
    return VerifyResult(ok=not errors, errors=errors, warnings=warnings)


def _get_or_create_project(session: Session, data: dict[str, Any]) -> tuple[Project, bool]:
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("project name must be a non-empty string")
    existing = session.execute(select(Project).where(Project.name == name)).scalar_one_or_none()
    if existing is not None:
        return existing, False
    project = Project(
        id=uuid.uuid5(uuid.NAMESPACE_URL, f"voxint-dev-env:{name}"),
        name=name,
        description=data.get("description"),
        vocabulary=data.get("vocabulary"),
        corrections=data.get("corrections"),
    )
    try:
        with session.begin_nested():
            session.add(project)
            session.flush()
    except IntegrityError:
        adopted = session.execute(select(Project).where(Project.name == name)).scalar_one_or_none()
        if adopted is None:
            raise
        return adopted, False
    return project, True


def _get_or_create_folder(
    session: Session, *, path: str, project_id: uuid.UUID, pack_name: str
) -> bool:
    existing = session.execute(
        select(MediaFolder).where(MediaFolder.path == path)
    ).scalar_one_or_none()
    if existing is not None:
        return False
    folder = MediaFolder(path=path, project_id=project_id, domain_pack=pack_name, watch=True)
    try:
        with session.begin_nested():
            session.add(folder)
            session.flush()
    except IntegrityError:
        adopted = session.execute(
            select(MediaFolder).where(MediaFolder.path == path)
        ).scalar_one_or_none()
        if adopted is None:
            raise
        return False
    return True


def apply_bundle(
    session: Session,
    bundle_path: Path,
    media_root: Path,
    settings: Settings,
    *,
    profile: str = "all",
    dry_run: bool = False,
) -> ApplyResult:
    """Idempotently seed one bundle; flush changes but leave commit to the caller."""
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    bundle_path = Path(bundle_path)
    media_root = Path(media_root)
    seed_data = _load_seed(bundle_path)
    revision = seed_data.get("schema_revision")
    head = _alembic_head()
    if str(revision) != head:
        raise ValueError(
            f"bundle schema revision {revision!r} is incompatible with Alembic head {head!r}"
        )

    packs_dir = bundle_path / "domain-packs"
    if not packs_dir.is_dir():
        raise ValueError(f"missing domain-packs directory: {packs_dir}")
    bundle_settings = settings.model_copy(update={"domain_packs_dir": packs_dir})
    packs = available_domain_packs(bundle_settings)

    projects_data = seed_data.get("projects", [])
    folders_data = seed_data.get("media_folders", [])
    settings_data = seed_data.get("app_settings", {})
    if not isinstance(projects_data, list) or not all(isinstance(v, dict) for v in projects_data):
        raise ValueError("projects must be a list of mappings")
    if not isinstance(folders_data, list) or not all(isinstance(v, dict) for v in folders_data):
        raise ValueError("media_folders must be a list of mappings")
    if not isinstance(settings_data, dict):
        raise ValueError("app_settings must be a mapping")
    unknown_settings = set(settings_data) - set(_SETTING_DEFAULTS)
    if unknown_settings:
        raise ValueError(f"unsupported app_settings fields: {sorted(unknown_settings)}")

    if session.get_bind().dialect.name == "postgresql":
        session.execute(select(func.pg_advisory_xact_lock(_ADVISORY_LOCK_KEY)))

    existing_settings = get_app_settings(session)
    settings_created = existing_settings is None
    project_rows = {project.name: project for project in session.execute(select(Project)).scalars()}
    planned_projects: dict[str, uuid.UUID] = dict(
        (name, project.id) for name, project in project_rows.items()
    )
    projects_created: list[str] = []
    for data in projects_data:
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("project name must be a non-empty string")
        if name not in planned_projects:
            planned_projects[name] = uuid.uuid5(uuid.NAMESPACE_URL, f"voxint-dev-env:{name}")
            projects_created.append(name)

    existing_paths = list(session.execute(select(MediaFolder.path)).scalars())
    created_paths: list[str] = []
    skipped_paths: list[str] = []
    folder_plans: list[tuple[str, str, str]] = []
    for data in folders_data:
        raw_profile = data.get("profile", "all")
        if raw_profile not in _PROFILES:
            raise ValueError(f"unknown media folder profile {raw_profile!r}")
        raw_path = data.get("path")
        shown_path = raw_path if isinstance(raw_path, str) else repr(raw_path)
        if profile != "all" and raw_profile != profile:
            skipped_paths.append(shown_path)
            continue
        path = _normalized_folder(raw_path, media_root)
        project_name = data.get("project")
        if not isinstance(project_name, str) or project_name not in planned_projects:
            raise ValueError(f"media folder {path!r} references unknown project {project_name!r}")
        pack_name = data.get("domain_pack")
        if not isinstance(pack_name, str) or pack_name not in packs:
            raise ValueError(
                f"media folder {path!r} references unknown domain pack "
                f"{pack_name!r}; available: {sorted(packs)}"
            )
        if path in existing_paths or path in created_paths:
            skipped_paths.append(path)
            continue
        overlap = overlapping_registration(path, [*existing_paths, *created_paths])
        if overlap is not None:
            raise ValueError(f"media folder {path!r} overlaps registered folder {overlap!r}")
        if len(existing_paths) + len(created_paths) + 1 > MAX_MEDIA_FOLDERS:
            raise ValueError(f"cannot register more than {MAX_MEDIA_FOLDERS} media folders")
        created_paths.append(path)
        folder_plans.append((path, project_name, pack_name))

    if dry_run:
        print(
            "settings: would create" if settings_created else "settings: would update defaults only"
        )
        for name in projects_created:
            print(f"project: would create {name}")
        for path in created_paths:
            print(f"folder: would create {path}")
        for path in skipped_paths:
            print(f"folder: would skip {path}")
        return ApplyResult(settings_created, projects_created, created_paths, skipped_paths)

    row = get_or_create(session, llm_enabled_default=bool(settings_data.get("llm_enabled", False)))
    for name, value in settings_data.items():
        if settings_created or getattr(row, name) == _SETTING_DEFAULTS[name]:
            setattr(row, name, value)
    session.flush()

    actual_projects_created: list[str] = []
    for data in projects_data:
        project, created = _get_or_create_project(session, data)
        planned_projects[project.name] = project.id
        if created:
            actual_projects_created.append(project.name)
    actual_created_paths: list[str] = []
    for path, project_name, pack_name in folder_plans:
        if _get_or_create_folder(
            session,
            path=path,
            project_id=planned_projects[project_name],
            pack_name=pack_name,
        ):
            actual_created_paths.append(path)
        else:
            skipped_paths.append(path)
    return ApplyResult(
        settings_created, actual_projects_created, actual_created_paths, skipped_paths
    )
