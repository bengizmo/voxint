"""Pure logic for the first-run setup wizard: step model, field validation, and
the optional bounded "scan for existing media" walk.

Everything here is deliberately transport-agnostic (no ``Request``/``Response``):
the routes in :mod:`voxint.api.app` call these functions, map
:class:`SetupValidationError` to a re-rendered form with a visible message, and
own the DB transaction. Keeping the wizard's rules here (not inline in the routes)
makes them unit-testable without a running app and keeps one home for the bounds.

Security posture (single-operator home host, but a public OSS tool):
  * media folders are validated to sit under ``MEDIA_ROOT`` (resolve + containment)
    and to be existing directories;
  * the scan walks only the *registered* folders — never all of ``MEDIA_ROOT`` —
    prunes the trees Voxint owns (``incoming``/``artifacts``) and every symlink, and
    is bounded on both entries inspected and candidates surfaced so a deep or wide
    tree can neither hang the step nor auto-queue an unbounded number of runs;
  * the LLM API key is env-only and never touched here.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.config import Settings, llm_budget_fits_stage_lease
from voxint.db.models import MediaItem

# Input-shape bounds. Module constants (not Settings env vars) — like
# ingest.service's _MAX_URL_BYTES / _MAX_FILENAME_BYTES, these bound the shape of
# operator-typed input, not an operational tunable. The scan's file/entry caps ARE
# Settings knobs (config.setup_scan_max_*) because they bound how many runs the
# convenience step may auto-queue.
MAX_MEDIA_FOLDERS = 64
MAX_VOCABULARY_TERMS = 500
MAX_VOCABULARY_TERM_CHARS = 120
MAX_LLM_URL_CHARS = 2048
MAX_LLM_MODEL_CHARS = 200

# Top-level trees under MEDIA_ROOT that Voxint writes itself — uploaded/acquired
# sources (incoming) and normalized audio (artifacts). Never scanned, even when the
# operator registers the media root itself (".") as a folder.
_RESERVED_TREES = frozenset({"incoming", "artifacts"})

# Case-insensitive suffix allowlist for the scan. A convenience filter so the walk
# does not queue READMEs or stray files; ffprobe is NOT run per candidate (the
# PREPARE stage validates the actual media when the run executes).
_MEDIA_SUFFIXES = frozenset(
    {
        ".wav", ".mp3", ".m4a", ".m4v", ".flac", ".ogg", ".oga", ".opus", ".aac",
        ".wma", ".aiff", ".aif", ".alac", ".mp4", ".mkv", ".mov", ".webm", ".avi",
        ".mpeg", ".mpg", ".ts", ".3gp",
    }
)


class SetupValidationError(Exception):
    """A wizard field failed validation. The message is written by us (never echoes
    a secret) and is safe to render back into the form as a visible error."""


class WizardStep(StrEnum):
    """The six ordered wizard screens. The value is the ``?step=`` query token."""

    WELCOME = "welcome"
    MEDIA = "media"
    VOCABULARY = "vocabulary"
    LLM = "llm"
    SERVICES = "services"
    FINISH = "finish"


# Presentation order, used for the step indicator and next/prev navigation.
STEP_ORDER: tuple[WizardStep, ...] = (
    WizardStep.WELCOME,
    WizardStep.MEDIA,
    WizardStep.VOCABULARY,
    WizardStep.LLM,
    WizardStep.SERVICES,
    WizardStep.FINISH,
)


def parse_step(raw: str | None) -> WizardStep:
    """Map the ``?step=`` query value to a :class:`WizardStep`.

    An absent or unknown value falls back to WELCOME rather than 422-ing with
    FastAPI's JSON error — the wizard is a navigational surface, not an API, so a
    mistyped step should land the operator on the first screen.
    """
    if not raw:
        return WizardStep.WELCOME
    try:
        return WizardStep(raw)
    except ValueError:
        return WizardStep.WELCOME


def next_step(step: WizardStep) -> WizardStep:
    """The step after ``step`` (FINISH is a fixed point)."""
    index = STEP_ORDER.index(step)
    return STEP_ORDER[min(index + 1, len(STEP_ORDER) - 1)]


def normalize_media_folders(raw_folders: Iterable[str], media_root: Path) -> list[str]:
    """Validate operator-entered folders and return MEDIA_ROOT-relative POSIX paths.

    Each folder is stripped; blanks are dropped. A folder must be a relative path
    that resolves to an existing directory under ``media_root`` (resolve + a
    containment check reject traversal and symlink-escape). Results are de-duplicated
    preserving first-seen order. Raises :class:`SetupValidationError` on an absolute
    path, an escape, a non-directory, or more than :data:`MAX_MEDIA_FOLDERS`.
    """
    root = media_root.resolve()
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_folders:
        folder = raw.strip()
        if not folder:
            continue
        if Path(folder).is_absolute():
            raise SetupValidationError(
                f"media folder must be relative to the media root: {folder!r}"
            )
        resolved = (root / folder).resolve()
        if not resolved.is_relative_to(root):
            raise SetupValidationError(f"media folder is outside the media root: {folder!r}")
        if not resolved.is_dir():
            raise SetupValidationError(
                f"media folder is not an existing directory: {folder!r}"
            )
        rel = resolved.relative_to(root).as_posix()  # "." when it is the root itself
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    if len(out) > MAX_MEDIA_FOLDERS:
        raise SetupValidationError(f"at most {MAX_MEDIA_FOLDERS} media folders")
    return out


def normalize_vocabulary(raw_text: str) -> list[str]:
    """One term per line: stripped, blanks dropped, de-duplicated in order.

    Terms may contain spaces (multi-word names), so only newlines split them.
    Raises :class:`SetupValidationError` on a term over
    :data:`MAX_VOCABULARY_TERM_CHARS` or more than :data:`MAX_VOCABULARY_TERMS`.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in raw_text.splitlines():
        term = line.strip()
        if not term:
            continue
        if len(term) > MAX_VOCABULARY_TERM_CHARS:
            raise SetupValidationError(
                f"vocabulary term exceeds {MAX_VOCABULARY_TERM_CHARS} characters"
            )
        if term not in seen:
            seen.add(term)
            out.append(term)
    if len(out) > MAX_VOCABULARY_TERMS:
        raise SetupValidationError(f"at most {MAX_VOCABULARY_TERMS} vocabulary terms")
    return out


def normalize_llm_base_url(raw: str) -> str | None:
    """Validate an optional LLM base URL override. Blank ⇒ None (env fallback).

    An absolute http/https URL with a plain host and no embedded credentials. No
    SSRF/public-address restriction — a local inference endpoint (localhost) is a
    legitimate target, unlike the ingest URL path. Raises
    :class:`SetupValidationError` otherwise.
    """
    value = raw.strip()
    if not value:
        return None
    if len(value) > MAX_LLM_URL_CHARS:
        raise SetupValidationError(f"LLM base URL exceeds {MAX_LLM_URL_CHARS} characters")
    if any(ch.isspace() for ch in value):
        raise SetupValidationError("LLM base URL must not contain whitespace")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise SetupValidationError("LLM base URL is malformed") from exc
    if parts.scheme not in ("http", "https"):
        raise SetupValidationError("LLM base URL must be an absolute http/https URL")
    if not parts.hostname:
        raise SetupValidationError("LLM base URL has no host")
    if parts.username is not None or parts.password is not None:
        raise SetupValidationError("LLM base URL must not embed credentials")
    return value


def normalize_llm_model(raw: str) -> str | None:
    """Validate an optional LLM model override. Blank ⇒ None (env fallback)."""
    value = raw.strip()
    if not value:
        return None
    if len(value) > MAX_LLM_MODEL_CHARS:
        raise SetupValidationError(f"LLM model exceeds {MAX_LLM_MODEL_CHARS} characters")
    return value


def validate_llm_enable(settings: Settings) -> None:
    """Guard the two preconditions for turning LLM enhancement on from the wizard.

    Raises :class:`SetupValidationError` (the route persists ``llm_enabled=False``
    and shows the message — fail closed) when the env has no API key, or when the
    configured run budget plus worst-case overrun would not fit the enhance_match
    stage lease (the same invariant the env-time validator enforces, shared via
    :func:`voxint.config.llm_budget_fits_stage_lease`). The API key is env-only, so
    the wizard can surface *whether* one is set but never reads or stores its value.
    """
    if not settings.llm_api_key.strip():
        raise SetupValidationError(
            "Set LLM_API_KEY in the environment (then restart the worker) before "
            "enabling LLM enhancement — the key is never stored here."
        )
    if not llm_budget_fits_stage_lease(settings):
        raise SetupValidationError(
            "The configured LLM run budget plus worst-case overrun does not fit the "
            "enhance_match stage lease. Lower LLM_RUN_BUDGET_SECONDS or raise "
            "STAGE_LEASE_SECONDS before enabling enhancement."
        )


@dataclass(frozen=True)
class ScanResult:
    """Outcome of a bounded media scan, for the step-2 preview and confirm."""

    candidates: list[str]  # net-new, MEDIA_ROOT-relative POSIX paths (bounded)
    inspected: int  # files examined (the walk's work metric)
    hit_entry_cap: bool  # walk stopped at setup_scan_max_entries
    hit_file_cap: bool  # candidate list stopped at setup_scan_max_files
    root_missing: bool  # MEDIA_ROOT is absent / not a directory


def scan_media_folders(
    session: Session, media_root: Path, folders: Iterable[str], settings: Settings
) -> ScanResult:
    """Walk the registered ``folders`` for net-new media, bounded and containment-safe.

    Only the given folders are walked (each re-validated to still be a directory
    under ``media_root`` — a stored folder may have been removed since it was saved).
    The reserved trees Voxint owns and every symlink are pruned. Files are filtered
    by :data:`_MEDIA_SUFFIXES` (case-insensitive), de-duplicated, and checked against
    existing ``MediaItem.source_path`` in one query so only genuinely new paths are
    returned. Both caps are honoured; a missing media root yields an empty result
    with ``root_missing=True`` rather than raising.
    """
    root = media_root.resolve()
    if not root.is_dir():
        return ScanResult([], 0, False, False, root_missing=True)

    reserved = {(root / name).resolve() for name in _RESERVED_TREES}
    max_entries = settings.setup_scan_max_entries
    max_files = settings.setup_scan_max_files

    found: dict[str, None] = {}  # ordered set of relative POSIX candidate paths
    inspected = 0
    hit_entry_cap = False
    hit_file_cap = False

    for folder in folders:
        base = (root / folder).resolve()
        if not (base.is_relative_to(root) and base.is_dir()):
            continue  # revalidate: a stored folder can vanish between save and scan
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            here = Path(dirpath)
            # Prune reserved trees and symlinked subdirs in place so os.walk never
            # descends into them.
            dirnames[:] = [
                d
                for d in dirnames
                if (here / d).resolve() not in reserved and not (here / d).is_symlink()
            ]
            for name in filenames:
                inspected += 1
                if inspected > max_entries:
                    hit_entry_cap = True
                    break
                path = here / name
                if path.is_symlink():
                    continue
                if path.suffix.lower() not in _MEDIA_SUFFIXES:
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    continue  # a symlinked file slipped through — containment wins
                rel = resolved.relative_to(root).as_posix()
                if rel not in found:
                    found[rel] = None
                    if len(found) >= max_files:
                        hit_file_cap = True
                        break
            if hit_entry_cap or hit_file_cap:
                break
        if hit_entry_cap or hit_file_cap:
            break

    candidates = list(found)
    if candidates:
        existing = set(
            session.execute(
                select(MediaItem.source_path).where(
                    MediaItem.source_path.in_(candidates)
                )
            ).scalars()
        )
        candidates = [c for c in candidates if c not in existing]
    return ScanResult(
        candidates=candidates,
        inspected=inspected,
        hit_entry_cap=hit_entry_cap,
        hit_file_cap=hit_file_cap,
        root_missing=False,
    )
