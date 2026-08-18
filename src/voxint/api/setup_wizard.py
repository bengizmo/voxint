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
  * the LLM API key may be UI-stored (issue #10): this module only shape-normalizes
    it (:func:`normalize_llm_api_key`) and presence-checks the *effective* key
    (:func:`validate_llm_enable`) — never renders, logs, or echoes its value.
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
MAX_LLM_KEY_CHARS = 512

# Immediate child directories listed per level by the folder browser (issue #63).
# A shape bound like MAX_MEDIA_FOLDERS, not an operational tunable: it caps one
# directory's rendered children so a pathologically wide folder cannot balloon a
# single fragment. Deeper folders beyond the cap stay reachable via the browser's
# server-canonicalized "go to folder" jump, so the cap never makes one unreachable.
MAX_BROWSE_ENTRIES = 500

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
    reserved = {(root / name).resolve() for name in _RESERVED_TREES}
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
        if _under_reserved(resolved, reserved):
            # incoming/ and artifacts/ are Voxint-owned (uploads and normalized
            # audio); registering them would re-ingest the pipeline's own outputs.
            raise SetupValidationError(
                f"media folder is a reserved Voxint directory: {folder!r}"
            )
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
        # .port is lazily parsed; touch it so a bad ":port" (":abc", out-of-range)
        # fails HERE rather than crashing httpx.Client construction in the worker —
        # which runs before run_pipeline's failure handling and would strand the run
        # QUEUED for the recovery sweep to re-publish forever.
        _ = parts.port
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


def normalize_api_key(raw: str, *, label: str, max_chars: int) -> str | None:
    """Normalize an optional API-key field. Blank ⇒ ``None`` = **no change**.

    The provider-neutral core shared by every credential field (the LLM key, the
    web-search key — issue #76): a password field submits blank on almost every
    save (it is never prefilled, so re-saving without re-typing the key must LEAVE
    THE STORED KEY UNTOUCHED — not wipe it). So blank returns ``None`` as a
    *no-change sentinel*, distinct from an explicit removal (a separate remove
    checkbox handled by the route). A non-blank value is stripped of surrounding
    whitespace, then rejected if it still contains any whitespace, control
    character, or non-ASCII character (a real API key has none — this catches paste
    accidents before the key reaches an ``Authorization`` header, whose latin-1
    encoding would otherwise crash the outbound request at run/doctor time instead
    of failing closed here) or exceeds ``max_chars``. ``label`` names the field in
    the message; the message is a fixed string and NEVER interpolates the submitted
    value (it is a credential).
    """
    value = raw.strip()
    if not value:
        return None
    if len(value) > max_chars:
        raise SetupValidationError(f"{label} exceeds {max_chars} characters")
    # Printable ASCII only (0x21-0x7E): rejects whitespace, control chars, and any
    # non-ASCII code point that httpx cannot encode into the Authorization header.
    if any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in value):
        raise SetupValidationError(
            f"{label} must contain only printable ASCII characters"
            " (no whitespace or control characters)"
        )
    return value


def normalize_llm_api_key(raw: str) -> str | None:
    """Normalize the optional LLM API key field (see :func:`normalize_api_key`)."""
    return normalize_api_key(raw, label="LLM API key", max_chars=MAX_LLM_KEY_CHARS)


def normalize_web_search_api_key(raw: str) -> str | None:
    """Normalize the optional web-search provider API key (issue #76). Same rules
    and the same no-change/blank sentinel as :func:`normalize_llm_api_key`; only the
    message label differs. Reuses ``MAX_LLM_KEY_CHARS`` as the shared length ceiling
    (a generous bound for any provider key — a second knob would not earn its place).
    """
    return normalize_api_key(raw, label="Web-search API key", max_chars=MAX_LLM_KEY_CHARS)


def validate_llm_enable(effective_api_key: str, settings: Settings) -> None:
    """Guard the two preconditions for turning LLM enhancement on.

    Raises :class:`SetupValidationError` (the route persists ``llm_enabled=False``
    and shows the message — fail closed) when no effective API key is configured, or
    when the configured run budget plus worst-case overrun would not fit the
    enhance_match stage lease (the same invariant the env-time validator enforces,
    shared via :func:`voxint.config.llm_budget_fits_stage_lease`).

    ``effective_api_key`` is the *post-save* effective key — the submitted-or-stored
    row value winning over env — resolved by the caller via
    :func:`voxint.app_settings.resolve_effective_llm_api_key` (already stripped, so
    ``""`` means "no key anywhere"). It is passed in rather than read from
    ``settings`` because the key may now be UI-stored on the ``app_settings`` row,
    not env-only; the value is used only for a presence check and is never rendered.
    """
    if not effective_api_key:
        raise SetupValidationError(
            "No LLM API key is configured. Enter one here (or set LLM_API_KEY in the "
            "environment) before enabling LLM enhancement."
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


def _under_reserved(path: Path, reserved: set[Path]) -> bool:
    """True if ``path`` is (or is nested under) a reserved tree Voxint owns."""
    return any(path.is_relative_to(r) for r in reserved)


def scan_media_folders(
    session: Session, media_root: Path, folders: Iterable[str], settings: Settings
) -> ScanResult:
    """Walk the registered ``folders`` for net-new media, bounded and containment-safe.

    Only the given folders are walked (each re-validated to still be a directory
    under ``media_root`` — a stored folder may have been removed since it was saved,
    and a folder that IS or sits under a reserved tree is skipped entirely, not just
    pruned as a child). The traversal is an explicit ``os.scandir`` stack rather than
    ``os.walk``: ``os.walk`` materializes a whole directory's entries before yielding,
    so a single directory with millions of children could blow ``setup_scan_max_entries``
    before the cap is checked; ``os.scandir`` is lazy, so counting each entry as it is
    consumed makes the cap a true bound on work. Symlinks (files and dirs) are never
    followed. Files are filtered by :data:`_MEDIA_SUFFIXES` (case-insensitive) and
    de-duplicated; the collected candidates are checked against existing
    ``MediaItem.source_path`` in one query and the ``setup_scan_max_files`` cap is
    applied to the *net-new* result (so an already-ingested first batch can't mask
    genuinely new media). A missing media root yields ``root_missing=True``, not a raise.
    """
    root = media_root.resolve()
    if not root.is_dir():
        return ScanResult([], 0, False, False, root_missing=True)

    reserved = {(root / name).resolve() for name in _RESERVED_TREES}
    max_entries = settings.setup_scan_max_entries
    max_files = settings.setup_scan_max_files

    stack: list[Path] = []
    seen_dirs: set[Path] = set()
    for folder in folders:
        base = (root / folder).resolve()
        if not (base.is_relative_to(root) and base.is_dir()):
            continue  # revalidate: a stored folder can vanish between save and scan
        if _under_reserved(base, reserved):
            continue  # a folder registered AS incoming/artifacts is never scanned
        if base not in seen_dirs:
            seen_dirs.add(base)
            stack.append(base)

    found: dict[str, None] = {}  # ordered set of relative POSIX candidate paths
    inspected = 0
    hit_entry_cap = False

    while stack and not hit_entry_cap:
        current = stack.pop()
        try:
            scanner = os.scandir(current)
        except OSError:
            continue  # a directory that vanished or is unreadable — skip, don't fail
        with scanner:
            for entry in scanner:
                inspected += 1
                if inspected > max_entries:
                    hit_entry_cap = True
                    break
                if entry.is_symlink():
                    continue  # never follow a symlink (avoids escapes and cycles)
                if entry.is_dir(follow_symlinks=False):
                    child = Path(entry.path).resolve()
                    if child in reserved or child in seen_dirs:
                        continue
                    seen_dirs.add(child)
                    stack.append(child)
                elif entry.is_file(follow_symlinks=False):
                    if Path(entry.name).suffix.lower() not in _MEDIA_SUFFIXES:
                        continue
                    resolved = Path(entry.path).resolve()
                    if not resolved.is_relative_to(root):
                        continue  # defence-in-depth containment (symlink-free here)
                    found[resolved.relative_to(root).as_posix()] = None

    candidates = list(found)
    hit_file_cap = False
    if candidates:
        existing = set(
            session.execute(
                select(MediaItem.source_path).where(
                    MediaItem.source_path.in_(candidates)
                )
            ).scalars()
        )
        net_new = [c for c in candidates if c not in existing]
        # Apply the file cap to NET-NEW results (after the existence filter), so an
        # already-ingested first batch never fills the cap and hides new media.
        if len(net_new) > max_files:
            hit_file_cap = True
            net_new = net_new[:max_files]
        candidates = net_new
    return ScanResult(
        candidates=candidates,
        inspected=inspected,
        hit_entry_cap=hit_entry_cap,
        hit_file_cap=hit_file_cap,
        root_missing=False,
    )


@dataclass(frozen=True)
class DirEntry:
    """One browsable child directory in the folder browser (issue #63)."""

    name: str  # leaf display name (the directory's own name)
    rel: str  # MEDIA_ROOT-relative POSIX path (what an Add would register)
    registered: bool  # already in media_folders


@dataclass(frozen=True)
class BrowseListing:
    """A single directory's browsable contents for the folder browser (issue #63).

    Only immediate CHILD directories are listed — the browser registers folders,
    not files, so files never appear. Every field is derived under a containment
    check against MEDIA_ROOT; a client path that escapes, is reserved, or is not a
    directory recovers to the root listing with ``invalid_path=True`` (an honest
    "that folder isn't available" signal, never a silent snap and never echoing the
    submitted path). ``root_missing`` is reserved for MEDIA_ROOT itself being absent.
    """

    current: str  # "." at the root, else the browsed dir's relative POSIX path
    current_registered: bool  # the browsed dir itself is in media_folders
    parent: str | None  # parent's relative POSIX path; None at the root
    breadcrumbs: list[tuple[str, str]]  # (label, rel) from root → current
    entries: list[DirEntry]  # child dirs, sorted by name, ≤ MAX_BROWSE_ENTRIES
    truncated: bool  # more children existed than the cap shows
    invalid_path: bool  # the submitted path was bad → recovered to a valid dir
    root_missing: bool  # MEDIA_ROOT itself is absent / not a directory


def _breadcrumbs(rel: str) -> list[tuple[str, str]]:
    """(label, rel) crumbs from the media root to ``rel`` (a normalized POSIX path).

    ``"."`` yields a single root crumb; ``"a/b"`` yields root, ``a``, ``a/b``.
    """
    crumbs: list[tuple[str, str]] = [("media root", ".")]
    if rel == ".":
        return crumbs
    parts = rel.split("/")
    for i, part in enumerate(parts):
        crumbs.append((part, "/".join(parts[: i + 1])))
    return crumbs


def list_media_subdirs(
    media_root: Path, rel_path: str, registered: set[str]
) -> BrowseListing:
    """List the immediate child directories of one folder under ``media_root``.

    Backs the issue #63 folder browser. ``rel_path`` is an operator-navigated,
    UNTRUSTED MEDIA_ROOT-relative path: it is resolved and containment-checked on
    every call (``resolve()`` + :meth:`Path.is_relative_to`), and a path that
    escapes the root, lands on a reserved Voxint tree (``incoming``/``artifacts``),
    or is not an existing directory recovers to the root listing with
    ``invalid_path=True`` rather than raising or disclosing anything outside the
    root. Only immediate child directories are returned (sorted by name, capped at
    :data:`MAX_BROWSE_ENTRIES`); symlinks — files and dirs — are never followed or
    listed, matching :func:`scan_media_folders`. A missing media root yields
    ``root_missing=True``. ``registered`` is the set of already-watched folders
    (MEDIA_ROOT-relative POSIX) used to mark each entry and the current directory.
    """
    root = media_root.resolve()
    if not root.is_dir():
        return BrowseListing(
            current=".",
            current_registered="." in registered,
            parent=None,
            breadcrumbs=_breadcrumbs("."),
            entries=[],
            truncated=False,
            invalid_path=False,
            root_missing=True,
        )

    reserved = {(root / name).resolve() for name in _RESERVED_TREES}
    invalid_path = False
    target = (root / rel_path).resolve() if rel_path else root
    # Recover (not raise) to the root on any bad client path — a traversal attempt,
    # a stale/removed folder, a reserved tree, or a non-directory — so navigation
    # stays safe and honest without echoing the untrusted submitted value.
    if (
        not target.is_relative_to(root)
        or _under_reserved(target, reserved)
        or not target.is_dir()
    ):
        invalid_path = rel_path not in ("", ".")
        target = root

    rel = target.relative_to(root).as_posix()  # "." when target IS the root
    parent = target.parent.relative_to(root).as_posix() if rel != "." else None

    entries: list[DirEntry] = []
    truncated = False
    try:
        scanner = os.scandir(target)
    except OSError:
        scanner = None  # unreadable/vanished after the containment check — empty
    if scanner is not None:
        with scanner:
            collected: list[DirEntry] = []
            for entry in scanner:
                try:
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        continue
                    child = Path(entry.path).resolve()
                except OSError:
                    continue  # a single flaky entry never fails the whole listing
                if not child.is_relative_to(root) or child in reserved:
                    continue
                child_rel = child.relative_to(root).as_posix()
                collected.append(
                    DirEntry(
                        name=entry.name,
                        rel=child_rel,
                        registered=child_rel in registered,
                    )
                )
        collected.sort(key=lambda e: e.name)
        if len(collected) > MAX_BROWSE_ENTRIES:
            truncated = True
            collected = collected[:MAX_BROWSE_ENTRIES]
        entries = collected

    return BrowseListing(
        current=rel,
        current_registered=rel in registered,
        parent=parent,
        breadcrumbs=_breadcrumbs(rel),
        entries=entries,
        truncated=truncated,
        invalid_path=invalid_path,
        root_missing=False,
    )
