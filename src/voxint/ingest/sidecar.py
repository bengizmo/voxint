"""YAML sidecar metadata for watch-folder media (issue #104).

A media file dropped into a watched folder may arrive with a companion YAML
sidecar — ``interview.wav.yaml`` (the full-name form) or ``interview.yaml``
(the stem form; the full-name form wins when both exist) — whose fields feed
the run at submit time. This module is the pure layer: parsing, validation,
and filename pairing, with no database or submit-path coupling, so the CLI
and upload paths can adopt it after v1 (watch-folder only).

Machine-read keys (applied strictly — a bad type or value HOLDS the file):
``title`` (display name), ``speakers`` (list of names, unioned into the
run's frozen domain-pack ``name_seeds``), ``domain_pack`` (pack name),
``notes`` (operator notes). Every other key is ignored for application but
preserved verbatim in the stored snapshot (``Sidecar.raw``), so a sidecar
can carry reference-only context from other tooling without being rejected.
A ``description`` key is deliberately NOT applied to notes — notes are for
deliberate operator remarks.

Every failure raises :class:`SidecarError` with an operator-facing
plain-language message (there is no internal-manifest audience here, unlike
domain packs, so no two-tier message rewrite is needed). The caller (the
watch sweep) holds the media un-submitted and retries next sweep — a
malformed sidecar must never silently drop or half-ingest a recording.
"""

from __future__ import annotations

import json
import os
import stat
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import yaml

from voxint.media.suffixes import MEDIA_SUFFIXES

# The one recognized sidecar extension. ``.yml`` is deliberately not accepted
# (mirrors the domain packs' single ``manifest.yaml`` spelling): one spelling
# to document, and no two-sidecars-for-one-file ambiguity between them.
SIDECAR_SUFFIX = ".yaml"

# Bounds are rejected, never truncated (the corrections.py posture): a sidecar
# that exceeds one is a visible error the operator must see and fix.
MAX_SIDECAR_BYTES = 65_536
MAX_TITLE_CHARS = 300
MAX_SPEAKERS = 64
MAX_SPEAKER_CHARS = 120
MAX_NOTES_CHARS = 10_000
MAX_PACK_NAME_CHARS = 120

# Budgets for normalizing the WHOLE mapping into a JSON-safe snapshot. YAML
# aliases/anchors can expand far beyond the input byte cap (a shared list
# referenced N times parses small but traverses large), and can even build
# self-referential structures — so traversal is bounded by depth, total node
# count, and the final serialized size, with path-based cycle detection.
_MAX_DEPTH = 32
_MAX_NODES = 10_000
_MAX_SNAPSHOT_BYTES = 4 * MAX_SIDECAR_BYTES

_APPLIED_KEYS = frozenset({"title", "speakers", "domain_pack", "notes"})

# Same rejected categories as domain-pack corrections: Cc control, Cf format
# (zero-width/bidi), Cs surrogate — invisible or unserializable characters have
# no place in a title/name/pack-name an operator will read back.
_NON_PRINTING_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


class SidecarError(Exception):
    """A sidecar file could not be applied.

    The message is written FOR the operator (plain language, names the file,
    states the fix where one is obvious); the watch sweep logs it and holds
    the media file un-submitted until the sidecar is fixed.
    """


@dataclass(frozen=True)
class Sidecar:
    """One parsed, validated sidecar.

    ``raw`` is the whole mapping normalized to JSON-safe values — stored
    write-once on the run for provenance, so reference-only keys from other
    tooling survive intact. ``ignored_keys`` lists (sorted) the keys present
    but not machine-read, for the sweep log.
    """

    title: str | None
    speakers: tuple[str, ...]
    domain_pack: str | None
    notes: str | None
    raw: dict[str, Any]
    ignored_keys: tuple[str, ...]


# --- YAML loading (strict) -----------------------------------------------------


class _StrictKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    Stock PyYAML keeps the LAST duplicate silently — for a sidecar that would
    quietly drop the operator's first value, so duplicates fail the parse
    instead (surfacing through the normal YAMLError channel).
    """


def _construct_mapping_no_duplicates(
    loader: _StrictKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            # Stock SafeLoader expands `<<:` merge keys via flatten_mapping;
            # this constructor deliberately does not, and letting the tag fall
            # through would produce a cryptic "could not determine a
            # constructor" error. Refuse plainly instead: merged-in keys with
            # explicit overrides would also complicate the duplicate rule.
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "merge keys (<<) are not supported in a sidecar; "
                "write each key out on its own line",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in mapping
        except TypeError as exc:  # unhashable key (a list/mapping used as a key)
            raise yaml.constructor.ConstructorError(
                None, None, f"unusable mapping key {key!r}", key_node.start_mark
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_duplicates
)


# --- parsing -------------------------------------------------------------------


def parse_sidecar(text: str, *, source_name: str) -> Sidecar:
    """Parse and validate one sidecar document (pure — no I/O).

    ``source_name`` is the filename used in error messages. A blank or
    comment-only document is a VALID empty sidecar (a stub applies nothing and
    holds nothing); an empty mapping and a mapping of only unrecognized keys
    are equally valid — the reference-only export use case.
    """
    try:
        data = yaml.load(text, Loader=_StrictKeyLoader)
    except yaml.YAMLError as exc:
        raise SidecarError(
            f"{source_name} is not valid YAML{_yaml_error_detail(exc)}"
        ) from exc
    except RecursionError as exc:
        raise SidecarError(f"{source_name} is not valid YAML: nested too deeply") from exc
    except (ValueError, OverflowError) as exc:
        # PyYAML's scalar constructors raise plain ValueError for values that
        # look valid but cannot be built (a date like 2020-99-99, an integer
        # past the int-str digit limit). Escaping here would crash the whole
        # recurring sweep over one file — convert to the normal hold instead.
        raise SidecarError(
            f"{source_name} contains a value that could not be read: {exc}"
        ) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SidecarError(
            f"{source_name} must be a mapping of keys to values (like "
            f"'title: My interview'), not {_type_word(data)}"
        )
    raw = _snapshot(data, source_name)
    title = _single_line_field(data, "title", MAX_TITLE_CHARS, source_name)
    domain_pack = _single_line_field(data, "domain_pack", MAX_PACK_NAME_CHARS, source_name)
    notes = _notes_field(data, source_name)
    speakers = _speakers_field(data, source_name)
    ignored = tuple(sorted(str(key) for key in data if key not in _APPLIED_KEYS))
    return Sidecar(
        title=title,
        speakers=speakers,
        domain_pack=domain_pack,
        notes=notes,
        raw=raw,
        ignored_keys=ignored,
    )


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    """A short, content-free description of a YAML parse failure.

    ``str(exc)`` on a marked error embeds a snippet of the source buffer;
    hold reasons are logged, and the sweep's contract is to log reasons and
    key names, never sidecar VALUES — so only the problem phrase and the
    line/column are surfaced.
    """
    problem = getattr(exc, "problem", None)
    mark = getattr(exc, "problem_mark", None)
    detail = f": {problem}" if problem else ""
    if mark is not None:
        detail += f" (line {mark.line + 1}, column {mark.column + 1})"
    return detail


def _type_word(value: Any) -> str:
    """A lay-reader word for a wrong YAML type ('a list', 'a number', ...)."""
    if isinstance(value, bool):
        return "a yes/no value"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, (date, datetime, time)):
        return "a date"
    if value is None:
        return "empty"
    if isinstance(value, str):
        return "text"
    return f"a {type(value).__name__}"


def _single_line_field(
    data: dict[Any, Any], key: str, max_chars: int, source_name: str
) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise SidecarError(
            f"{source_name}: '{key}' must be text, not {_type_word(value)} — "
            f"remove the line if unused"
        )
    cleaned = value.strip()
    if not cleaned:
        raise SidecarError(
            f"{source_name}: '{key}' is empty — give it a value or remove the line"
        )
    if "\n" in cleaned or "\r" in cleaned:
        raise SidecarError(f"{source_name}: '{key}' must be a single line")
    if len(cleaned) > max_chars:
        raise SidecarError(
            f"{source_name}: '{key}' is longer than {max_chars} characters"
        )
    _reject_non_printing(cleaned, key, source_name)
    return cleaned


def _notes_field(data: dict[Any, Any], source_name: str) -> str | None:
    if "notes" not in data:
        return None
    value = data["notes"]
    if not isinstance(value, str):
        raise SidecarError(
            f"{source_name}: 'notes' must be text, not {_type_word(value)} — "
            f"remove the line if unused"
        )
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise SidecarError(
            f"{source_name}: 'notes' is empty — give it a value or remove the line"
        )
    if len(cleaned) > MAX_NOTES_CHARS:
        raise SidecarError(
            f"{source_name}: 'notes' is longer than {MAX_NOTES_CHARS} characters"
        )
    for ch in cleaned:
        if ch != "\n" and unicodedata.category(ch) in _NON_PRINTING_CATEGORIES:
            raise SidecarError(
                f"{source_name}: 'notes' contains a non-printing character "
                f"(U+{ord(ch):04X})"
            )
    return cleaned


def _speakers_field(data: dict[Any, Any], source_name: str) -> tuple[str, ...]:
    if "speakers" not in data:
        return ()
    value = data["speakers"]
    if not isinstance(value, list):
        raise SidecarError(
            f"{source_name}: 'speakers' must be a list of names (one '- Name' "
            f"per line), not {_type_word(value)}"
        )
    if len(value) > MAX_SPEAKERS:
        raise SidecarError(
            f"{source_name}: 'speakers' lists {len(value)} names; the maximum "
            f"is {MAX_SPEAKERS}"
        )
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise SidecarError(
                f"{source_name}: speaker {index + 1} must be text, not "
                f"{_type_word(item)}"
            )
        cleaned = item.strip()
        if not cleaned:
            raise SidecarError(
                f"{source_name}: speaker {index + 1} is empty — remove it or "
                f"give it a name"
            )
        if "\n" in cleaned or "\r" in cleaned:
            raise SidecarError(
                f"{source_name}: speaker {index + 1} must be a single line"
            )
        if len(cleaned) > MAX_SPEAKER_CHARS:
            raise SidecarError(
                f"{source_name}: speaker {index + 1} is longer than "
                f"{MAX_SPEAKER_CHARS} characters"
            )
        _reject_non_printing(cleaned, f"speaker {index + 1}", source_name)
        names.append(cleaned)
    return tuple(names)


def _reject_non_printing(text: str, label: str, source_name: str) -> None:
    for ch in text:
        if unicodedata.category(ch) in _NON_PRINTING_CATEGORIES:
            raise SidecarError(
                f"{source_name}: '{label}' contains a non-printing character "
                f"(U+{ord(ch):04X})"
            )


# --- snapshot normalization ------------------------------------------------------


def _snapshot(data: dict[Any, Any], source_name: str) -> dict[str, Any]:
    """Normalize the whole mapping to a JSON-safe dict for JSONB storage.

    Ordinary safe-YAML values are normalized (dates to ISO strings, unknown
    scalars stringified, non-string keys stringified) so a reference-only
    export survives intact; structures that cannot be normalized
    deterministically — cycles, alias expansion past the node budget, depth
    past the budget, stringified-key collisions, non-finite numbers — are
    errors, not best-effort guesses.
    """
    budget = [_MAX_NODES]
    normalized = _json_safe(data, source_name, depth=0, path=[], budget=budget)
    assert isinstance(normalized, dict)  # the root was a dict going in
    try:
        encoded = json.dumps(normalized, allow_nan=False)
    except ValueError as exc:
        raise SidecarError(
            f"{source_name} contains a value that cannot be stored: {exc}"
        ) from exc
    if len(encoded.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise SidecarError(
            f"{source_name} expands to more data than can be stored — remove "
            f"repeated blocks (YAML anchors/aliases) or shrink it"
        )
    return normalized


def _json_safe(
    value: Any, source_name: str, *, depth: int, path: list[int], budget: list[int]
) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise SidecarError(
            f"{source_name} contains too many values to store — remove "
            f"repeated blocks (YAML anchors/aliases) or shrink it"
        )
    if depth > _MAX_DEPTH:
        raise SidecarError(f"{source_name} is nested too deeply to store")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SidecarError(
                f"{source_name} contains a non-finite number (.nan/.inf), "
                f"which cannot be stored"
            )
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        if id(value) in path:
            raise SidecarError(f"{source_name} contains a value that refers to itself")
        path.append(id(value))
        try:
            # A YAML !!set has no order; sort it so the write-once snapshot is
            # deterministic across processes (hash randomization would
            # otherwise freeze a different order per run).
            items = sorted(value, key=repr) if isinstance(value, set) else value
            return [
                _json_safe(item, source_name, depth=depth + 1, path=path, budget=budget)
                for item in items
            ]
        finally:
            path.pop()
    if isinstance(value, dict):
        if id(value) in path:
            raise SidecarError(f"{source_name} contains a value that refers to itself")
        path.append(id(value))
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                text_key = key if isinstance(key, str) else str(key)
                if text_key in result:
                    raise SidecarError(
                        f"{source_name} has two keys that both read as "
                        f"'{text_key}' — rename one of them"
                    )
                result[text_key] = _json_safe(
                    item, source_name, depth=depth + 1, path=path, budget=budget
                )
            return result
        finally:
            path.pop()
    # Any other safe-YAML scalar (e.g. !!binary bytes): keep a string trace
    # rather than rejecting a reference-only value outright.
    return str(value)


# --- file reading -----------------------------------------------------------------


def read_sidecar(path: Path) -> Sidecar:
    """Read and parse one sidecar file, defensively.

    Refuses symlinks and non-regular files, caps the read at
    :data:`MAX_SIDECAR_BYTES`, requires strict UTF-8, and re-stats the open
    file after the read so a sidecar replaced or grown mid-read is held (and
    retried next sweep) rather than half-parsed.
    """
    name = path.name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SidecarError(
            f"{name} could not be read: {exc.strerror or exc}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SidecarError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        remaining = MAX_SIDECAR_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
    except OSError as exc:
        raise SidecarError(
            f"{name} could not be read: {exc.strerror or exc}"
        ) from exc
    finally:
        os.close(fd)
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise SidecarError(
            f"{name} changed while it was being read; it will be retried on "
            f"the next check"
        )
    # The fd-based compare above cannot see an atomic rename over the path
    # (the fd stays on the OLD inode) — the normal editor/exporter save
    # pattern. Re-stat the PATH and hold on any divergence, so the next sweep
    # reads the replacement instead of freezing stale metadata.
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise SidecarError(
            f"{name} was removed while it was being read; it will be "
            f"re-checked on the next sweep"
        ) from exc
    if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
        raise SidecarError(
            f"{name} was replaced while it was being read; it will be "
            f"retried on the next check"
        )
    payload = b"".join(chunks)
    if len(payload) > MAX_SIDECAR_BYTES:
        raise SidecarError(
            f"{name} is larger than {MAX_SIDECAR_BYTES // 1024} KiB — a "
            f"sidecar should be a small text file"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SidecarError(f"{name} is not UTF-8 text: {exc.reason}") from exc
    return parse_sidecar(text, source_name=name)


# --- pairing -----------------------------------------------------------------------


def find_sidecar(media_path: Path) -> Path | None:
    """Locate the sidecar for ``media_path``, or ``None`` when there is none.

    The full-name form (``clip.wav.yaml``) wins and, once present, is FINAL —
    a broken full-name sidecar is returned for :func:`read_sidecar` to hold,
    never silently bypassed for the stem form. The stem form (``clip.yaml``)
    only applies when exactly one media file in the directory carries that
    stem: with a second same-stem media file present (even an already-ingested
    one — a stem sidecar beside ``clip.wav`` and ``clip.mp4`` was plausibly
    written for either, and misapplying metadata is worse than waiting) the
    sidecar is ambiguous and the error names the fix. An unreadable directory
    is an error too — "cannot verify" must hold, not guess.
    """
    directory = media_path.parent
    full_form = directory / (media_path.name + SIDECAR_SUFFIX)
    if _lexists(full_form, media_path.name):
        return full_form
    stem = media_path.stem
    stem_form = directory / (stem + SIDECAR_SUFFIX)
    if not _lexists(stem_form, media_path.name):
        return None
    # Lazy iteration (the scan_media_folders posture): the check raises on the
    # first ambiguous sibling, so materializing a huge directory buys nothing.
    # Stems compare case-folded — the suffix check already is, and on the
    # ambiguity side more conservative is safer (Clip.mp4 beside clip.wav is a
    # human-identical stem even on a case-sensitive filesystem).
    folded_stem = stem.casefold()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name == media_path.name:
                    continue
                candidate = Path(entry.name)
                if candidate.stem.casefold() != folded_stem:
                    continue
                if candidate.suffix.lower() not in MEDIA_SUFFIXES:
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                raise SidecarError(
                    f"{stem_form.name} could apply to more than one recording "
                    f"({media_path.name} and {entry.name}) — rename it to "
                    f"{media_path.name}{SIDECAR_SUFFIX} to say which one it "
                    f"describes"
                )
    except OSError as exc:
        raise SidecarError(
            f"{stem_form.name} could not be checked against the folder's other "
            f"files: {exc.strerror or exc}"
        ) from exc
    return stem_form


def _lexists(path: Path, media_name: str) -> bool:
    """Whether ``path`` exists as a directory entry (symlinks included, not
    followed) — a present-but-broken sidecar must be SEEN so it holds its
    media file, not skipped. Permission failures hold too."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SidecarError(
            f"{path.name} (next to {media_name}) could not be checked: "
            f"{exc.strerror or exc}"
        ) from exc
    return True
