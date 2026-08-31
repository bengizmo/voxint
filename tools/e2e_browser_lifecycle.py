#!/usr/bin/env python3
"""Canonical lifecycle for the browser E2E acceptance lane (Phase 3).

This is the single source of truth for the review-console browser lane: it
builds and stages the islands, seeds a disposable database with a COMPLETED run
shaped for the review loop, serves a throwaway instance, and — after a browser
has driven the UI — reconciles the *durable* state the browser was the sole
writer of. The interaction layer (Playwright MCP, driven by the
``voxint-e2e-review`` skill) is deliberately NOT here: seed + reconcile are
tool-neutral so the same lane survives swapping the driver for a Python
Playwright harness later.

Why a lane at all: two review-console island behaviours (#53/#58) — the
verify-and-advance loop, click-to-edit, the unsaved-edit discard warning, and
the keymap suppression on focused form controls — are runtime-only; every
server-side gate is green while the island misbehaves. This lane makes that
class gate-able and reproducible instead of eyeballed.

Subcommands (run under ``uv`` — this imports ``voxint``):

    uv run python tools/e2e_browser_lifecycle.py setup
    uv run python tools/e2e_browser_lifecycle.py seed  --database-url <dsn> [--create-db]
    uv run python tools/e2e_browser_lifecycle.py serve --database-url <dsn>  # backgrounded
    uv run python tools/e2e_browser_lifecycle.py reconcile --database-url <dsn> \
        --run-id <uuid> --expect-file <expectation.json>
    uv run python tools/e2e_browser_lifecycle.py teardown [--drop-db --database-url <dsn>]

The database URL must name a DISPOSABLE database (its ``public`` schema is
dropped and rebuilt from the alembic chain). The live database is named
``voxint`` and is refused — every DB-touching subcommand fails closed on a name
without ``test`` or ``e2e``. Host-specific invocation (which maintainer box,
which ports) stays in a maintainer-only runbook, never in this public repo.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import verified_progress
from voxint.app_settings import complete_onboarding
from voxint.clients.llm import enhanced_size_ceiling
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    SegmentSplitBoundary,
    Speaker,
    TranscriptAnnotation,
    TranscriptSegment,
)
from voxint.domain_packs.corrections import parse_corrections
from voxint.domain_packs.corrector import CORRECTOR_VERSION, apply_corrections

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONTEND_DIR = REPO_ROOT / "frontend"
DEFAULT_STATIC_DIR = REPO_ROOT / "src" / "voxint" / "api" / "static" / "app"
GITKEEP = ".gitkeep"  # the only tracked entry under the static app dir
DEFAULT_PORT = 8099
DEFAULT_MEDIA_DIR = REPO_ROOT / "media-e2e"

# The seeded transcript: varied confidence so the review loop has both certain
# and "uncertain"-chip (< the 0.6 low-confidence default) segments, and one NULL
# (never flagged). Duration is derived from the segment spans so playback follows.
_SEGMENT_SECONDS = 5.0
_SEED_SEGMENTS: tuple[tuple[str, str, float | None], ...] = (
    ("S0", "Good morning everyone, thanks for joining.", 0.95),
    ("S1", "Happy to be here with you all today.", 0.42),  # uncertain
    ("S0", "Let us start by reviewing last week.", 0.88),
    ("S1", "I had one question about the numbers.", 0.31),  # uncertain
    ("S0", "Of course, go right ahead and ask.", None),  # no confidence → never flagged
)

# Deterministic domain-pack correction provenance (issue #83) for the browser lane.
# The run freezes this ONE pack; the browser lane asserts the "corrected by domain
# pack" marker, the raw-compare affordance, and the run-level reconciliation panel.
# `greet` fires on segment 0's raw ("everyone" → applied, appliedCount 1); `ghost`
# matches no segment's raw ("declared but never fired" → no_raw_match) — so the
# reconciliation panel shows both statuses. A high-confidence, non-uncertain segment
# is corrected (index 0) so the two `.tp-uncertain-chip` assertions on segments 1 & 3
# stay untouched. Kept as an unmistakable-but-plausible edit so the raw-vs-corrected
# compare is visibly different.
_E2E_PACK_NAME = "e2e-corrections"
_E2E_CORRECTIONS: tuple[dict[str, str], ...] = (
    {"id": "greet", "match": "everyone", "replace": "everybody"},
    {"id": "ghost", "match": "quarterly synergies", "replace": "Q3 results"},
)
_CORRECTED_SEGMENT_INDEX = 0

# ---------------------------------------------------------------------------
# Editor fixture (30 segments, 4 speakers) — Phase 6a (#157).
# A meeting between S0 (moderator), S1-S3 (participants). Varied confidence,
# long texts for split testing, correction provenance on segment 0 (same
# "everyone" → "everybody" rule fires). Used by `--fixture editor`.
# ---------------------------------------------------------------------------
_EDITOR_SEGMENTS: tuple[tuple[str, str, float | None], ...] = (
    ("S0", "Welcome everyone, let us get started with the weekly sync.", 0.93),
    ("S1", "Thanks for having me.", 0.89),
    ("S0", "First item on the agenda is the progress report from last week.", 0.91),
    ("S2", "I can cover that.", 0.38),
    (  # long, split-eligible
        "S2",
        "We shipped the database migration on Tuesday and ran the backfill"
        " overnight which completed without errors by Wednesday morning.",
        0.87,
    ),
    ("S1", "That was faster than expected.", 0.44),
    ("S3", "I have a question about the migration.", 0.92),
    (  # long, split-eligible
        "S3",
        "Did we validate the row counts after the backfill finished"
        " or did we just check the exit code?",
        0.85,
    ),
    (
        "S2",
        "We validated both the row counts and ran a checksum"
        " comparison against the source tables.",
        0.90,
    ),
    ("S0", "Good.", 0.96),
    ("S1", "Moving on.", None),
    ("S0", "Next item is the deployment timeline for the new feature.", 0.88),
    (  # long, split-eligible
        "S3",
        "We are targeting Thursday for staging and Friday for production"
        " if the smoke tests pass.",
        0.82,
    ),
    ("S1", "I think we should add an extra day of buffer.", 0.35),
    ("S2", "Agreed.", 0.91),
    ("S0", "Let us plan for Thursday staging and Monday production then.", 0.94),
    ("S3", "Works for me.", 0.87),
    ("S1", "Same here.", 0.42),
    ("S0", "Third item is the customer feedback from the beta program.", 0.89),
    (  # long, split-eligible
        "S2",
        "The main feedback themes were around the onboarding flow and the"
        " search functionality both of which came up in multiple sessions.",
        0.86,
    ),
    (  # long, split-eligible
        "S3",
        "The search complaints were specifically about relevance ranking"
        " and missing results for partial matches.",
        0.83,
    ),
    ("S1", "We saw similar patterns in the support tickets.", 0.39),
    ("S0", "What is the proposed fix?", 0.95),
    (  # long, split-eligible
        "S2",
        "We are planning to switch from exact match to fuzzy matching"
        " with a relevance threshold.",
        0.88,
    ),
    ("S3", "That should cover most of the reported cases.", 0.84),
    ("S1", "Can we get metrics on the impact?", None),
    (
        "S0",
        "Yes let us track the search success rate before and after the change.",
        0.92,
    ),
    ("S2", "I will set up the dashboard this week.", 0.30),
    ("S3", "I can help with the A B testing framework if needed.", 0.41),
    ("S0", "Great, that wraps up the agenda for today.", 0.97),
)

# Indexes of editor segments that carry word timings (split-eligible).
# Must NOT include _CORRECTED_SEGMENT_INDEX (correction trace blocks splitting).
_EDITOR_SPLIT_ELIGIBLE: frozenset[int] = frozenset({4, 7, 12, 19, 20, 23})

FIXTURE_CHOICES = ("review", "editor", "benchmark")

_FIXTURE_SEGMENTS: dict[str, tuple[tuple[str, str, float | None], ...]] = {
    "review": _SEED_SEGMENTS,
    "editor": _EDITOR_SEGMENTS,
}


def _benchmark_segments(count: int = 2000) -> tuple[tuple[str, str, float | None], ...]:
    """Generate ``count`` segments by cycling the editor fixture with varied confidence."""
    base = _EDITOR_SEGMENTS
    result: list[tuple[str, str, float | None]] = []
    for i in range(count):
        _label, text, conf = base[i % len(base)]
        speaker = f"S{i % 6}"
        if conf is not None:
            conf = round(max(0.1, min(0.99, conf + (i % 7 - 3) * 0.05)), 2)
        result.append((speaker, f"[{i}] {text}", conf))
    return tuple(result)


def _faithful_word_timings(
    raw_text: str, start: float, end: float
) -> list[dict[str, object]]:
    """Deterministic word timings faithful to ``raw_text``.

    Splits on whitespace and distributes time evenly. The joined word texts
    reconstruct ``raw_text`` exactly (leading space on each token after the
    first preserves inter-word whitespace), satisfying ``splittable_words``.
    """
    tokens = raw_text.split(" ")
    duration = end - start
    step = duration / len(tokens)
    words: list[dict[str, object]] = []
    for i, tok in enumerate(tokens):
        w_start = round(start + i * step, 4)
        w_end = round(start + (i + 1) * step, 4)
        word_text = f" {tok}" if i > 0 else tok
        words.append({"word": word_text, "start": w_start, "end": w_end})
    return words


def fail(msg: str) -> None:
    print(f"LIFECYCLE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Disposable-database guard (mirrors tests/e2e/conftest.py::_assert_disposable_db)
# --------------------------------------------------------------------------- #
def assert_disposable_db(url: str) -> None:
    """Raise ValueError unless ``url`` names an unmistakably throwaway database.

    Every DB-touching subcommand drops and rebuilds the ``public`` schema (seed)
    or the whole database (teardown --drop-db). Pointed at the live ``voxint``
    database, a copy-pasted DSN would destroy operator data. Fail closed: the
    database name must contain ``test`` or ``e2e``.

    The guard inspects the URL *path*, but psycopg lets a ``dbname``/``database``
    query parameter OVERRIDE the path at connect time — so ``…/voxint_e2e?dbname=
    voxint`` would pass a path-only check yet connect to the live DB (codex+kimi
    review). Reject any database-selecting query key outright, and restrict the
    name to a plain identifier so the disposable check cannot be smuggled past.
    """
    parts = urlsplit(url)
    selecting = {"dbname", "database"} & {k.lower() for k in parse_qs(parts.query)}
    if selecting:
        raise ValueError(
            f"database URL must not select the database via a query parameter "
            f"({', '.join(sorted(selecting))}); it can override the path and point "
            "destructive operations at the live database. Put the database in the path."
        )
    db_name = parts.path.lstrip("/").lower()
    if not re.fullmatch(r"[a-z0-9_]+", db_name or ""):
        raise ValueError(
            f"database name {db_name!r} is not a plain identifier ([a-z0-9_]); "
            "refusing to build destructive DDL around it."
        )
    if "test" not in db_name and "e2e" not in db_name:
        raise ValueError(
            "database URL must name a DISPOSABLE database whose name contains "
            f"'test' or 'e2e' (its schema is dropped and rebuilt); got {db_name!r}. "
            "Refusing to run destructive setup against a database that could be live."
        )


def _guarded(url: str | None) -> str:
    if not url:
        fail("--database-url is required (a disposable DSN containing 'test' or 'e2e').")
    assert url is not None
    try:
        assert_disposable_db(url)
    except ValueError as exc:
        fail(str(exc))
    return url


# --------------------------------------------------------------------------- #
# setup — build + stage the islands
# --------------------------------------------------------------------------- #
def stage_build(frontend_dir: Path, static_dir: Path) -> list[str]:
    """Build the islands and overlay ``dist/`` onto the served static dir.

    The app reads the Vite manifest once at import, so the bundles must be in
    place BEFORE ``serve``. Returns the top-level names copied in, for logging.
    (``teardown`` does not consume this list — ``unstage_build`` removes every
    entry except the tracked ``.gitkeep``, which is the git-clean outcome we
    want even if a prior run left something behind.) ``.gitkeep`` is never touched.
    """
    dist = frontend_dir / "dist"
    subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)
    if not dist.is_dir():
        fail(f"npm build produced no dist/ at {dist}")
    static_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for child in sorted(dist.iterdir()):
        if child.name == GITKEEP:
            continue
        target = static_dir / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
        staged.append(child.name)
    return staged


def cmd_setup(args: argparse.Namespace) -> None:
    staged = stage_build(Path(args.frontend_dir), Path(args.static_dir))
    print(f"ok: built islands and staged {staged} into {args.static_dir}")
    print("SETUP PASS")


# --------------------------------------------------------------------------- #
# seed — disposable DB with a COMPLETED review run
# --------------------------------------------------------------------------- #
def _silent_wav_bytes(seconds: float) -> bytes:
    """Quiet constant-amplitude PCM (NOT pure silence): the waveform strip
    (issue #57) draws an amplitude envelope, and all-zero peaks would make the
    seeded strip an unreviewable flat line. A small DC value keeps the file
    trivially cheap to generate while giving every peak bucket a visible bar."""
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((4096).to_bytes(2, "little", signed=True) * frames)
    return buf.getvalue()


def seed_browser_run(
    session: Session,
    media_root: Path,
    *,
    fixture: str = "review",
) -> tuple[uuid.UUID, uuid.UUID]:
    """A COMPLETED run shaped for the review/editor loop.

    Returns ``(run_id, media_id)``. ``fixture`` selects the segment set:
    ``"review"`` (5 segments, backward compat), ``"editor"`` (30 segments,
    4 speakers, split-eligible text), or ``"benchmark"`` (2000 segments).
    """
    if fixture not in FIXTURE_CHOICES:
        raise ValueError(
            f"unknown fixture {fixture!r}; must be one of {FIXTURE_CHOICES}"
        )
    segments = (
        _benchmark_segments() if fixture == "benchmark"
        else _FIXTURE_SEGMENTS[fixture]
    )
    seg_seconds = 0.5 if fixture == "benchmark" else _SEGMENT_SECONDS
    duration = seg_seconds * len(segments)
    media = MediaItem(
        source_path=f"e2e/{uuid.uuid4().hex}.wav",
        duration_seconds=duration,
    )
    session.add(media)
    session.flush()
    # Freeze the #83 provenance pack onto the run's snapshot column (the same shape
    # the pipeline persists: a dict with `name` + `corrections`). `_load_run_rule_index`
    # reads this dict DIRECTLY, so the console resolves fired rules and reconciles
    # declared-but-never-fired ones exactly as it would for a real run.
    run = PipelineRun(
        media_item_id=media.id,
        status=RunStatus.COMPLETED.value,
        domain_pack={"name": _E2E_PACK_NAME, "corrections": list(_E2E_CORRECTIONS)},
    )
    session.add(run)
    session.flush()

    # Precompute the corrected segment's provenance with the REAL corrector, exactly
    # as the raw-pass path in enhance_match does (LLM off ⇒ input_base "raw"), so the
    # seeded envelope is byte-faithful — never hand-rolled. Self-check: the seed rule
    # MUST materially fire, or the lane would assert a marker that never renders.
    correction_rules = parse_corrections([dict(c) for c in _E2E_CORRECTIONS])
    corrected_raw = segments[_CORRECTED_SEGMENT_INDEX][1]
    corrected = apply_corrections(
        corrected_raw,
        correction_rules,
        max_output_chars=enhanced_size_ceiling(corrected_raw),
    )
    if not corrected.trace or corrected.growth_rejected:
        raise AssertionError(
            "e2e seed: the #83 correction rule did not materially fire on segment "
            f"{_CORRECTED_SEGMENT_INDEX}'s raw text — fix _E2E_CORRECTIONS so the "
            "browser lane has a real 'corrected by domain pack' marker to assert."
        )

    # A small ACTIVE roster so the browser lane can exercise the whole-segment
    # speaker assignment (issue #51: the 1-9 / 0 keys and the "Assign speaker"
    # <select>). These are curation identities only — no embeddings, so speaker
    # MATCHING still never runs; the segments keep their detected S0/S1 labels
    # and the roster is merely the assignable set the relabel endpoint accepts.
    for name in ("Ada Roster", "Blair Roster"):
        session.add(Speaker(display_name=name))
    session.flush()

    audio_rel = f"artifacts/{run.id}/normalized.wav"
    audio_abs = media_root / audio_rel
    audio_abs.parent.mkdir(parents=True, exist_ok=True)
    audio_abs.write_bytes(_silent_wav_bytes(duration))
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=audio_rel,
        )
    )

    for index, (label, seg_text, confidence) in enumerate(segments):
        segment = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=index,
            start_seconds=float(index) * seg_seconds,
            end_seconds=float(index) * seg_seconds + seg_seconds,
            raw_text=seg_text,
            diarization_label=label,
            confidence=confidence,
        )
        if index == _CORRECTED_SEGMENT_INDEX:
            segment.enhanced_text = corrected.text
            segment.correction_trace = {
                "version": CORRECTOR_VERSION,
                "input_base": "raw",
                "entries": [entry.to_mapping() for entry in corrected.trace],
            }
            segment.corrector_version = CORRECTOR_VERSION
        if fixture == "editor" and index in _EDITOR_SPLIT_ELIGIBLE:
            segment.words = _faithful_word_timings(
                seg_text, segment.start_seconds, segment.end_seconds
            )
        session.add(segment)
        # A matching diarization turn per segment (issue #57): the waveform
        # strip paints TURNS, so without these the seeded strip would render
        # bare gray bars and the region assertions would have nothing to test.
        # No embedding — the browser lane never runs speaker matching — so the
        # embedding-XOR-skip CHECK requires an honest skip_reason instead.
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index) * _SEGMENT_SECONDS,
                end_seconds=float(index) * _SEGMENT_SECONDS + _SEGMENT_SECONDS,
                label=label,
                skip_reason="e2e seed: embedding lane not exercised",
            )
        )
    session.commit()
    return run.id, media.id


def _admin_url(url: str) -> str:
    """The same server, pointed at the ``postgres`` maintenance database.

    The query is dropped as well as the path: a lingering ``dbname=`` there would
    redirect the "maintenance" connection back at a named database (the guard
    already rejects such query keys, but do not depend on that here).
    """
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/postgres", query=""))


def _create_database(url: str) -> None:
    """Create the disposable DB and the ``vector`` extension if absent (idempotent)."""
    db_name = urlsplit(url).path.lstrip("/")
    admin = create_engine(_admin_url(url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": db_name}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            print(f"ok: created database {db_name}")
    admin.dispose()
    target = create_engine(url, isolation_level="AUTOCOMMIT")
    with target.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    target.dispose()


def _reset_schema_and_migrate(url: str) -> None:
    eng = create_engine(url)
    with eng.connect() as conn:
        # Belt-and-suspenders over the URL guard: assert the database we actually
        # connected to is disposable BEFORE dropping its schema, so no query-param
        # or driver-kwarg override can land the DROP on the live DB (codex+kimi).
        live = conn.execute(text("SELECT current_database()")).scalar_one()
        assert_disposable_db(f"postgresql:///{live}")
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    eng.dispose()
    # alembic/env.py overrides sqlalchemy.url with get_settings().database_url,
    # which reads DATABASE_URL from the environment (over any .env) — so pin the
    # env var to the disposable DSN, exactly as tests/integration/conftest.py does.
    os.environ["DATABASE_URL"] = url
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.upgrade(cfg, "head")


def cmd_seed(args: argparse.Namespace) -> None:
    url = _guarded(args.database_url)
    media_root = Path(args.media_root)
    fixture = getattr(args, "fixture", "review")
    if args.create_db:
        _create_database(url)
    _reset_schema_and_migrate(url)
    engine = create_engine(url)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        complete_onboarding(session, llm_enabled_default=False)
        session.commit()
        run_id, media_id = seed_browser_run(session, media_root, fixture=fixture)
    engine.dispose()
    seg_count = len(_benchmark_segments()) if fixture == "benchmark" else len(
        _FIXTURE_SEGMENTS.get(fixture, _SEED_SEGMENTS)
    )
    print(f"ok: seeded COMPLETED {fixture} run with {seg_count} segments")
    print(f"RUN_ID={run_id}")
    print(f"MEDIA_ID={media_id}")
    print("SEED PASS")


# --------------------------------------------------------------------------- #
# serve — a throwaway instance from the working tree
# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> None:
    url = _guarded(args.database_url)
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": url,
            "MEDIA_ROOT": str(Path(args.media_root).resolve()),
            "API_HOST": "127.0.0.1",  # loopback — default-cred refusal is satisfied
            "API_PORT": str(args.port),
            "VOXINT_USER": args.user,
            "VOXINT_PASSWORD": args.password,
        }
    )
    Path(args.media_root).mkdir(parents=True, exist_ok=True)
    print(
        f"ok: serving working-tree instance on http://127.0.0.1:{args.port} "
        f"(user={args.user}); stop with: fuser -k {args.port}/tcp",
        flush=True,
    )
    # Replace this process so signals and `fuser -k <port>/tcp` reach uvicorn
    # directly (the skill backgrounds this call).
    os.execvpe("voxint", ["voxint", "serve"], env)


# --------------------------------------------------------------------------- #
# reconcile — fail-closed durable-state verifier (post-hoc)
# --------------------------------------------------------------------------- #
def _strict_index(value: object, what: str) -> int:
    """A non-negative integer, rejecting bool and float (JSON ``true``/``1.9``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class Expectation:
    """The durable outcome the browser was supposed to leave behind.

    ``verified_indexes`` — segment_index values whose review state must be
    verified (all others must NOT be). ``corrections`` — segment_index → exact
    corrected text (a segment absent here must carry no correction). ``progress``
    — the ``(verified, total)`` N-of-M counter the console showed.

    Editor-specific (Phase 6a, #157 — all optional, backward compatible):

    ``split_parent_indexes`` — segment_index values expected to have at least
    one ``SegmentSplitBoundary``; segments NOT listed here must have none.
    ``expected_annotations`` — exact ``TranscriptAnnotation`` count for the run
    (``None`` skips the check; default ``None``).
    """

    verified_indexes: frozenset[int]
    corrections: dict[int, str]
    progress: tuple[int, int]
    split_parent_indexes: frozenset[int] = frozenset()
    expected_annotations: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Expectation:
        # Validate strictly (never coerce): a malformed hand-written expectation
        # that silently coerces — bool→int, 1.9→1, null-correction→"None" — could
        # accidentally MATCH durable state and pass a fail-closed check that should
        # have failed (codex+kimi). This parser is the fail-closed gate's input.
        if not isinstance(data, dict):
            raise ValueError("expectation must be a JSON object")
        verified = data.get("verified_segment_indexes", [])
        if not isinstance(verified, list):
            raise ValueError("verified_segment_indexes must be a list of integers")
        indexes = [_strict_index(i, "a verified segment index") for i in verified]
        if len(set(indexes)) != len(indexes):
            raise ValueError("verified_segment_indexes contains duplicates")
        raw_corr = data.get("corrections", {})
        if not isinstance(raw_corr, dict):
            raise ValueError("corrections must be an object of index → text")
        corrections: dict[int, str] = {}
        for key, value in raw_corr.items():
            if not isinstance(value, str):
                raise ValueError(f"correction for {key!r} must be a string, got {value!r}")
            corrections[_strict_index(int(key), "a correction index")] = value
        prog = data.get("progress", {})
        if not isinstance(prog, dict) or "verified" not in prog or "total" not in prog:
            raise ValueError("progress must be an object with 'verified' and 'total'")
        raw_splits = data.get("split_parent_indexes", [])
        if not isinstance(raw_splits, list):
            raise ValueError("split_parent_indexes must be a list of integers")
        split_idxs = [_strict_index(i, "a split parent index") for i in raw_splits]
        if len(set(split_idxs)) != len(split_idxs):
            raise ValueError("split_parent_indexes contains duplicates")
        raw_annot = data.get("expected_annotations")
        if raw_annot is not None and (
            isinstance(raw_annot, bool) or not isinstance(raw_annot, int) or raw_annot < 0
        ):
            raise ValueError(
                "expected_annotations must be a non-negative integer"
                f" or null, got {raw_annot!r}"
            )
        return cls(
            verified_indexes=frozenset(indexes),
            corrections=corrections,
            progress=(
                _strict_index(prog["verified"], "progress.verified"),
                _strict_index(prog["total"], "progress.total"),
            ),
            split_parent_indexes=frozenset(split_idxs),
            expected_annotations=raw_annot,
        )


def reconcile_run(session: Session, run_id: uuid.UUID, expect: Expectation) -> list[str]:
    """Compare durable DB state against ``expect``; return human-readable problems.

    Pure (no I/O beyond the session, no exit) so it is unit-testable and the CLI
    owns the fail-closed exit. An empty list means the browser's writes match the
    expectation exactly — including that segments NOT expected verified/corrected
    carry no such state (the browser must not have written more than it drove).
    """
    problems: list[str] = []
    segments = (
        session.query(TranscriptSegment)
        .filter(TranscriptSegment.pipeline_run_id == run_id)
        .order_by(TranscriptSegment.segment_index)
        .all()
    )
    if not segments:
        return [f"no transcript segments found for run {run_id}"]
    by_index = {seg.segment_index: seg for seg in segments}

    unknown_verified = expect.verified_indexes - by_index.keys()
    if unknown_verified:
        problems.append(f"expected verified indexes not in run: {sorted(unknown_verified)}")
    unknown_corr = expect.corrections.keys() - by_index.keys()
    if unknown_corr:
        problems.append(f"expected corrected indexes not in run: {sorted(unknown_corr)}")

    for index, seg in by_index.items():
        state = session.get(SegmentReviewState, seg.id)
        is_verified = state is not None and state.verified_at is not None
        want_verified = index in expect.verified_indexes
        if is_verified != want_verified:
            problems.append(
                f"segment {index}: verified={is_verified}, expected {want_verified}"
            )
        got_corr = state.corrected_text if state is not None else None
        want_corr = expect.corrections.get(index)
        if got_corr != want_corr:
            problems.append(
                f"segment {index}: corrected_text={got_corr!r}, expected {want_corr!r}"
            )
        # The paired-shape CHECK guarantees corrected_at iff corrected_text; assert
        # the pairing held so a half-written row is caught here, not at export.
        if state is not None and (state.corrected_text is None) != (state.corrected_at is None):
            problems.append(
                f"segment {index}: corrected_text/corrected_at pairing broken "
                f"(text={state.corrected_text!r}, at={state.corrected_at!r})"
            )

    got_progress = verified_progress(session, run_id)
    if got_progress != expect.progress:
        problems.append(f"progress={got_progress}, expected {expect.progress}")

    split_rows = (
        session.query(SegmentSplitBoundary)
        .filter(SegmentSplitBoundary.pipeline_run_id == run_id)
        .all()
    )
    split_parent_ids = {row.parent_segment_id for row in split_rows}
    seg_id_to_index = {seg.id: seg.segment_index for seg in segments}
    unmappable = split_parent_ids - seg_id_to_index.keys()
    if unmappable:
        problems.append(
            f"split rows reference parent segment IDs not in this run: "
            f"{sorted(str(s) for s in unmappable)}"
        )
    got_split_indexes = {seg_id_to_index[sid] for sid in split_parent_ids if sid in seg_id_to_index}
    missing_splits = expect.split_parent_indexes - got_split_indexes
    if missing_splits:
        problems.append(f"expected splits on segments {sorted(missing_splits)} but none found")
    extra_splits = got_split_indexes - expect.split_parent_indexes
    if extra_splits:
        problems.append(f"unexpected splits on segments {sorted(extra_splits)}")

    if expect.expected_annotations is not None:
        got_annot = (
            session.query(TranscriptAnnotation)
            .filter(
                TranscriptAnnotation.pipeline_run_id == run_id,
                TranscriptAnnotation.deleted_at.is_(None),
            )
            .count()
        )
        if got_annot != expect.expected_annotations:
            problems.append(
                f"annotation count={got_annot}, expected {expect.expected_annotations}"
            )

    return problems


def cmd_reconcile(args: argparse.Namespace) -> None:
    url = _guarded(args.database_url)
    try:
        run_id = uuid.UUID(args.run_id)
    except ValueError:
        fail(f"--run-id is not a valid UUID: {args.run_id!r}")
    if not args.expect_file and not args.expect:
        fail("provide --expect-file <path> or --expect '<json>' with the durable outcome.")
    source = Path(args.expect_file).read_text() if args.expect_file else args.expect
    try:
        expect = Expectation.from_dict(json.loads(source))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        fail(f"invalid expectation: {exc}")

    engine = create_engine(url)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        problems = reconcile_run(session, run_id, expect)
    engine.dispose()
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        fail(f"durable state does not match the expectation ({len(problems)} mismatch(es)).")
    print(f"ok: {expect.progress[0]} of {expect.progress[1]} verified; corrections match")
    print("RECONCILE PASS")


# --------------------------------------------------------------------------- #
# teardown — kill by port, unstage artifacts, optionally drop the DB
# --------------------------------------------------------------------------- #
def unstage_build(static_dir: Path) -> list[str]:
    """Remove everything the build staged, keep ``.gitkeep``. Returns names removed."""
    removed: list[str] = []
    if not static_dir.is_dir():
        return removed
    for child in sorted(static_dir.iterdir()):
        if child.name == GITKEEP:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child.name)
    return removed


def _drop_database(url: str) -> None:
    db_name = urlsplit(url).path.lstrip("/")
    admin = create_engine(_admin_url(url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = :n AND pid <> pg_backend_pid()"
            ),
            {"n": db_name},
        )
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
    admin.dispose()
    print(f"ok: dropped database {db_name}")


def cmd_teardown(args: argparse.Namespace) -> None:
    # Kill by PORT, never `pkill -f "voxint serve"` (that also restarts the
    # dockerized api container). `fuser` is Linux — the maintainer host is Linux;
    # on macOS the equivalent is `lsof -ti tcp:<port> | xargs kill`.
    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], check=False)
    removed = unstage_build(Path(args.static_dir))
    if removed:
        print(f"ok: unstaged build artifacts {removed}")
    media_root = Path(args.media_root).resolve()
    # rm -rf on an operator-supplied path is a footgun (`--media-root .` would
    # delete the working tree). Only remove a disposable-looking dir under the
    # repo (kimi review); refuse anything else rather than guess.
    if media_root.is_dir():
        looks_disposable = REPO_ROOT in media_root.parents and (
            "e2e" in media_root.name or "test" in media_root.name
        )
        if not looks_disposable:
            fail(
                f"refusing to rm -rf {media_root}: not a disposable media dir under "
                f"{REPO_ROOT} (name must contain 'e2e' or 'test')."
            )
        shutil.rmtree(media_root)
        print(f"ok: removed {media_root}")
    gitkeep = Path(args.static_dir) / GITKEEP
    subprocess.run(["git", "checkout", "--", str(gitkeep)], cwd=REPO_ROOT, check=False)
    if args.drop_db:
        url = _guarded(args.database_url)
        _drop_database(url)
    print("TEARDOWN PASS")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="build + stage the islands")
    p_setup.add_argument("--frontend-dir", default=str(DEFAULT_FRONTEND_DIR))
    p_setup.add_argument("--static-dir", default=str(DEFAULT_STATIC_DIR))
    p_setup.set_defaults(func=cmd_setup)

    p_seed = sub.add_parser("seed", help="reset + migrate a disposable DB, seed a review run")
    p_seed.add_argument("--database-url", required=True)
    p_seed.add_argument("--media-root", default=str(DEFAULT_MEDIA_DIR))
    p_seed.add_argument(
        "--create-db",
        action="store_true",
        help="create the database + vector extension first (fresh host)",
    )
    p_seed.add_argument(
        "--fixture",
        choices=FIXTURE_CHOICES,
        default="review",
        help="segment fixture: review (5 segs), editor (30 segs), benchmark (2000 segs)",
    )
    p_seed.set_defaults(func=cmd_seed)

    p_serve = sub.add_parser("serve", help="serve a working-tree instance (exec; background it)")
    p_serve.add_argument("--database-url", required=True)
    p_serve.add_argument("--media-root", default=str(DEFAULT_MEDIA_DIR))
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.add_argument("--user", default="admin")
    p_serve.add_argument("--password", default="e2epass")
    p_serve.set_defaults(func=cmd_serve)

    p_rec = sub.add_parser("reconcile", help="fail-closed durable-state verifier")
    p_rec.add_argument("--database-url", required=True)
    p_rec.add_argument("--run-id", required=True)
    p_rec.add_argument("--expect-file", help="JSON file with the expected durable outcome")
    p_rec.add_argument("--expect", help="inline JSON with the expected durable outcome")
    p_rec.set_defaults(func=cmd_reconcile)

    p_td = sub.add_parser("teardown", help="kill by port, unstage artifacts, optional DB drop")
    p_td.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_td.add_argument("--static-dir", default=str(DEFAULT_STATIC_DIR))
    p_td.add_argument("--media-root", default=str(DEFAULT_MEDIA_DIR))
    p_td.add_argument("--drop-db", action="store_true", help="also DROP the disposable database")
    p_td.add_argument("--database-url", help="required with --drop-db (disposable name only)")
    p_td.set_defaults(func=cmd_teardown)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
