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
import shutil
import subprocess
import sys
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import verified_progress
from voxint.app_settings import complete_onboarding
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    TranscriptSegment,
)

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
    """
    db_name = urlsplit(url).path.lstrip("/").lower()
    if not db_name or ("test" not in db_name and "e2e" not in db_name):
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
    place BEFORE ``serve``. Returns the top-level names copied in (so teardown
    removes exactly those and nothing tracked). ``.gitkeep`` is never touched.
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
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def seed_browser_run(session: Session, media_root: Path) -> uuid.UUID:
    """A COMPLETED run shaped for the review loop: audio + varied-confidence text.

    Mirrors ``tests/integration/test_review_api.py::_seed_run_with_confidences``
    but adds what the browser lane needs and the enrichment seed lacks: a
    PREPROCESSED_AUDIO artifact (playback source) and ``duration_seconds`` set on
    the media item (without it ``playback_capability`` gates seeking off). LLM is
    left off (``complete_onboarding`` is the caller's job) — this lane exercises
    the transcript loop, not enrichment.
    """
    duration = _SEGMENT_SECONDS * len(_SEED_SEGMENTS)
    media = MediaItem(
        source_path=f"e2e/{uuid.uuid4().hex}.wav",
        duration_seconds=duration,
    )
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
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

    for index, (label, seg_text, confidence) in enumerate(_SEED_SEGMENTS):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index) * _SEGMENT_SECONDS,
                end_seconds=float(index) * _SEGMENT_SECONDS + _SEGMENT_SECONDS,
                raw_text=seg_text,
                diarization_label=label,
                confidence=confidence,
            )
        )
    session.commit()
    return run.id


def _admin_url(url: str) -> str:
    """The same server, pointed at the ``postgres`` maintenance database."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/postgres"))


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
    if args.create_db:
        _create_database(url)
    _reset_schema_and_migrate(url)
    engine = create_engine(url)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        complete_onboarding(session, llm_enabled_default=False)
        session.commit()
        run_id = seed_browser_run(session, media_root)
    engine.dispose()
    print(f"ok: seeded COMPLETED review run with {len(_SEED_SEGMENTS)} segments")
    # Machine-readable line for the skill to capture.
    print(f"RUN_ID={run_id}")
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
@dataclass(frozen=True)
class Expectation:
    """The durable outcome the browser was supposed to leave behind.

    ``verified_indexes`` — segment_index values whose review state must be
    verified (all others must NOT be). ``corrections`` — segment_index → exact
    corrected text (a segment absent here must carry no correction). ``progress``
    — the ``(verified, total)`` N-of-M counter the console showed.
    """

    verified_indexes: frozenset[int]
    corrections: dict[int, str]
    progress: tuple[int, int]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Expectation:
        verified = data.get("verified_segment_indexes", [])
        if not isinstance(verified, list):
            raise ValueError("verified_segment_indexes must be a list of integers")
        raw_corr = data.get("corrections", {})
        if not isinstance(raw_corr, dict):
            raise ValueError("corrections must be an object of index → text")
        corrections = {int(k): str(v) for k, v in raw_corr.items()}
        prog = data.get("progress", {})
        if not isinstance(prog, dict) or "verified" not in prog or "total" not in prog:
            raise ValueError("progress must be an object with 'verified' and 'total'")
        return cls(
            verified_indexes=frozenset(int(i) for i in verified),
            corrections=corrections,
            progress=(int(prog["verified"]), int(prog["total"])),
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
    return problems


def cmd_reconcile(args: argparse.Namespace) -> None:
    url = _guarded(args.database_url)
    try:
        run_id = uuid.UUID(args.run_id)
    except ValueError:
        fail(f"--run-id is not a valid UUID: {args.run_id!r}")
    if args.expect_file:
        raw = json.loads(Path(args.expect_file).read_text())
    elif args.expect:
        raw = json.loads(args.expect)
    else:
        fail("provide --expect-file <path> or --expect '<json>' with the durable outcome.")
    try:
        expect = Expectation.from_dict(raw)
    except (ValueError, TypeError) as exc:
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
    # dockerized api container).
    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], check=False)
    removed = unstage_build(Path(args.static_dir))
    if removed:
        print(f"ok: unstaged build artifacts {removed}")
    media_root = Path(args.media_root)
    if media_root.is_dir():
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
