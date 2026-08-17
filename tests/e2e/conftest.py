"""End-to-end fixtures: the *real* pipeline against *real* model services.

This is a maintainer-run, opt-in gate — never public CI (see
``docs/release-process.md``). Unlike ``tests/integration`` (a real DB, faked
model I/O) and ``tests/parity`` (numerics against committed references), these
tests submit audio and drive faster-whisper + pyannote + TitaNet running in
their containers, asserting the persistence invariants the whole pipeline is
supposed to leave behind.

Gate semantics (deliberately asymmetric — codex review):

* ``VOXINT_E2E`` **unset** → the whole directory is *skipped* at collection, so
  a bare ``pytest`` run stays green on any developer machine.
* ``VOXINT_E2E=1`` → an explicit E2E run. A missing prerequisite (test DB,
  model-service health, wrong device identity) is a **failure, not a skip** — an
  operator who asked for the real gate must never get a green board because the
  suite quietly skipped itself.

Prerequisites when enabled:

* ``VOXINT_TEST_DATABASE_URL`` points at a **disposable** database (its schema is
  dropped and rebuilt from the alembic chain, and every table is truncated
  between tests — never aim it at live data).
* The three model services answer ``/healthz`` with the expected identity and no
  device fallback. Host-specific bring-up (compose overlays, CPU caps, the AMD
  render gid) lives outside this repo; this file only *asserts* the contract.
* ``MEDIA_ROOT`` (via ``Settings``) is the host directory the model-service
  containers mount at ``/data/media``. Fixtures stage audio there so a container
  reading a path relative to *its* mount sees the same bytes.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import Base

E2E_ENABLED = os.environ.get("VOXINT_E2E") is not None
TEST_DB_URL = os.environ.get("VOXINT_TEST_DATABASE_URL")
REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_WAV = REPO_ROOT / "src" / "voxint" / "tutorial" / "assets" / "sample-3speaker.wav"

# The identity each model service must report at /healthz for the real-pipeline
# lane. ``device`` is load-bearing: a container that silently fell back to CPU
# (whisper) or CUDA (a mis-tagged image) would still "work" but stops being the
# thing we mean to gate — abort on any drift, never skip.
EXPECTED_SERVICES: dict[str, dict[str, str]] = {
    "asr_url": {"service": "whisper", "device": "rocm", "model": "large-v2"},
    "diarizer_url": {
        "service": "pyannote",
        "device": "cpu",
        "model": "pyannote/speaker-diarization-3.1",
    },
    "embedder_url": {"service": "titanet", "device": "cpu"},
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # pytestmark in a conftest does NOT propagate to sibling modules, and this
    # hook sees the whole session — mark only this directory's items. Skip the
    # E2E lane unless it was explicitly requested; when it IS requested, the
    # fixtures below FAIL (not skip) on any missing prerequisite.
    if E2E_ENABLED:
        return
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="VOXINT_E2E not set (opt-in maintainer gate)")
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(skip)


def _healthz(base_url: str) -> dict[str, object]:
    resp = httpx.get(f"{base_url.rstrip('/')}/healthz", timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    assert isinstance(data, dict)
    return data


def assert_service_identity(base_url: str, expected: dict[str, str]) -> None:
    """Fail hard unless the service at ``base_url`` reports the expected identity.

    Called by the ``model_services`` fixture; raises ``pytest.fail`` (a failure,
    not a skip) so an enabled E2E run cannot go green against the wrong image,
    an unloaded model, or a device fallback.
    """
    try:
        health = _healthz(base_url)
    except (httpx.HTTPError, ValueError) as exc:  # unreachable / non-JSON
        pytest.fail(
            f"VOXINT_E2E=1 but model service {expected['service']} at {base_url} "
            f"is not answering /healthz: {exc}"
        )
    if health.get("model_loaded") is not True:
        pytest.fail(f"{expected['service']} at {base_url}: model_loaded is not True: {health}")
    if health.get("contract_version") != "v1":
        pytest.fail(
            f"{expected['service']} at {base_url}: contract_version "
            f"{health.get('contract_version')!r} != 'v1': {health}"
        )
    for key, want in expected.items():
        got = health.get(key)
        if got != want:
            pytest.fail(
                f"{expected['service']} at {base_url}: {key}={got!r} != {want!r} "
                f"(device fallback or wrong image?): {health}"
            )


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Application settings for the E2E lane, pinned to the disposable test DB.

    Fails (not skips) when the test DB URL is absent: an explicit E2E run with no
    isolated database is a misconfiguration, and we must never silently fall back
    to ``Settings``' default live DSN.
    """
    if TEST_DB_URL is None:
        pytest.fail(
            "VOXINT_E2E=1 requires VOXINT_TEST_DATABASE_URL pointing at a "
            "disposable database (its schema is dropped and rebuilt)."
        )
    # Settings reads DATABASE_URL from the environment over any .env value; pin it
    # to the disposable DB so the in-process pipeline never touches live data.
    os.environ["DATABASE_URL"] = TEST_DB_URL
    return Settings()


@pytest.fixture(scope="session")
def model_services(settings: Settings) -> None:
    """Assert all three model services are up with the expected identity."""
    assert_service_identity(settings.asr_url, EXPECTED_SERVICES["asr_url"])
    assert_service_identity(settings.diarizer_url, EXPECTED_SERVICES["diarizer_url"])
    assert_service_identity(settings.embedder_url, EXPECTED_SERVICES["embedder_url"])


@pytest.fixture(scope="session")
def engine(settings: Settings) -> Iterator[Engine]:
    """A fresh schema on the disposable test DB, migrated by the alembic chain."""
    assert TEST_DB_URL is not None  # guarded by the settings fixture
    eng = create_engine(str(settings.database_url))
    with eng.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    yield sessionmaker(engine, expire_on_commit=False)
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
        conn.commit()


@pytest.fixture()
def stage_media(settings: Settings) -> Iterator[Callable[[], str]]:
    """Return a factory that stages the tutorial WAV under the media root.

    Each call copies ``sample-3speaker.wav`` to a unique path *relative to*
    ``MEDIA_ROOT`` and returns that relative path (what ``MediaItem.source_path``
    stores). The model-service containers mount the same host directory at
    ``/data/media``, so a path they resolve against their mount points at these
    bytes. A unique name per call lets one test submit several runs without
    colliding on ``source_path``. All staged files are removed afterwards.
    """
    if not TUTORIAL_WAV.is_file():
        pytest.fail(f"tutorial fixture WAV missing: {TUTORIAL_WAV}")
    media_root = Path(settings.media_root)
    staged: list[Path] = []

    def _stage() -> str:
        rel = Path("e2e") / f"{uuid.uuid4().hex}.wav"
        dest = media_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TUTORIAL_WAV, dest)
        staged.append(dest)
        return str(rel)

    yield _stage

    for dest in staged:
        dest.unlink(missing_ok=True)
