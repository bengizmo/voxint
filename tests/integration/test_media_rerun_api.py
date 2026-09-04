"""POST /media/rerun and /media/rerun/confirm — the P2b bulk re-run (issue #154).

Commit 4 adds a two-step, non-destructive bulk re-run: an advisory preview that
resolves the config a fresh run would freeze and captures a per-file latest-run
baseline, then an atomic confirm that row-locks the selection, skips any file that
gained a newer run since preview, and mints one fresh run per surviving file in a
single transaction. These tests pin the wiring the pure seams cannot see: the CSRF
gate, whole-selection prevalidation, precedence resolved off the STORED settings
folder (AC-2), preview==dispatch parity, double-confirm idempotency, the honest
dropping of the prior run's notes/hints, on-disk sidecar re-read, and the flag-off
404.

Needs the real Postgres test DB (the FOR UPDATE lock / window baseline are
Postgres behaviour), so skipped without VOXINT_TEST_DATABASE_URL.
"""

import re
import uuid
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_MEDIA_RERUN,
    CSRF_MEDIA_RERUN_CONFIRM,
    mint_csrf_token,
)
from voxint.api.media_query import MEDIA_LIBRARY_LIMIT
from voxint.config import Settings
from voxint.db.models import MediaFolder, MediaItem, PipelineRun, Project, RunStatus
from voxint.ingest import submit_media_item

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "media-rerun-test-csrf-key"

BASE_PACK = "base"
BASE_VOCAB = ["base-term"]
PROJECT_VOCAB = ["project-term"]


def _write_pack(root: Path, name: str, vocab: list[str]) -> None:
    pack_dir = root / name
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"name": name, "vocabulary": vocab, "corrections": []})
    )


def _settings(tmp_path: Path, *, media_enabled: bool = True) -> Settings:
    packs = tmp_path / "packs"
    _write_pack(packs, BASE_PACK, BASE_VOCAB)
    return Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path / "media",
        console_media_enabled=media_enabled,
        csrf_secret=_CSRF_KEY,
        domain_packs_dir=packs,
        domain_pack_path=packs / BASE_PACK,
    )


def _make_client(
    session_factory: sessionmaker[Session], settings: Settings
) -> TestClient:
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = _settings(tmp_path)
    (s.media_root).mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], settings: Settings
) -> TestClient:
    return _make_client(session_factory, settings)


def _data(csrf_action: str, **fields: object) -> dict[str, object]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, csrf_action), **fields}


def _add_media(
    session: Session, *, source_path: str, folder_id: uuid.UUID | None = None
) -> MediaItem:
    media = MediaItem(source_path=source_path, media_folder_id=folder_id, size_bytes=7)
    session.add(media)
    session.flush()
    return media


def _baseline_run(
    session: Session,
    source_path: str,
    settings: Settings,
    *,
    notes: str | None = None,
    num_speakers: int | None = None,
) -> PipelineRun:
    result = submit_media_item(
        session,
        source_path,
        settings=settings,
        diarization_num_speakers=num_speakers,
    )
    run = session.get(PipelineRun, result.run_id)
    if notes is not None:
        run.operator_notes = notes
    session.flush()
    return run


def _run_count(session_factory: sessionmaker[Session], media_id: uuid.UUID) -> int:
    with session_factory() as session:
        return session.execute(
            select(func.count())
            .select_from(PipelineRun)
            .where(PipelineRun.media_item_id == media_id)
        ).scalar_one()


def _latest_run(
    session_factory: sessionmaker[Session], media_id: uuid.UUID
) -> PipelineRun:
    with session_factory() as session:
        run = session.execute(
            select(PipelineRun)
            .where(PipelineRun.media_item_id == media_id)
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
            .limit(1)
        ).scalar_one()
        session.expunge(run)
        return run


def _pair(media_id: uuid.UUID, baseline: str) -> str:
    return f"{media_id}:{baseline}"


def _scrape_pairs(html: str) -> list[str]:
    return re.findall(r'name="item" value="([^"]+)"', html)


# ---- preview -----------------------------------------------------------------


def test_rerun_preview_renders_and_creates_no_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()

    resp = client.post(
        "/media/rerun",
        data=_data(CSRF_MEDIA_RERUN, media_id=[str(m_id)]),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert 'action="/media/rerun/confirm"' in resp.text
    assert _pair(m_id, "none") in resp.text  # no prior run -> the no-run sentinel
    assert _run_count(session_factory, m_id) == 0  # advisory: nothing minted


def test_rerun_preview_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()
    resp = client.post(
        "/media/rerun",
        data={"csrf_token": "forged", "media_id": [str(m_id)]},
        follow_redirects=False,
    )
    assert resp.status_code == 403


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, 400),  # empty selection
        ({"media_id": ["not-a-uuid"]}, 400),  # malformed
    ],
)
def test_rerun_preview_prevalidation(
    client: TestClient, payload: dict[str, object], expected: int
) -> None:
    resp = client.post(
        "/media/rerun", data=_data(CSRF_MEDIA_RERUN, **payload), follow_redirects=False
    )
    assert resp.status_code == expected


def test_rerun_preview_over_cap_rejected(client: TestClient) -> None:
    too_many = [str(uuid.uuid4()) for _ in range(MEDIA_LIBRARY_LIMIT + 1)]
    resp = client.post(
        "/media/rerun",
        data=_data(CSRF_MEDIA_RERUN, media_id=too_many),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert f"at most {MEDIA_LIBRARY_LIMIT}" in resp.text


# ---- confirm: dispatch, precedence, parity -----------------------------------


def test_confirm_mints_run_with_folder_precedence_and_parity(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    """AC-2 + preview==dispatch parity via the real preview->confirm flow."""
    with session_factory() as session:
        project = Project(name="P", vocabulary=list(PROJECT_VOCAB))
        session.add(project)
        session.flush()
        folder = MediaFolder(path="interviews", project_id=project.id)
        session.add(folder)
        session.flush()
        m = _add_media(
            session, source_path="interviews/a.wav", folder_id=folder.id
        )
        m_id = m.id
        session.commit()

    preview = client.post(
        "/media/rerun",
        data=_data(CSRF_MEDIA_RERUN, media_id=[str(m_id)]),
        follow_redirects=False,
    )
    assert preview.status_code == 200
    # The preview shows the project layer supplying the vocabulary.
    assert "project" in preview.text
    pairs = _scrape_pairs(preview.text)
    assert pairs == [_pair(m_id, "none")]

    confirm = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=pairs),
        follow_redirects=False,
    )
    assert confirm.status_code == 200
    assert _run_count(session_factory, m_id) == 1
    snapshot = _latest_run(session_factory, m_id).domain_pack
    assert snapshot is not None
    # Precedence followed the STORED folder/project, not the raw path or global.
    assert snapshot["vocabulary"] == PROJECT_VOCAB
    assert snapshot["config_resolution_version"] == 2
    # Parity: what the preview advertised is exactly what was frozen.
    assert f"{len(snapshot['vocabulary'])} glossary term" in preview.text


def test_confirm_double_submit_creates_at_most_one_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()

    payload = _data(CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(m_id, "none")])
    first = client.post("/media/rerun/confirm", data=payload, follow_redirects=False)
    assert first.status_code == 200
    assert _run_count(session_factory, m_id) == 1

    # A replay of the SAME confirm (stale baseline) mints nothing more.
    second = client.post("/media/rerun/confirm", data=payload, follow_redirects=False)
    assert second.status_code == 200
    assert "Skipped" in second.text
    assert "a newer run appeared" in second.text
    assert _run_count(session_factory, m_id) == 1


def test_confirm_skips_item_that_gained_a_newer_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run0 = _baseline_run(session, "incoming/a.wav", settings)
        m_id, run0_id = m.id, run0.id
        session.commit()

    # A newer run appears after the (simulated) preview captured run0 as baseline.
    with session_factory() as session:
        _baseline_run(session, "incoming/a.wav", settings)
        session.commit()
    assert _run_count(session_factory, m_id) == 2

    resp = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(m_id, str(run0_id))]),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "a newer run appeared" in resp.text
    assert _run_count(session_factory, m_id) == 2  # nothing new minted


def test_confirm_drops_prior_notes_and_hints(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run0 = _baseline_run(
            session, "incoming/a.wav", settings, notes="carry me?", num_speakers=2
        )
        m_id, run0_id = m.id, run0.id
        session.commit()

    resp = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(m_id, str(run0_id))]),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    fresh = _latest_run(session_factory, m_id)
    assert fresh.id != run0_id
    # The prior run's operator notes and manual speaker hint do NOT carry.
    assert fresh.operator_notes is None
    assert fresh.diarization_num_speakers is None


def test_confirm_rereads_on_disk_sidecar(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    media_dir = settings.media_root / "interviews"
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir / "clip.wav.yaml").write_text(
        yaml.safe_dump({"speakers": ["Alice", "Bob"]})
    )
    with session_factory() as session:
        m = _add_media(session, source_path="interviews/clip.wav")
        # Baseline was created WITHOUT a sidecar (submit_media_item never reads disk).
        run0 = _baseline_run(session, "interviews/clip.wav", settings)
        m_id, run0_id = m.id, run0.id
        assert run0.sidecar is None
        session.commit()

    resp = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(m_id, str(run0_id))]),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    fresh = _latest_run(session_factory, m_id)
    assert fresh.id != run0_id
    # The re-run re-read the on-disk sidecar and froze it onto the fresh run.
    assert fresh.sidecar is not None


def test_confirm_skips_item_with_unreadable_sidecar(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    media_dir = settings.media_root / "interviews"
    media_dir.mkdir(parents=True, exist_ok=True)
    # Non-UTF-8 bytes -> read_sidecar raises SidecarError -> the item is skipped.
    (media_dir / "clip.wav.yaml").write_bytes(b"\xff\xfe not utf-8")
    with session_factory() as session:
        m = _add_media(session, source_path="interviews/clip.wav")
        m_id = m.id
        session.commit()

    # Preview flags it before the operator confirms.
    preview = client.post(
        "/media/rerun",
        data=_data(CSRF_MEDIA_RERUN, media_id=[str(m_id)]),
        follow_redirects=False,
    )
    assert "sidecar could not be read" in preview.text

    resp = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(m_id, "none")]),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "sidecar could not be read" in resp.text
    assert _run_count(session_factory, m_id) == 0  # nothing minted for the bad file


def test_confirm_mixed_selection_queues_runnable_and_skips_stale(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        good = _add_media(session, source_path="incoming/good.wav")
        stale = _add_media(session, source_path="incoming/stale.wav")
        good_id, stale_id = good.id, stale.id
        session.commit()

    # `good` has a matching (no-run) baseline; `stale` claims a baseline run that
    # never existed, so its "latest" (none) differs -> it is skipped.
    resp = client.post(
        "/media/rerun/confirm",
        data=_data(
            CSRF_MEDIA_RERUN_CONFIRM,
            item=[_pair(good_id, "none"), _pair(stale_id, str(uuid.uuid4()))],
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert _run_count(session_factory, good_id) == 1
    assert _run_count(session_factory, stale_id) == 0


def test_confirm_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()
    resp = client.post(
        "/media/rerun/confirm",
        data={"csrf_token": "forged", "item": [_pair(m_id, "none")]},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert _run_count(session_factory, m_id) == 0


def test_confirm_malformed_pair_rejected(client: TestClient) -> None:
    resp = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=["no-colon-here"]),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "not valid" in resp.text


def test_confirm_count_mismatch_rejected(client: TestClient) -> None:
    # A pair referencing a media row that does not exist -> 409, zero writes.
    resp = client.post(
        "/media/rerun/confirm",
        data=_data(
            CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(uuid.uuid4(), "none")]
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 409
    assert "no longer exist" in resp.text


def test_rerun_routes_404_when_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = _settings(tmp_path, media_enabled=False)
    settings.media_root.mkdir(parents=True, exist_ok=True)
    off = _make_client(session_factory, settings)
    assert (
        off.post(
            "/media/rerun",
            data=_data(CSRF_MEDIA_RERUN, media_id=[str(uuid.uuid4())]),
            follow_redirects=False,
        ).status_code
        == 404
    )
    assert (
        off.post(
            "/media/rerun/confirm",
            data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=[_pair(uuid.uuid4(), "none")]),
            follow_redirects=False,
        ).status_code
        == 404
    )


# ---- publish batch cap and broker-failure short-circuit --------------------


def test_confirm_publish_cap_limits_broker_traffic(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The publish loop stops after rerun_publish_batch_size; all runs are
    committed (commit-before-publish) but items beyond the cap stay deferred."""
    import unittest.mock

    from voxint.ingest.service import SubmissionResult

    s = _settings(tmp_path)
    s = s.model_copy(update={"rerun_publish_batch_size": 2})
    s.media_root.mkdir(parents=True, exist_ok=True)
    client = _make_client(session_factory, s)

    media_ids: list[uuid.UUID] = []
    with session_factory() as session:
        for i in range(5):
            m = _add_media(session, source_path=f"file_{i}.wav")
            media_ids.append(m.id)
        session.commit()

    publish_calls: list[uuid.UUID] = []

    def counting_publish(self: SubmissionResult) -> bool:
        publish_calls.append(self.run_id)
        return True

    preview = client.post(
        "/media/rerun",
        data=_data(CSRF_MEDIA_RERUN, media_id=[str(mid) for mid in media_ids]),
        follow_redirects=False,
    )
    assert preview.status_code == 200
    pairs = _scrape_pairs(preview.text)
    assert len(pairs) == 5

    with unittest.mock.patch.object(SubmissionResult, "publish", counting_publish):
        confirm = client.post(
            "/media/rerun/confirm",
            data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=pairs),
            follow_redirects=False,
        )

    assert confirm.status_code == 200
    # All 5 runs committed (commit-before-publish contract).
    for mid in media_ids:
        assert _run_count(session_factory, mid) == 1
    # Only 2 publish calls (the batch cap).
    assert len(publish_calls) == 2
    # Items beyond the cap show the deferred marker.
    assert "will start when the worker is available" in confirm.text


def test_confirm_broker_failure_short_circuits(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker failure on the first publish stops all further publish attempts;
    runs are still committed QUEUED."""
    from voxint.ingest.service import SubmissionResult

    s = _settings(tmp_path)
    s.media_root.mkdir(parents=True, exist_ok=True)
    client = _make_client(session_factory, s)

    media_ids: list[uuid.UUID] = []
    with session_factory() as session:
        for i in range(3):
            m = _add_media(session, source_path=f"broker_{i}.wav")
            media_ids.append(m.id)
        session.commit()

    publish_calls: list[uuid.UUID] = []

    def failing_publish(self: SubmissionResult) -> bool:
        publish_calls.append(self.run_id)
        return False

    preview = client.post(
        "/media/rerun",
        data=_data(CSRF_MEDIA_RERUN, media_id=[str(mid) for mid in media_ids]),
        follow_redirects=False,
    )
    assert preview.status_code == 200
    pairs = _scrape_pairs(preview.text)
    assert len(pairs) == 3

    monkeypatch.setattr(SubmissionResult, "publish", failing_publish)

    confirm = client.post(
        "/media/rerun/confirm",
        data=_data(CSRF_MEDIA_RERUN_CONFIRM, item=pairs),
        follow_redirects=False,
    )

    assert confirm.status_code == 200
    # Only 1 publish call — short-circuited after the first failure.
    assert len(publish_calls) == 1
    # All 3 runs committed as QUEUED (commit-before-publish).
    for mid in media_ids:
        assert _run_count(session_factory, mid) == 1
    with session_factory() as session:
        runs = session.execute(
            select(PipelineRun).where(PipelineRun.media_item_id.in_(media_ids))
        ).scalars().all()
        assert all(r.status == RunStatus.QUEUED.value for r in runs)
    # Deferred banner and per-item marker present.
    assert "Some runs are queued but not yet handed to the worker" in confirm.text
    assert "will start when the worker is available" in confirm.text
