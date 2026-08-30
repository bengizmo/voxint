"""The Home page (Console 2.0 P1, #152) end to end against real Postgres.

Home replaced the operator dashboard (issue #13): the needs-attention cards,
the quick actions, the windowed activity counts, and the recent-activity feed.
The counts are thin views over ``stats_query`` (unit-covered in
``tests/unit/test_stats_query.py``); these tests pin the wiring the unit tests
cannot see — auth, that the attention counts match the queues they link to,
that the window switch changes the rendered figures and degrades honestly on a
bad value, that Home and ``/metrics`` agree on one seed, the activity feed's
content and ordering, and the retired ``/dashboard`` redirect.
"""

import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    Stage,
    StageRun,
    StageStatus,
)

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def make_run(
    session: Session,
    *,
    status: RunStatus,
    created_at: datetime | None = None,
    stages: Iterable[dict[str, object]] = (),
) -> uuid.UUID:
    """Seed one media item + run, optionally with StageRun attempts."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=status.value)
    if created_at is not None:
        run.created_at = created_at
    session.add(run)
    session.flush()
    for spec in stages:
        session.add(StageRun(pipeline_run_id=run.id, **spec))
    session.commit()
    return run.id


def seed_snapshot(session_factory: sessionmaker[Session]) -> None:
    """A fixed spread of runs/stages/speakers for Home to aggregate."""
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    with session_factory() as session:
        # Two runs in the transient AWAITING_ADJUDICATION status. These exercise
        # the status counts and the raw voxint_runs gauge, but they are NOT the
        # review backlog: a successful pipeline terminates COMPLETED, so the queue
        # (issue #117) is keyed on COMPLETED runs with unresolved speaker labels,
        # not on this status.
        make_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        make_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        # One completed run with a finished transcribe (30s) and a failed diarize.
        # It carries an unresolved diarization label, so it is the one genuine
        # review-backlog entry -> review backlog == 1, unresolved voices == 1.
        completed = make_run(
            session,
            status=RunStatus.COMPLETED,
            stages=[
                {
                    "stage": Stage.TRANSCRIBE.value,
                    "status": StageStatus.COMPLETED.value,
                    "started_at": now - timedelta(seconds=30),
                    "finished_at": now,
                },
                {
                    "stage": Stage.DIARIZE_EMBED.value,
                    "status": StageStatus.FAILED.value,
                    "started_at": now - timedelta(seconds=5),
                    "finished_at": now,
                },
            ],
        )
        vector = [0.0] * EMBEDDING_DIM
        vector[0] = 1.0
        session.add(
            DiarizationTurn(
                pipeline_run_id=completed,
                turn_index=0,
                start_seconds=0.0,
                end_seconds=8.0,
                label="S0",
                embedding=vector,
                embedding_space="titanet-large-v1",
            )
        )
        session.commit()
        # One old FAILED run outside a short window (window-narrowing + the
        # failed-runs attention card, which is all-time). Backdate its media
        # item too: make_run backdates only the run row.
        old_failed = make_run(session, status=RunStatus.FAILED, created_at=old)
        old_run = session.get(PipelineRun, old_failed)
        assert old_run is not None
        old_run.media_item.created_at = old
        # Two enrolled speakers -> roster size == 2.
        session.add(Speaker(display_name="Alice"))
        session.add(Speaker(display_name="Bob"))
        session.commit()


def _stat_value(body: str, label: str) -> int:
    """Pull an oc-stat-tile's value by its label; ties the label to its adjacent
    value so a tile cannot silently show the wrong figure."""
    match = re.search(
        rf'class="tile-value">\s*(\d+)\s*(?:<span[^>]*>[^<]*</span>\s*)?</div>\s*'
        rf'<div class="tile-label">{re.escape(label)}</div>',
        body,
    )
    assert match is not None, f"stat tile {label!r} missing from Home render"
    return int(match.group(1))


def test_home_renders_attention_cards_and_stats(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    # Nav marks Home current.
    assert 'aria-current="page"' in body
    # Attention cards above the fold, before the activity section.
    cards_at = body.index('class="attention-cards"')
    assert cards_at < body.index("ACTIVITY")
    # Attention counts: the seed's one unresolved COMPLETED run (singular
    # label inflection, #318).
    assert "recordings to review" in body
    assert "voice without a name" in body
    assert "failed runs" in body
    # Non-zero cards link to their queues.
    assert 'href="/review"' in body
    # Default window is the day: the 3-day-old run is outside it.
    assert _stat_value(body, "JOBS RUN") == 3
    assert _stat_value(body, "MEDIA ADDED") == 3
    assert _stat_value(body, "SPEAKERS IDENTIFIED") == 2


def test_home_attention_counts_match_their_queues(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The card counts derive from the queues they link to and cannot drift."""
    from voxint.adjudication.resolver import adjudication_queue

    seed_snapshot(session_factory)
    with session_factory() as session:
        queue = adjudication_queue(session)
        eligible = len(queue)
        voices = sum(e.unresolved_labels for e in queue)
    assert (eligible, voices) == (1, 1)
    body = client.get("/").text
    assert "recordings to review" in body
    assert "voice without a name" in body


def test_home_empty_states_are_quiet_not_links(client: TestClient) -> None:
    """Zero attention cards show counts with neutral copy, no arrow links."""
    body = client.get("/").text
    # All three attention cards show zero with neutral notes.
    assert "recordings to review" in body
    assert "voices without a name" in body
    assert "nothing broke" in body
    # No arrow links on zero cards (the attention-arrow only renders when > 0).
    assert 'class="attention-arrow"' not in body
    # And the activity feed says so plainly.
    assert "No recent activity" in body


def test_home_quick_actions(client: TestClient) -> None:
    body = client.get("/").text
    # Add media is the primary command-bar action; start actions row below.
    assert 'cb-btn-primary' in body
    assert 'href="/runs#add-media"' in body
    assert "Upload a recording" in body
    assert "Review speakers" in body
    # Projects ships dark: no New-project action while the flag is off.
    assert "New project" not in body
    assert 'href="/projects"' not in body


def test_home_window_switch_changes_counts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    # Default day window: 3 recent runs. Week window: the 3-day-old one too.
    day = client.get("/").text
    week = client.get("/", params={"window": "week"}).text
    assert _stat_value(day, "JOBS RUN") == 3
    assert _stat_value(week, "JOBS RUN") == 4
    # The switch marks the active window (default day; week when chosen).
    assert 'is-active" href="/?window=day"' in day
    assert 'is-active" href="/?window=week"' in week
    # All time includes everything as well.
    assert _stat_value(client.get("/", params={"window": "all"}).text, "JOBS RUN") == 4


def test_home_malformed_window_degrades_with_notice(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/", params={"window": "not-a-window"})
    assert resp.status_code == 200
    body = resp.text
    # Falls back to the day window and says so; never silently a different one.
    assert "Unrecognized" in body
    assert _stat_value(body, "JOBS RUN") == 3
    assert 'is-active" href="/?window=day"' in body


def test_home_matches_metrics_snapshot(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The core invariant: Home's day-window figures agree with /metrics (and so
    with ``voxint stats``, which shares collect_stats) on one seed."""
    seed_snapshot(session_factory)
    home = client.get("/").text
    metrics = client.get("/metrics").text
    assert _stat_value(home, "JOBS RUN") == 3
    assert "voxint_runs_created_24h 3" in metrics
    assert "voxint_roster_speakers 2" in metrics
    assert _stat_value(home, "SPEAKERS IDENTIFIED") == 2
    # The raw awaiting_adjudication status gauge (2) is deliberately NOT the
    # review backlog: the backlog is the /review queue's eligibility count —
    # COMPLETED runs with unresolved labels (issue #117), one here.
    assert 'voxint_runs{status="awaiting_adjudication"} 2' in metrics
    assert "recordings to review" in home


def test_home_activity_feed_lists_runs_and_speakers(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    body = client.get("/").text
    # The feed names each family with a link to its surface.
    assert "Run started" in body
    assert "Run finished" in body
    assert "Run failed" in body
    assert "Speaker verified" in body
    assert re.search(r'href="/speakers">(Alice|Bob)</a>', body)
    # Newest first: the seeded speakers (enrolled last) render before the
    # 3-day-old failed run's entry.
    assert body.index("Speaker verified") < body.index("Run failed")


def test_dashboard_redirects_to_home(client: TestClient) -> None:
    """P1 (#152): the dashboard folded into Home; the route 303s for bookmarks."""
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"


def test_windowed_counts_cutoff_and_all_time(
    session_factory: sessionmaker[Session],
) -> None:
    """The Home stat switcher's counts (#152): inclusive cutoff, None = all time,
    speaker enrollments keep counting after a merge/archive, and the run count
    shares the archived-run exclusion with the rest of the stats."""
    from voxint.api.stats_query import windowed_counts

    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    with session_factory() as session:
        old_run_id = make_run(session, status=RunStatus.COMPLETED, created_at=old)
        # make_run backdates only the run; backdate its media item too so the
        # media cutoff is actually exercised.
        old_run = session.get(PipelineRun, old_run_id)
        assert old_run is not None
        old_run.media_item.created_at = old
        recent = make_run(session, status=RunStatus.COMPLETED)
        # An archived recent run: its media item still counts as added; the run
        # itself drops out (same policy as run_status_counts).
        archived = make_run(session, status=RunStatus.COMPLETED)
        run = session.get(PipelineRun, archived)
        assert run is not None
        run.archived_at = now
        # One old speaker, one recent-but-archived speaker: the archive must not
        # shrink the enrollment count.
        session.add(Speaker(display_name="old voice", created_at=old))
        session.add(Speaker(display_name="curated voice", deleted_at=now))
        session.commit()

    cutoff = now - timedelta(hours=1)
    with session_factory() as session:
        windowed = windowed_counts(session, since=cutoff)
        assert windowed.since == cutoff
        assert windowed.runs_started == 1  # recent only; archived excluded
        assert windowed.media_added == 2  # both recent media items
        assert windowed.speakers_enrolled == 1  # the archived-but-recent one

        all_time = windowed_counts(session, since=None)
        assert all_time.since is None
        assert all_time.runs_started == 2
        assert all_time.media_added == 3
        assert all_time.speakers_enrolled == 2
        # Sanity: the recent run is really there.
        assert session.get(PipelineRun, recent) is not None


def test_stats_exclude_archived_runs(session_factory: sessionmaker[Session]) -> None:
    """Archived runs (issue #5) drop out of the status counts and the created
    window so Home / metrics / CLI all report *active* runs."""
    from voxint.api.stats_query import run_status_counts, runs_created_since

    now = datetime.now(UTC)
    with session_factory() as session:
        active = make_run(session, status=RunStatus.COMPLETED)
        archived = make_run(session, status=RunStatus.COMPLETED)
        run = session.get(PipelineRun, archived)
        assert run is not None
        run.archived_at = now
        session.commit()

    with session_factory() as session:
        counts = run_status_counts(session)
        # Only the active completed run is counted, not the archived one.
        assert counts.get(RunStatus.COMPLETED.value, 0) == 1
        assert runs_created_since(session, since=now - timedelta(hours=1)) == 1
        # Sanity: the active run really exists (guards against an over-broad filter).
        assert session.get(PipelineRun, active) is not None
