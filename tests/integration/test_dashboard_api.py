"""The operator dashboard (issue #13) end to end against real Postgres.

The route is a thin HTML view over ``stats_query.collect_stats`` (already
covered in ``tests/unit/test_stats_query.py``); these tests pin the wiring the
unit tests cannot see — auth, that the rendered page and the htmx fragment carry
the aggregated numbers, the ``?since=`` throughput window, and that a malformed
``?since=`` degrades to the 24h default instead of 500-ing.
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
from voxint.api.presentation import humanize_status
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
    """A fixed spread of runs/stages/speakers for the dashboard to aggregate."""
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    with session_factory() as session:
        # Two runs in the transient AWAITING_ADJUDICATION status. These exercise
        # the status table and the raw voxint_runs gauge, but they are NOT the
        # review backlog: a successful pipeline terminates COMPLETED, so the queue
        # (issue #117) is keyed on COMPLETED runs with unresolved speaker labels,
        # not on this status.
        make_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        make_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        # One completed run with a finished transcribe (30s) and a failed diarize.
        # It carries an unresolved diarization label, so it is the one genuine
        # review-backlog entry -> review_backlog_count == 1.
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
        # One old run outside a short window (for the ?since= narrowing test).
        make_run(session, status=RunStatus.FAILED, created_at=old)
        # Two enrolled speakers -> roster size == 2.
        session.add(Speaker(display_name="Alice"))
        session.add(Speaker(display_name="Bob"))
        session.commit()


def test_dashboard_renders_aggregated_numbers(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text
    # Full page: nav chrome present, active tab marked.
    assert 'href="/dashboard" aria-current="page"' in body
    # Roster (2 speakers) surfaces as a summary stat card (issue #91); the value
    # is tied to its label, and each card keeps its unit as text so the number is
    # never bare. (The review backlog is no longer a stat card — issue #117 Phase C
    # promoted it to the static "Continue review (N)" task card, and a dedicated
    # test pins that count against the queue eligibility.)
    assert _stat_value(body, "Roster") == 2
    assert "enrolled speaker(s)" in body
    # Status table zero-fills the enum: a status with no runs renders a 0 row.
    # The LABEL is humanized (issue #56) but the pill's CSS class stays the raw
    # enum so it keeps its colour — assert both, since keeping the class raw is
    # the load-bearing invariant of the humanization.
    assert re.search(r"Queued</span></td>\s*<td>0</td>", body)
    assert 'class="pill queued"' in body
    # Status is never conveyed by colour ALONE (issue #64): every rendered status
    # pill carries its state word as text inside the span, so a colour-blind or
    # AT user still reads the status. Assert the humanized label sits inside each
    # pill span across the enum-backed rows, not just one.
    for status in RunStatus:
        label = humanize_status(status.value)
        # Guard against humanize_status regressing to "" — an empty label would let
        # the pill go colour-only and the regex below would still match (review).
        assert label.strip(), f"humanize_status({status.value}) returned empty"
        assert re.search(
            rf'class="pill {re.escape(status.value)}">{re.escape(label)}</span>', body
        ), f"status pill for {status.value} missing its adjacent text label"
    # All three metrics tables (statuses, timing, failures — all seeded here) each
    # scroll inside their own container on narrow screens.
    assert body.count('class="table-wrap"') == 3
    # Stage timing binds the seeded values, not just the stage names: transcribe
    # ran 30s, the failed diarize_embed attempt 5s (finished attempts count for
    # duration regardless of terminal status). Stage names render humanized.
    assert "Transcribe" in body
    assert "30.00s" in body
    assert "5.00s" in body
    # Stage timing carries a supplemental relative mini-bar (issue #91): the bar
    # is aria-hidden (the exact seconds above are the real cue), and the slowest
    # seeded stage (transcribe, 30s) scales to a full-width bar. Pins both the
    # presence and the max_avg width math so a broken bar can't ship green.
    assert 'class="minibar" aria-hidden="true"' in body
    assert "width: 100.0%" in body
    # The one seeded diarize_embed failure renders in the failures table, with the
    # humanized stage label.
    assert re.search(r"Diarize &amp; embed</td>\s*<td>1</td>", body)


def test_dashboard_htmx_returns_fragment_without_chrome(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/dashboard", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    body = resp.text
    # Fragment carries the metric numbers but not the page chrome / nav.
    assert _stat_value(body, "Roster") == 2
    assert "enrolled speaker(s)" in body
    assert "<nav" not in body
    assert "<h1>Dashboard</h1>" not in body
    # Consistency policy (issue #117 Phase C): the review backlog is NOT in the
    # 15s-polled fragment — it lives only on the static task card — so the poll can
    # never refresh a count that contradicts the "Continue review (N)" card.
    assert "Review backlog" not in body
    assert 'class="task-cards"' not in body
    # The fragment must NOT re-emit the polling container or its hx-get: the
    # outer #dashboard-metrics div lives in the full page and persists across
    # innerHTML swaps, so a nested one would duplicate the id and the 15s timer.
    assert "hx-get" not in body
    assert 'id="dashboard-metrics"' not in body


def test_dashboard_since_narrows_created_window(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    # Default (24h) excludes the 3-day-old run: 3 recent runs created.
    default_body = client.get("/dashboard").text
    assert _stat_value(default_body, "Runs in window") == 3
    # A 7-day window includes the old run too (all 4).
    wide_body = client.get("/dashboard", params={"since": "7d"}).text
    # The window line differs; assert the wide window reports one more run than
    # the default by checking both counts render distinctly.
    assert _created_count(default_body) == 3
    assert _created_count(wide_body) == 4
    # The 15s poll URL carries the active window so it survives the refresh; the
    # default page polls the bare route.
    assert 'hx-get="/dashboard?since=7d"' in wide_body
    assert 'hx-get="/dashboard"' in default_body


def test_dashboard_window_selector_marks_the_active_preset(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    # In-page window picker (issue #56): a GET <select name="since"> whose options
    # are exactly the strings parse_since accepts. Default (no ?since=) selects 24h.
    default_body = client.get("/dashboard").text
    assert '<select name="since"' in default_body
    assert re.search(r'<option value="24h" selected>', default_body)
    # Choosing 7d marks that option selected instead and re-seeds the poll URL.
    wide_body = client.get("/dashboard", params={"since": "7d"}).text
    assert re.search(r'<option value="7d" selected>', wide_body)
    assert "24h\" selected" not in wide_body


def test_dashboard_window_selector_echoes_valid_custom_window(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A valid ?since= that is not one of the presets (parse_since accepts 48h) is
    # the *active* window, so the selector must echo it as selected — never show a
    # shorter preset than the data reflects (issue #56 review).
    seed_snapshot(session_factory)
    body = client.get("/dashboard", params={"since": "48h"}).text
    assert '<option value="48h" selected>Custom (48h)</option>' in body
    # The presets are not falsely marked selected in that case.
    assert '<option value="24h" selected>' not in body


def test_dashboard_malformed_since_shows_notice_and_drops_poll_param(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    body = client.get("/dashboard", params={"since": "not-a-window"}).text
    # Operator is told the window was ignored, and the rejected value is not
    # echoed into the poll URL (each poll would just re-take the fallback).
    assert "Unrecognized" in body
    assert 'hx-get="/dashboard"' in body
    assert "not-a-window" not in body


def test_dashboard_matches_metrics_snapshot(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The core invariant: dashboard numbers agree with /metrics on one seed."""
    seed_snapshot(session_factory)
    dash = client.get("/dashboard").text
    metrics = client.get("/metrics").text
    # runs_created (default 24h window) — same figure on both surfaces.
    assert _created_count(dash) == 3
    assert "voxint_runs_created_24h 3" in metrics
    # Roster lines up with its Prometheus gauge.
    assert "voxint_roster_speakers 2" in metrics
    assert _stat_value(dash, "Roster") == 2
    # The raw awaiting_adjudication status gauge (2) is deliberately NOT the
    # review backlog: the backlog is the /review queue's eligibility count —
    # COMPLETED runs with unresolved labels (issue #117), one here. The two
    # numbers are different by design; /metrics exposes the raw status, the
    # dashboard's "Continue review (N)" task card the actionable queue.
    assert 'voxint_runs{status="awaiting_adjudication"} 2' in metrics
    assert re.search(r'<span class="task-title">Continue review \(1\)', dash)


def test_dashboard_malformed_since_degrades_to_default(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/dashboard", params={"since": "not-a-window"})
    assert resp.status_code == 200
    # Falls back to the 24h default (3 recent runs), not a 500 or the wide count.
    assert _created_count(resp.text) == 3


def _created_count(body: str) -> int:
    """Pull the runs-created count out of the rendered HTML via its stable hook."""
    match = re.search(r'data-metric="runs-created"[^>]*>\s*(\d+)', body)
    assert match is not None, "runs-created metric hook missing from dashboard render"
    return int(match.group(1))


def _stat_value(body: str, label: str) -> int:
    """Pull a summary stat-card's value by its label (issue #91). Ties the label
    to its adjacent value, so a card can't silently show the wrong figure."""
    match = re.search(
        rf'class="stat-label">{re.escape(label)}</dt>\s*'
        rf'<dd class="stat-value"[^>]*>\s*(\d+)',
        body,
    )
    assert match is not None, f"stat card {label!r} missing from dashboard render"
    return int(match.group(1))


def _completed_run_id(session_factory: sessionmaker[Session]) -> uuid.UUID:
    """The single COMPLETED, non-archived run the snapshot seed creates."""
    from sqlalchemy import select

    with session_factory() as session:
        return session.execute(
            select(PipelineRun.id).where(
                PipelineRun.status == RunStatus.COMPLETED.value,
                PipelineRun.archived_at.is_(None),
            )
        ).scalar_one()


def test_dashboard_task_cards_render_first(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Phase C (issue #117): three task cards open the dashboard, above the
    demoted metrics — the first-run operator sees what to do before any numbers."""
    seed_snapshot(session_factory)
    completed = _completed_run_id(session_factory)
    body = client.get("/dashboard").text
    # The task list carries exactly the three first-run tasks, as links.
    assert 'class="task-cards"' in body
    assert 'href="/runs#add-media"' in body  # Add audio -> Runs Add-media section
    assert 'href="/review"' in body  # Continue review -> the queue
    assert f'href="/runs/{completed}"' in body  # Last finished run -> run detail
    # Task cards sit ABOVE the demoted metrics disclosure, not below it.
    cards_at = body.index('class="task-cards"')
    details_at = body.index('<details class="run-details">')
    assert cards_at < details_at
    # And above the metrics polling container specifically.
    assert cards_at < body.index('id="dashboard-metrics"')


def test_dashboard_continue_review_count_matches_queue(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The Continue-review headline shares the /review queue's own eligibility
    count (issue #117): it cannot drift from adjudication_queue, and the seed's one
    unresolved COMPLETED run makes it 1 (not the 2 awaiting_adjudication runs)."""
    from voxint.adjudication.resolver import adjudication_queue

    seed_snapshot(session_factory)
    with session_factory() as session:
        eligible = len(adjudication_queue(session))
    assert eligible == 1
    body = client.get("/dashboard").text
    # The task-card count equals the queue eligibility — the single source of this
    # number, shown once, statically.
    assert re.search(
        rf'href="/review">\s*<span class="task-title">Continue review \({eligible}\)',
        body,
    )


def test_dashboard_last_finished_run_empty_state(client: TestClient) -> None:
    """With nothing completed, the Last-finished card is honest, quiet text — not a
    dead link (issue #117 Phase C)."""
    # No snapshot: the onboarded client has no completed runs.
    body = client.get("/dashboard").text
    # The empty card is a non-interactive .is-empty block carrying the honest copy.
    assert re.search(
        r'<div class="task-link is-empty">\s*'
        r'<span class="task-title">Last finished run</span>\s*'
        r'<span class="task-note">No recordings have finished yet\.</span>',
        body,
    )
    # It is not rendered as a run-detail link (nothing to point at).
    assert 'href="/runs/None"' not in body


def test_dashboard_metrics_behind_disclosure_with_carveouts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Metrics move inside a "Show run details" <details>, but the ?since= error
    and the window <select> stay OUTSIDE it — an error and an input are never
    hidden (issue #117 Phase C, codex carve-outs)."""
    seed_snapshot(session_factory)
    body = client.get("/dashboard", params={"since": "not-a-window"}).text
    details_at = body.index('<details class="run-details">')
    assert '<summary>Show run details</summary>' in body
    # The stat cards and detail tables are inside the disclosure.
    assert body.index('id="dashboard-metrics"') > details_at
    assert body.index('class="stat-cards"') > details_at
    assert body.index('<h2>Runs by status</h2>') > details_at
    # The invalid-?since= notice renders OUTSIDE (before) the disclosure.
    assert "Unrecognized" in body
    assert body.index("Unrecognized") < details_at
    # The time-window control renders OUTSIDE (before) the disclosure too.
    assert body.index('<select name="since"') < details_at


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
    window so the dashboard / metrics / CLI all report *active* runs."""
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
