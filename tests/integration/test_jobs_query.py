"""The Jobs read models against real Postgres (Console 2.0 P5, #160).

``jobs_query`` is pure SQL over a session, so its truth rules — the stage strip's
queued/active bucketing and the auxiliary-job normalization — need a live
database (group-by, distinct, the real check constraints), which is why these
live in the integration suite alongside the other query tests.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from voxint.api.jobs_query import recent_aux_jobs, stage_activity
from voxint.api.stats_query import run_status_counts
from voxint.db.models import (
    STAGE_ORDER,
    EmbeddingJob,
    MediaItem,
    PipelineRun,
    ResearchJob,
    RunAssetJob,
    RunStatus,
    Speaker,
    Stage,
    StageRun,
    StageStatus,
    TranslationJob,
)

_HASH = "a" * 64


def _make_run(
    session: Session,
    *,
    status: RunStatus,
    current_stage: str | None = None,
    archived: bool = False,
    stages: Iterable[dict[str, object]] = (),
) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id, status=status.value, current_stage=current_stage
    )
    if archived:
        run.archived_at = datetime.now(UTC)
    session.add(run)
    session.flush()
    for spec in stages:
        session.add(StageRun(pipeline_run_id=run.id, **spec))
    session.commit()
    return run.id


# ---- stage_activity ---------------------------------------------------------


def test_stage_activity_returns_every_stage_in_order(
    session_factory: sessionmaker[Session],
) -> None:
    """Always one entry per stage, in canonical order, even with an empty DB."""
    with session_factory() as session:
        activity = stage_activity(session)
    assert [a.stage for a in activity] == [s.value for s in STAGE_ORDER]
    assert all(a.queued == 0 and a.active == 0 for a in activity)


def test_stage_activity_queued_buckets_reconcile_with_status_counts(
    session_factory: sessionmaker[Session],
) -> None:
    """Queued runs bucket by current_stage (NULL -> first stage) and the buckets
    partition the queued runs, so their sum equals the queued status count."""
    with session_factory() as session:
        _make_run(session, status=RunStatus.QUEUED, current_stage=Stage.TRANSCRIBE.value)
        _make_run(session, status=RunStatus.QUEUED, current_stage=None)
        _make_run(session, status=RunStatus.QUEUED, current_stage=Stage.TRANSCRIBE.value)
        with session_factory() as read:
            activity = {a.stage: a for a in stage_activity(read)}
            counts = run_status_counts(read)

    assert activity[Stage.TRANSCRIBE.value].queued == 2
    # The current_stage-less queued run falls under the first stage (acquire).
    assert activity[STAGE_ORDER[0].value].queued == 1
    assert sum(a.queued for a in activity.values()) == counts.get(
        RunStatus.QUEUED.value, 0
    )


def test_stage_activity_active_is_distinct_and_reconciles(
    session_factory: sessionmaker[Session],
) -> None:
    """Active per stage counts DISTINCT live runs with a running attempt; a run
    with several running attempts is one active run, and the active total equals
    the running status count when the fixture is consistent."""
    with session_factory() as session:
        # One running run with TWO running attempts at the same stage: distinct
        # counting must fold these into a single active run.
        _make_run(
            session,
            status=RunStatus.RUNNING,
            current_stage=Stage.DIARIZE_EMBED.value,
            stages=[
                {
                    "stage": Stage.DIARIZE_EMBED.value,
                    "status": StageStatus.RUNNING.value,
                    "attempt": 1,
                },
                {
                    "stage": Stage.DIARIZE_EMBED.value,
                    "status": StageStatus.RUNNING.value,
                    "attempt": 2,
                },
            ],
        )
        with session_factory() as read:
            activity = {a.stage: a for a in stage_activity(read)}
            counts = run_status_counts(read)

    assert activity[Stage.DIARIZE_EMBED.value].active == 1
    assert sum(a.active for a in activity.values()) == counts.get(
        RunStatus.RUNNING.value, 0
    )


def test_stage_activity_excludes_stale_running_on_dead_runs(
    session_factory: sessionmaker[Session],
) -> None:
    """A running StageRun left behind by a terminal or archived run does not
    inflate the active strip — only live, non-archived runs count."""
    running_stage = {
        "stage": Stage.TRANSCRIBE.value,
        "status": StageStatus.RUNNING.value,
    }
    with session_factory() as session:
        # Terminal run with a leftover running attempt.
        _make_run(
            session,
            status=RunStatus.COMPLETED,
            current_stage=Stage.TRANSCRIBE.value,
            stages=[dict(running_stage)],
        )
        # Archived (still "running") run with a running attempt.
        _make_run(
            session,
            status=RunStatus.RUNNING,
            current_stage=Stage.TRANSCRIBE.value,
            archived=True,
            stages=[dict(running_stage)],
        )
        with session_factory() as read:
            activity = {a.stage: a for a in stage_activity(read)}

    assert activity[Stage.TRANSCRIBE.value].active == 0


def test_stage_activity_active_anchors_on_current_stage(
    session_factory: sessionmaker[Session],
) -> None:
    """A stale running attempt left at a stage the run has already moved past
    must not make the run appear active at two stages: active is anchored on the
    run's current_stage, so one live run contributes to exactly one stage."""
    with session_factory() as session:
        # Run is now at DIARIZE_EMBED, but a stale running TRANSCRIBE attempt
        # from before the handoff is still on the ledger.
        _make_run(
            session,
            status=RunStatus.RUNNING,
            current_stage=Stage.DIARIZE_EMBED.value,
            stages=[
                {
                    "stage": Stage.TRANSCRIBE.value,
                    "status": StageStatus.RUNNING.value,
                    "attempt": 1,
                },
                {
                    "stage": Stage.DIARIZE_EMBED.value,
                    "status": StageStatus.RUNNING.value,
                    "attempt": 1,
                },
            ],
        )
        with session_factory() as read:
            activity = {a.stage: a for a in stage_activity(read)}

    assert activity[Stage.DIARIZE_EMBED.value].active == 1
    # The stale prior-stage attempt does NOT count the run active at transcribe.
    assert activity[Stage.TRANSCRIBE.value].active == 0
    assert sum(a.active for a in activity.values()) == 1


# ---- recent_aux_jobs --------------------------------------------------------


def _seed_aux_jobs(session: Session) -> uuid.UUID:
    """One job in each family, with controlled created_at for ordering.

    Returns the run id the run-scoped families share.
    """
    run_id = _make_run(session, status=RunStatus.COMPLETED)
    speaker = Speaker(display_name="Test Speaker")
    session.add(speaker)
    session.flush()
    base = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)

    asset = RunAssetJob(
        pipeline_run_id=run_id,
        asset_kind="summary",
        status="succeeded",
        config={},
    )
    asset.created_at = base
    translation = TranslationJob(
        pipeline_run_id=run_id,
        target_language="es",
        status="running",
        config={},
        source_content_hash=_HASH,
    )
    translation.created_at = base + timedelta(minutes=1)
    embedding = EmbeddingJob(
        pipeline_run_id=run_id,
        embedding_space="minilm-v1",
        status="failed",
        source_content_hash=_HASH,
    )
    embedding.created_at = base + timedelta(minutes=2)
    research = ResearchJob(
        speaker_id=speaker.id,
        pipeline_run_id=None,
        status="queued",
        budget={},
    )
    research.created_at = base + timedelta(minutes=3)
    session.add_all([asset, translation, embedding, research])
    session.commit()
    return run_id


def test_recent_aux_jobs_normalizes_all_families(
    session_factory: sessionmaker[Session],
) -> None:
    """All four families surface in one read model, newest first, with the
    run-scoped and speaker-scoped targets normalized (research has no run)."""
    with session_factory() as session:
        run_id = _seed_aux_jobs(session)
        with session_factory() as read:
            jobs = recent_aux_jobs(read)

    by_family = {j.family: j for j in jobs}
    assert set(by_family) == {"asset", "translation", "embedding", "research"}
    # Newest first (research seeded last).
    assert [j.family for j in jobs] == ["research", "embedding", "translation", "asset"]
    # succeeded, not completed — the aux vocab differs from RunStatus.
    assert by_family["asset"].status == "succeeded"
    assert by_family["asset"].detail == "summary"
    assert by_family["translation"].detail == "es"
    assert by_family["embedding"].detail == "minilm-v1"
    # Research is speaker-scoped with a nullable run link.
    assert by_family["research"].pipeline_run_id is None
    assert by_family["research"].speaker_id is not None
    # The three run-scoped families carry the run.
    assert by_family["asset"].pipeline_run_id == run_id
    assert by_family["research"].detail == ""


def test_recent_aux_jobs_limit_truncates_merged_list(
    session_factory: sessionmaker[Session],
) -> None:
    """The overall limit truncates the merged, recency-sorted list."""
    with session_factory() as session:
        _seed_aux_jobs(session)
        with session_factory() as read:
            jobs = recent_aux_jobs(read, limit=2)

    assert len(jobs) == 2
    # The two newest across families.
    assert [j.family for j in jobs] == ["research", "embedding"]


def test_recent_aux_jobs_per_family_bound(
    session_factory: sessionmaker[Session],
) -> None:
    """per_family bounds each family's read before the merge, so one busy family
    cannot crowd the others out of the list."""
    with session_factory() as session:
        run_id = _make_run(session, status=RunStatus.COMPLETED)
        base = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)
        # Three asset jobs (distinct kinds so the partial-active index is happy).
        for offset, kind in enumerate(("summary", "topics", "entity_mentions")):
            job = RunAssetJob(
                pipeline_run_id=run_id,
                asset_kind=kind,
                status="failed",
                config={},
            )
            job.created_at = base + timedelta(minutes=offset)
            session.add(job)
        # One translation, older than the assets.
        translation = TranslationJob(
            pipeline_run_id=run_id,
            target_language="es",
            status="failed",
            config={},
            source_content_hash=_HASH,
        )
        translation.created_at = base - timedelta(minutes=1)
        session.add(translation)
        session.commit()
        with session_factory() as read:
            jobs = recent_aux_jobs(read, limit=10, per_family=1)

    # Only the newest asset survives the per-family bound; the translation stays.
    assert len(jobs) == 2
    assert {j.family for j in jobs} == {"asset", "translation"}
    assert next(j for j in jobs if j.family == "asset").detail == "entity_mentions"


def test_recent_aux_jobs_ties_break_by_id_deterministically(
    session_factory: sessionmaker[Session],
) -> None:
    """Jobs sharing a commit timestamp order by id desc, so the merge and the
    per-family cutoff are deterministic rather than arbitrary."""
    with session_factory() as session:
        run_id = _make_run(session, status=RunStatus.COMPLETED)
        same = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)
        for kind in ("summary", "topics", "entity_mentions"):
            job = RunAssetJob(
                pipeline_run_id=run_id, asset_kind=kind, status="failed", config={}
            )
            job.created_at = same
            session.add(job)
        session.commit()
        with session_factory() as read:
            first = [j.id for j in recent_aux_jobs(read)]
            second = [j.id for j in recent_aux_jobs(read)]

    assert first == second
    # Descending id order for the tied rows.
    assert first == sorted(first, reverse=True)
