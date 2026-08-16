"""Run-level asset writer + job lifecycle (#41) over real Postgres.

Writer: monotonic per-(run, kind) generations, same-kind-only supersession,
idempotent replay vs conflicting key reuse, staleness via the source hash.
Jobs: create/skip-active, claim-once, cancel semantics, and execute_job end
to end with an injected fake LLM — including failure isolation (one kind
failing records no asset and leaves the others' assets standing).
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.clients.llm import LLMError
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunAssetJob,
    RunAssetJobStatus,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
    TranscriptSegment,
)
from voxint.enrichment.asset_jobs import (
    RunAssetJobError,
    claim_job,
    create_jobs,
    execute_job,
    kinds_needing_generation,
    request_cancel,
)
from voxint.enrichment.review import ConflictingReplayError
from voxint.enrichment.run_assets import (
    latest_assets,
    load_source,
    record_asset,
    source_content_hash,
)

NOW = datetime.now(tz=UTC)


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"llm_enabled": True, "enrichment_run_assets_enabled": True}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def seed_run(session: Session, *, text: str = "Hello, I am Joanne from Acme Corp.") -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    session.add(
        TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=5.0,
            raw_text=text,
            diarization_label="S0",
        )
    )
    session.commit()
    return run.id


def record_summary(
    session: Session, run_id: uuid.UUID, *, key: str, summary: str = "An abstract."
) -> RunEnrichmentAsset:
    source = load_source(session, run_id)
    return record_asset(
        session,
        source=source,
        kind=RunAssetKind.SUMMARY,
        payload={"summary": summary},
        payload_schema_version=1,
        producer="run_assets.llm",
        producer_version="1",
        model="test-model",
        idempotency_key=key,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
    )


class TestWriter:
    def test_generations_monotonic_and_supersession_same_kind_only(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            source = load_source(session, run_id)
            first = record_summary(session, run_id, key="k1")
            topics = record_asset(
                session,
                source=source,
                kind=RunAssetKind.TOPICS,
                payload={"topics": [{"label": "Widgets"}]},
                payload_schema_version=1,
                producer="run_assets.llm",
                producer_version="1",
                model="test-model",
                idempotency_key="k-topics",
                started_at=NOW,
                completed_at=NOW + timedelta(seconds=1),
            )
            second = record_summary(session, run_id, key="k2", summary="Newer abstract.")
            session.commit()

            assert (first.generation, second.generation) == (1, 2)
            assert topics.generation == 1  # independent counter per kind
            session.expire_all()
            assert session.get(RunEnrichmentAsset, first.id).superseded_by_asset_id == second.id
            # The other kind is untouched by a summary regeneration.
            assert session.get(RunEnrichmentAsset, topics.id).superseded_by_asset_id is None
            current = latest_assets(session, run_id)
            assert current["summary"].id == second.id
            assert current["topics"].id == topics.id

    def test_replay_adopts_and_conflict_raises(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            first = record_summary(session, run_id, key="same-key")
            session.commit()
            replay = record_summary(session, run_id, key="same-key")
            assert replay.id == first.id
            with pytest.raises(ConflictingReplayError):
                record_summary(session, run_id, key="same-key", summary="Different.")

    def test_source_hash_marks_staleness(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            asset = record_summary(session, run_id, key="k1")
            session.commit()
            assert asset.source_content_hash == source_content_hash(load_source(session, run_id))
            assert kinds_needing_generation(session, run_id) == (
                RunAssetKind.TOPICS,
                RunAssetKind.ENTITY_MENTIONS,
            )
            # Enhancement changes what a regeneration would read → stale.
            segment = session.query(TranscriptSegment).filter_by(pipeline_run_id=run_id).one()
            segment.enhanced_text = "Hello, I am Joanne from Acme Corporation."
            session.commit()
            assert asset.source_content_hash != source_content_hash(load_source(session, run_id))
            assert RunAssetKind.SUMMARY in kinds_needing_generation(session, run_id)


class FakeLLM:
    """Returns canned bodies per call; a list entry may be an exception."""

    def __init__(self, bodies: Sequence[object]) -> None:
        self._bodies = list(bodies)
        self.calls = 0
        # System message content per call, so tests can assert the #11
        # summary_context fragment reaches the prompt.
        self.systems: list[str] = []

    def chat_json(self, messages: object) -> dict[str, object]:
        assert isinstance(messages, Sequence)
        self.systems.append(str(messages[0].content))  # type: ignore[attr-defined]
        body = self._bodies[self.calls]
        self.calls += 1
        if isinstance(body, Exception):
            raise body
        assert isinstance(body, dict)
        return body


SUMMARY_BODY = {"summary": "A short abstract of the conversation."}
TOPICS_BODY = {"topics": [{"label": "Widgets", "confidence": 0.9}]}
MENTIONS_BODY = {
    "mentions": [
        {
            "surface": "Acme Corp",
            "kind": "organization",
            "occurrences": [{"segment_index": 0, "quote": "Acme Corp"}],
        }
    ]
}


class TestJobs:
    def test_create_jobs_requires_gates_and_transcript(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            with pytest.raises(RunAssetJobError, match="disabled"):
                create_jobs(
                    session,
                    pipeline_run_id=run_id,
                    kinds=(RunAssetKind.SUMMARY,),
                    settings=Settings(_env_file=None),
                )
            with pytest.raises(RunAssetJobError, match="unknown pipeline run"):
                create_jobs(
                    session,
                    pipeline_run_id=uuid.uuid4(),
                    kinds=(RunAssetKind.SUMMARY,),
                    settings=make_settings(),
                )

    def test_create_jobs_honors_row_llm_disable_over_env(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # The enrichment gate resolves enablement row-over-env (issue #10). The env
        # feature flag (enrichment_run_assets_enabled) itself requires env
        # llm_enabled=true, so the reachable, security-relevant divergence is a UI
        # DISABLE: env has LLM on, the operator turns it off in the UI, and no
        # further enrichment jobs may be enqueued — no restart, no leaked calls.
        from voxint.app_settings import get_or_create

        with session_factory() as session:
            run_id = seed_run(session)
            # Baseline: row enabled (matching env) -> gate open.
            row = get_or_create(session, llm_enabled_default=True)
            row.llm_enabled = True
            session.flush()
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=make_settings(llm_enabled=True),
            )
            assert len(created) == 1
            # UI disable (row False) now closes the gate despite env LLM on.
            row.llm_enabled = False
            session.flush()
            with pytest.raises(RunAssetJobError, match="disabled"):
                create_jobs(
                    session,
                    pipeline_run_id=run_id,
                    kinds=(RunAssetKind.TOPICS,),
                    settings=make_settings(llm_enabled=True),
                )

    def test_create_jobs_skips_active_kind(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            settings = make_settings()
            created, skipped = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=settings,
            )
            session.commit()
            assert [j.asset_kind for j in created] == ["summary"] and not skipped
            # "Generate all" with summary still active: only summary skips.
            created, skipped = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=tuple(RunAssetKind),
                settings=settings,
            )
            session.commit()
            assert sorted(j.asset_kind for j in created) == ["entity_mentions", "topics"]
            assert skipped == [RunAssetKind.SUMMARY]

    def test_claim_is_exactly_once_and_cancel_queued_wins(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY, RunAssetKind.TOPICS),
                settings=make_settings(),
            )
            session.commit()
            summary_job, topics_job = created
        with session_factory() as session:
            assert claim_job(session, summary_job.id) is not None
            assert claim_job(session, summary_job.id) is None  # duplicate delivery
        with session_factory() as session:
            assert request_cancel(session, topics_job.id) is True
            session.commit()
        with session_factory() as session:
            row = session.get(RunAssetJob, topics_job.id)
            assert row.status == RunAssetJobStatus.CANCELLED.value
            assert claim_job(session, topics_job.id) is None  # cancel wins the race

    def _one_job(
        self, session_factory: sessionmaker[Session], run_id: uuid.UUID, kind: RunAssetKind
    ) -> uuid.UUID:
        with session_factory() as session:
            created, _ = create_jobs(
                session, pipeline_run_id=run_id, kinds=(kind,), settings=make_settings()
            )
            session.commit()
            return created[0].id

    def test_execute_job_success_records_asset(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id, RunAssetKind.ENTITY_MENTIONS)
        execute_job(session_factory, job_id, settings=make_settings(), llm=FakeLLM([MENTIONS_BODY]))
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.SUCCEEDED.value
            assert job.error is None
            asset = session.get(RunEnrichmentAsset, job.asset_id)
            assert asset.asset_kind == "entity_mentions"
            assert asset.payload["mentions"][0]["occurrences"][0]["start_char"] == 24
            assert asset.config["model"] == "gpt-4o-mini"
            assert asset.config["truncated"] is False

    def test_execute_job_injects_run_pack_summary_context(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """The run's frozen #11 pack ``summary_context`` fragment reaches the
        producer's system prompt; a run without one leaves it out (#11)."""
        from voxint.db.models import PipelineRun
        from voxint.domain_packs.base import DomainPack

        fragment = "This series covers amateur radio astronomy."
        with session_factory() as session:
            run_id = seed_run(session)
            run = session.get(PipelineRun, run_id)
            run.domain_pack = DomainPack(
                name="radio", prompt_fragments={"summary_context": fragment}
            ).to_mapping()
            session.commit()
        job_id = self._one_job(session_factory, run_id, RunAssetKind.SUMMARY)
        llm = FakeLLM([SUMMARY_BODY])
        execute_job(session_factory, job_id, settings=make_settings(), llm=llm)
        assert fragment in llm.systems[0]
        assert "advisory" in llm.systems[0]

        # A legacy run (no snapshot) keeps the bare system prompt.
        with session_factory() as session:
            legacy_run = seed_run(session)
        legacy_job = self._one_job(session_factory, legacy_run, RunAssetKind.SUMMARY)
        legacy_llm = FakeLLM([SUMMARY_BODY])
        execute_job(session_factory, legacy_job, settings=make_settings(), llm=legacy_llm)
        assert fragment not in legacy_llm.systems[0]
        assert "advisory" not in legacy_llm.systems[0]

    def test_failure_isolation_between_kinds(self, session_factory: sessionmaker[Session]) -> None:
        """One kind failing records no asset, consumes no generation, and
        leaves the other kinds' assets standing (the issue's core invariant)."""
        with session_factory() as session:
            run_id = seed_run(session)
        settings = make_settings()
        summary_id = self._one_job(session_factory, run_id, RunAssetKind.SUMMARY)
        execute_job(session_factory, summary_id, settings=settings, llm=FakeLLM([SUMMARY_BODY]))
        topics_id = self._one_job(session_factory, run_id, RunAssetKind.TOPICS)
        execute_job(
            session_factory,
            topics_id,
            settings=settings,
            llm=FakeLLM([LLMError("HTTP 500: secret endpoint body")]),
        )
        with session_factory() as session:
            topics_job = session.get(RunAssetJob, topics_id)
            assert topics_job.status == RunAssetJobStatus.FAILED.value
            # Classification only — endpoint response bodies never persist.
            assert topics_job.error == "HTTP 500"
            assert topics_job.asset_id is None
            current = latest_assets(session, run_id)
            assert set(current) == {"summary"}  # the summary asset stands
            assert current["summary"].generation == 1

    def test_execute_job_failed_reply_is_honest(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id, RunAssetKind.TOPICS)
        execute_job(
            session_factory,
            job_id,
            settings=make_settings(),
            llm=FakeLLM([{"topics": []}]),
        )
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.FAILED.value
            assert "no usable topics" in job.error

    def test_execute_job_gates_recheck(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id, RunAssetKind.SUMMARY)
        # Gates shut down between enqueue and execution.
        execute_job(
            session_factory,
            job_id,
            settings=Settings(_env_file=None),
            llm=FakeLLM([SUMMARY_BODY]),
        )
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.FAILED.value
            assert "disabled" in job.error

    def test_cancel_that_races_the_llm_call_wins(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id, RunAssetKind.SUMMARY)

        class CancellingLLM:
            def chat_json(self, messages: object) -> dict[str, object]:
                # The operator cancels while the call is in flight.
                with session_factory() as inner:
                    request_cancel(inner, job_id)
                    inner.commit()
                return dict(SUMMARY_BODY)

        execute_job(session_factory, job_id, settings=make_settings(), llm=CancellingLLM())
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.CANCELLED.value
            assert job.asset_id is None
            assert latest_assets(session, run_id) == {}

    def test_rerun_supersedes_via_new_job(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        settings = make_settings()
        first = self._one_job(session_factory, run_id, RunAssetKind.SUMMARY)
        execute_job(session_factory, first, settings=settings, llm=FakeLLM([SUMMARY_BODY]))
        second = self._one_job(session_factory, run_id, RunAssetKind.SUMMARY)
        execute_job(
            session_factory,
            second,
            settings=settings,
            llm=FakeLLM([{"summary": "Regenerated abstract."}]),
        )
        with session_factory() as session:
            current = latest_assets(session, run_id)
            assert current["summary"].generation == 2
            assert current["summary"].payload["summary"] == "Regenerated abstract."


class TestAutogenerateHook:
    def test_hash_skip_and_best_effort(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voxint.worker import tasks

        delays: list[str] = []
        monkeypatch.setattr(tasks.generate_run_asset, "delay", delays.append)
        with session_factory() as session:
            run_id = seed_run(session)
            # A fresh summary asset → only the two missing kinds get jobs.
            record_summary(session, run_id, key="auto-k1")
            session.commit()
        settings = make_settings(enrichment_run_assets_autogenerate=True)
        tasks._autogenerate_run_assets(session_factory, run_id, settings)
        with session_factory() as session:
            kinds = sorted(j.asset_kind for j in session.query(RunAssetJob).all())
        assert kinds == ["entity_mentions", "topics"]
        assert len(delays) == 2

    def test_disabled_or_failing_never_raises(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voxint.worker import tasks

        delays: list[str] = []
        monkeypatch.setattr(tasks.generate_run_asset, "delay", delays.append)
        with session_factory() as session:
            run_id = seed_run(session)
        # Autogenerate off → no-op.
        tasks._autogenerate_run_assets(session_factory, run_id, make_settings())
        # A run with no transcript → the failure is logged and swallowed,
        # never propagated into the pipeline task.
        tasks._autogenerate_run_assets(
            session_factory,
            uuid.uuid4(),
            make_settings(enrichment_run_assets_autogenerate=True),
        )
        assert delays == []


class TestCancelHardening:
    """Review-driven lifecycle cases: stale force-cancel, terminal CAS,
    snapshot-executes, and force-cancel racing the executor."""

    def _claimed_job(self, session_factory: sessionmaker[Session], run_id: uuid.UUID) -> uuid.UUID:
        with session_factory() as session:
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=make_settings(),
            )
            session.commit()
            job_id = created[0].id
        with session_factory() as session:
            assert claim_job(session, job_id) is not None
        return job_id

    def test_stale_running_job_is_force_cancelled(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from sqlalchemy import text as sql_text

        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._claimed_job(session_factory, run_id)
        with session_factory() as session:
            # Backdate BOTH stamps (started_at >= created_at CHECK) past the
            # llm_timeout + grace bound.
            session.execute(
                sql_text(
                    "UPDATE run_asset_jobs SET"
                    " created_at = now() - interval '1 hour',"
                    " started_at = now() - interval '1 hour'"
                    " WHERE id = :id"
                ),
                {"id": str(job_id)},
            )
            session.commit()
        with session_factory() as session:
            assert request_cancel(session, job_id) is True
            session.commit()
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.CANCELLED.value
            assert job.finished_at is not None

    def test_fresh_running_job_is_not_force_cancelled(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._claimed_job(session_factory, run_id)
        with session_factory() as session:
            assert request_cancel(session, job_id) is True
            session.commit()
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.RUNNING.value  # live executor owns it
            assert job.cancel_requested is True

    def test_cancel_on_terminal_job_returns_false(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=make_settings(),
            )
            session.commit()
            job_id = created[0].id
        execute_job(session_factory, job_id, settings=make_settings(), llm=FakeLLM([SUMMARY_BODY]))
        with session_factory() as session:
            assert session.get(RunAssetJob, job_id).status == RunAssetJobStatus.SUCCEEDED.value
            assert request_cancel(session, job_id) is False

    def test_finish_never_overwrites_terminal_and_resolves_cancel(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from voxint.enrichment.asset_jobs import _finish

        with session_factory() as session:
            run_id = seed_run(session)
        # A late FAILED verdict must not overwrite an already-CANCELLED row.
        job_id = self._claimed_job(session_factory, run_id)
        with session_factory() as session:
            from sqlalchemy import text as sql_text

            session.execute(
                sql_text(
                    "UPDATE run_asset_jobs SET status = 'cancelled',"
                    " cancel_requested = true, finished_at = now() WHERE id = :id"
                ),
                {"id": str(job_id)},
            )
            session.commit()
        with session_factory() as session:
            _finish(session, job_id, status=RunAssetJobStatus.FAILED, error="late failure")
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.CANCELLED.value
            assert job.error is None
        # A FAILED verdict on a RUNNING row with the cancel flag set resolves
        # to CANCELLED — the operator asked for exactly that outcome.
        second = self._claimed_job(session_factory, run_id)
        with session_factory() as session:
            request_cancel(session, second)
            session.commit()
        with session_factory() as session:
            _finish(session, second, status=RunAssetJobStatus.FAILED, error="llm broke")
        with session_factory() as session:
            job = session.get(RunAssetJob, second)
            assert job.status == RunAssetJobStatus.CANCELLED.value

    def test_force_cancelled_job_never_records_an_asset(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)

        class ForceCancellingLLM:
            def chat_json(self, messages: object) -> dict[str, object]:
                # Simulate a force-cancel resolving the row mid-call.
                from sqlalchemy import text as sql_text

                with session_factory() as inner:
                    inner.execute(
                        sql_text(
                            "UPDATE run_asset_jobs SET status = 'cancelled',"
                            " cancel_requested = true, finished_at = now()"
                            " WHERE pipeline_run_id = :rid"
                        ),
                        {"rid": str(run_id)},
                    )
                    inner.commit()
                return dict(SUMMARY_BODY)

        with session_factory() as session:
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=make_settings(),
            )
            session.commit()
            job_id = created[0].id
        execute_job(session_factory, job_id, settings=make_settings(), llm=ForceCancellingLLM())
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            assert job.status == RunAssetJobStatus.CANCELLED.value
            assert job.asset_id is None
            assert latest_assets(session, run_id) == {}

    def test_snapshot_executes_not_live_settings(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=make_settings(llm_model="model-a"),
            )
            session.commit()
            job_id = created[0].id
        # Settings changed between enqueue and execution — the snapshot wins.
        execute_job(
            session_factory,
            job_id,
            settings=make_settings(llm_model="model-b"),
            llm=FakeLLM([SUMMARY_BODY]),
        )
        with session_factory() as session:
            job = session.get(RunAssetJob, job_id)
            asset = session.get(RunEnrichmentAsset, job.asset_id)
            assert asset.model == "model-a"
            assert asset.config["model"] == "model-a"
