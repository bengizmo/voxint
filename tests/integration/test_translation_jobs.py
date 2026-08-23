"""Translation writer + job lifecycle (#133) over real Postgres.

Writer: monotonic per-(run, language) generations, same-language-only
supersession, exact line coverage, freshness via the source hash. Jobs:
create/skip-active/same-language refusal, claim-once, cancel semantics
(cooperative between batches), execute_job end to end with an injected fake
LLM, the source-changed race guard (an edit landing mid-generation fails the
job and leaves the previous generation current), and the post-finalize
auto-hook's gating.
"""

import uuid
from collections.abc import Sequence

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import set_correction
from voxint.clients.llm import LLMError
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    RunTranslation,
    TranscriptSegment,
    TranslationJob,
    TranslationJobStatus,
)
from voxint.enrichment.translation_jobs import (
    SOURCE_CHANGED_ERROR,
    TranslationJobError,
    claim_job,
    create_job,
    execute_job,
    request_cancel,
    translation_needed,
)
from voxint.enrichment.translations import (
    TranslationError,
    current_translation,
    current_translations,
    load_translation_source,
    record_translation,
    translation_source_hash,
)


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"llm_enabled": True}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def seed_run(
    session: Session,
    *,
    texts: Sequence[str] = ("Hello there.", "Goodbye now."),
    detected_language: str | None = "en",
) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id,
        status=RunStatus.COMPLETED.value,
        detected_language=detected_language,
    )
    session.add(run)
    session.flush()
    for index, text in enumerate(texts):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index * 5),
                end_seconds=float(index * 5 + 5),
                raw_text=text,
                diarization_label="S0",
            )
        )
    session.commit()
    return run.id


def record_spanish(
    session: Session, run_id: uuid.UUID, *, suffix: str = ""
) -> RunTranslation:
    from datetime import UTC, datetime, timedelta

    source = load_translation_source(session, run_id)
    now = datetime.now(tz=UTC)
    return record_translation(
        session,
        source=source,
        target_language="es",
        translated={line.line_index: f"ES:{line.text}{suffix}" for line in source.lines},
        model="test-model",
        producer="translation.llm",
        producer_version="1",
        started_at=now,
        completed_at=now + timedelta(seconds=1),
    )


class FakeLLM:
    """Answers every batch correctly by echoing numbered lines; optional
    side-effect hook runs once before the first reply (to simulate an operator
    edit landing mid-generation)."""

    def __init__(
        self, prefix: str = "ES:", *, bodies: Sequence[object] = (), on_first_call=None
    ) -> None:
        self._prefix = prefix
        self._bodies = list(bodies)
        self._on_first_call = on_first_call
        self.calls = 0

    def chat_json(self, messages: object) -> dict[str, object]:
        assert isinstance(messages, Sequence)
        if self.calls == 0 and self._on_first_call is not None:
            self._on_first_call()
        if self.calls < len(self._bodies):
            body = self._bodies[self.calls]
            self.calls += 1
            if isinstance(body, Exception):
                raise body
            assert isinstance(body, dict)
            return body
        self.calls += 1
        prompt = str(messages[-1].content)  # type: ignore[attr-defined]
        entries = []
        for row in prompt.splitlines():
            if row.startswith("[") and "]" in row:
                idx, _, rest = row.partition("] ")
                entries.append({"i": int(idx[1:]), "text": f"{self._prefix}{rest}"})
        return {"translations": entries}


class TestWriter:
    def test_generations_monotonic_and_supersession_same_language_only(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            first = record_spanish(session, run_id)
            source = load_translation_source(session, run_id)
            from datetime import UTC, datetime

            now = datetime.now(tz=UTC)
            french = record_translation(
                session,
                source=source,
                target_language="fr",
                translated={ln.line_index: f"FR:{ln.text}" for ln in source.lines},
                model="test-model",
                producer="translation.llm",
                producer_version="1",
                started_at=now,
                completed_at=now,
            )
            second = record_spanish(session, run_id, suffix=" v2")
            session.commit()

            assert (first.generation, second.generation) == (1, 2)
            assert french.generation == 1  # independent counter per language
            session.expire_all()
            assert (
                session.get(RunTranslation, first.id).superseded_by_translation_id == second.id
            )
            assert session.get(RunTranslation, french.id).superseded_by_translation_id is None
            heads = {t.target_language: t for t in current_translations(session, run_id)}
            assert heads["es"].id == second.id
            assert heads["fr"].id == french.id

    def test_incomplete_or_invented_coverage_refused(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from datetime import UTC, datetime

        with session_factory() as session:
            run_id = seed_run(session)
            source = load_translation_source(session, run_id)
            now = datetime.now(tz=UTC)
            for translated in (
                {0: "ES:only the first line"},  # missing line 1
                {0: "ES:a", 1: "ES:b", 7: "ES:phantom"},  # unknown line
                {0: "ES:a", 1: "   "},  # empty for non-empty source
            ):
                with pytest.raises(TranslationError):
                    record_translation(
                        session,
                        source=source,
                        target_language="es",
                        translated=translated,
                        model="m",
                        producer="p",
                        producer_version="1",
                        started_at=now,
                        completed_at=now,
                    )

    def test_correction_flips_freshness(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            head = record_spanish(session, run_id)
            session.commit()
            fresh_hash = translation_source_hash(load_translation_source(session, run_id))
            assert head.source_content_hash == fresh_hash
            assert translation_needed(session, run_id, "es") is False

            segment = session.execute(
                select(TranscriptSegment).where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
            ).scalar_one()
            set_correction(session, segment=segment, text="Hello there, corrected.")
            session.commit()
            stale_hash = translation_source_hash(load_translation_source(session, run_id))
            assert stale_hash != fresh_hash
            assert translation_needed(session, run_id, "es") is True


class TestJobs:
    def _one_job(
        self, session_factory: sessionmaker[Session], run_id: uuid.UUID, target: str = "es"
    ) -> uuid.UUID:
        with session_factory() as session:
            job, already = create_job(
                session, pipeline_run_id=run_id, target_language=target,
                settings=make_settings(),
            )
            assert job is not None and not already
            session.commit()
            return job.id

    def test_create_requires_gates_transcript_and_known_language(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            with pytest.raises(TranslationJobError, match="LLM enablement"):
                create_job(
                    session, pipeline_run_id=run_id, target_language="es",
                    settings=make_settings(llm_enabled=False),
                )
            with pytest.raises(TranslationJobError, match="unknown target language"):
                create_job(
                    session, pipeline_run_id=run_id, target_language="klingon",
                    settings=make_settings(),
                )
            with pytest.raises(TranslationJobError, match="no transcript"):
                bare_media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
                session.add(bare_media)
                session.flush()
                bare = PipelineRun(
                    media_item_id=bare_media.id, status=RunStatus.COMPLETED.value
                )
                session.add(bare)
                session.flush()
                create_job(
                    session, pipeline_run_id=bare.id, target_language="es",
                    settings=make_settings(),
                )

    def test_create_refuses_same_language(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session, detected_language="es")
            with pytest.raises(TranslationJobError, match="already in"):
                create_job(
                    session, pipeline_run_id=run_id, target_language="ES",
                    settings=make_settings(),
                )

    def test_create_skips_active_duplicate(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        self._one_job(session_factory, run_id)
        with session_factory() as session:
            job, already = create_job(
                session, pipeline_run_id=run_id, target_language="es",
                settings=make_settings(),
            )
            assert job is None and already
            # A different language is its own slot.
            other, other_active = create_job(
                session, pipeline_run_id=run_id, target_language="fr",
                settings=make_settings(),
            )
            assert other is not None and not other_active

    def test_claim_is_exactly_once_and_cancel_queued_wins(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id)
        with session_factory() as session:
            assert claim_job(session, job_id) is not None
        with session_factory() as session:
            assert claim_job(session, job_id) is None  # duplicate delivery no-ops
        queued = self._one_job(session_factory, run_id, target="fr")
        with session_factory() as session:
            assert request_cancel(session, queued)
            session.commit()
        with session_factory() as session:
            assert claim_job(session, queued) is None
            row = session.get(TranslationJob, queued)
            assert row.status == TranslationJobStatus.CANCELLED.value

    def test_execute_success_records_translation(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id)
        execute_job(session_factory, job_id, settings=make_settings(), llm=FakeLLM())
        with session_factory() as session:
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.SUCCEEDED.value
            head = current_translation(session, run_id, "es")
            assert head is not None and job.translation_id == head.id
            assert head.source_language == "en"
            assert head.model == "gpt-4o-mini"  # the snapshotted default model
            texts = [entry["text"] for entry in head.lines]
            assert texts == ["ES:Hello there.", "ES:Goodbye now."]
            assert [entry["i"] for entry in head.lines] == [0, 1]
            assert head.lines[0]["source_text"] == "Hello there."
            assert translation_needed(session, run_id, "es") is False

    def test_rerun_supersedes_via_new_job(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        first = self._one_job(session_factory, run_id)
        execute_job(session_factory, first, settings=make_settings(), llm=FakeLLM())
        second = self._one_job(session_factory, run_id)
        execute_job(
            session_factory, second, settings=make_settings(), llm=FakeLLM(prefix="ES2:")
        )
        with session_factory() as session:
            head = current_translation(session, run_id, "es")
            assert head.generation == 2
            assert head.lines[0]["text"] == "ES2:Hello there."

    def test_execute_failed_reply_is_honest(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id)
        # Both attempts of the (single) batch fail, then both single-line
        # bisections of each line fail — irreducible.
        execute_job(
            session_factory,
            job_id,
            settings=make_settings(llm_attempts_per_batch=1),
            llm=FakeLLM(bodies=[LLMError("endpoint down")] * 8),
        )
        with session_factory() as session:
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.FAILED.value
            assert "could not be translated" in (job.error or "")
            assert current_translation(session, run_id, "es") is None

    def test_execute_gates_recheck(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id)
        execute_job(
            session_factory, job_id, settings=make_settings(llm_enabled=False), llm=FakeLLM()
        )
        with session_factory() as session:
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.FAILED.value
            assert "disabled" in (job.error or "")

    def test_cancel_between_batches_wins(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._one_job(session_factory, run_id)

        def cancel_mid_generation() -> None:
            with session_factory() as inner:
                assert request_cancel(inner, job_id)
                inner.commit()

        # Two single-line batches (max_segments=1); the cancel lands during the
        # first call, so the between-batch check stops the second.
        llm = FakeLLM(on_first_call=cancel_mid_generation)
        execute_job(
            session_factory,
            job_id,
            settings=make_settings(llm_batch_max_segments=1),
            llm=llm,
        )
        with session_factory() as session:
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.CANCELLED.value
            assert current_translation(session, run_id, "es") is None
        assert llm.calls == 1

    def test_source_changed_mid_generation_fails_and_keeps_previous(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        # A valid earlier generation that must survive the race.
        first = self._one_job(session_factory, run_id)
        execute_job(session_factory, first, settings=make_settings(), llm=FakeLLM())

        def edit_mid_generation() -> None:
            with session_factory() as inner:
                segment = inner.execute(
                    select(TranscriptSegment).where(
                        TranscriptSegment.pipeline_run_id == run_id,
                        TranscriptSegment.segment_index == 0,
                    )
                ).scalar_one()
                set_correction(inner, segment=segment, text="Edited while translating.")
                inner.commit()

        second = self._one_job(session_factory, run_id)
        execute_job(
            session_factory,
            second,
            settings=make_settings(),
            llm=FakeLLM(prefix="RACE:", on_first_call=edit_mid_generation),
        )
        with session_factory() as session:
            job = session.get(TranslationJob, second)
            assert job.status == TranslationJobStatus.FAILED.value
            assert job.error == SOURCE_CHANGED_ERROR
            head = current_translation(session, run_id, "es")
            # The pre-race generation is still current (now honestly stale).
            assert head is not None and head.generation == 1
            assert head.lines[0]["text"] == "ES:Hello there."
            assert translation_needed(session, run_id, "es") is True


class TestAutogenerateHook:
    def test_enqueues_when_configured(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voxint.worker import tasks

        delays: list[str] = []
        monkeypatch.setattr(tasks.translate_run, "delay", delays.append)
        with session_factory() as session:
            run_id = seed_run(session)
        settings = make_settings(
            translation_autogenerate=True, translation_target_language="es"
        )
        tasks._autogenerate_translation(session_factory, run_id, settings)
        with session_factory() as session:
            jobs = session.query(TranslationJob).all()
            assert [j.target_language for j in jobs] == ["es"]
        assert len(delays) == 1

    def test_gating_skips_and_never_raises(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voxint.worker import tasks

        delays: list[str] = []
        monkeypatch.setattr(tasks.translate_run, "delay", delays.append)
        with session_factory() as session:
            run_id = seed_run(session)
            spanish_run = seed_run(session, detected_language="es")
        # Autogenerate off → no-op.
        tasks._autogenerate_translation(
            session_factory, run_id, make_settings(translation_target_language="es")
        )
        # No target language → no-op (env validator forbids the combo, so gate
        # on the resolved value directly).
        tasks._autogenerate_translation(session_factory, run_id, make_settings())
        # LLM off → no-op.
        tasks._autogenerate_translation(
            session_factory,
            run_id,
            make_settings(
                llm_enabled=False,
                translation_autogenerate=True,
                translation_target_language="es",
            ),
        )
        # Detected language already matches the target → no-op.
        tasks._autogenerate_translation(
            session_factory,
            spanish_run,
            make_settings(translation_autogenerate=True, translation_target_language="es"),
        )
        # An unknown run → the failure is logged and swallowed, never raised.
        tasks._autogenerate_translation(
            session_factory,
            uuid.uuid4(),
            make_settings(translation_autogenerate=True, translation_target_language="es"),
        )
        with session_factory() as session:
            assert session.query(TranslationJob).count() == 0
        assert delays == []

    def test_fresh_translation_skips_requeue(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voxint.worker import tasks

        delays: list[str] = []
        monkeypatch.setattr(tasks.translate_run, "delay", delays.append)
        with session_factory() as session:
            run_id = seed_run(session)
            record_spanish(session, run_id)
            session.commit()
        tasks._autogenerate_translation(
            session_factory,
            run_id,
            make_settings(translation_autogenerate=True, translation_target_language="es"),
        )
        with session_factory() as session:
            assert session.query(TranslationJob).count() == 0
        assert delays == []


class TestCancelHardening:
    """Lifecycle hardening: client-init failure, unexpected-exception honesty,
    the finalize-time cancel check with a single batch, and the deadline-aware
    force-cancel arm (the asset-job precedents, held here too)."""

    def _claimed_job(
        self, session_factory: sessionmaker[Session], run_id: uuid.UUID
    ) -> uuid.UUID:
        with session_factory() as session:
            job, _ = create_job(
                session, pipeline_run_id=run_id, target_language="es",
                settings=make_settings(),
            )
            assert job is not None
            session.commit()
            job_id = job.id
        with session_factory() as session:
            assert claim_job(session, job_id) is not None
        return job_id

    def test_malformed_base_url_fails_not_stuck_running(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # httpx.InvalidURL at construction (the asset-job precedent): the bad
        # URL is frozen into the config snapshot at create, and building the
        # worker's own client (llm=None) must fail INSIDE the boundary — the
        # job lands FAILED, never stranded RUNNING.
        bad_settings = make_settings(llm_base_url="http://[::1")
        with session_factory() as session:
            run_id = seed_run(session)
            job, _ = create_job(
                session,
                pipeline_run_id=run_id,
                target_language="es",
                settings=bad_settings,
            )
            assert job is not None
            session.commit()
            job_id = job.id
        execute_job(session_factory, job_id, settings=bad_settings, llm=None)
        with session_factory() as session:
            row = session.get(TranslationJob, job_id)
            assert row.status == TranslationJobStatus.FAILED.value
            assert row.error == (
                "LLM endpoint could not be initialized"
                " (check the LLM endpoint setting or LLM_BASE_URL)"
            )
            assert row.translation_id is None

    def test_unexpected_exception_is_honest_and_bounded(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        class ExplodingLLM:
            def chat_json(self, messages: object) -> dict[str, object]:
                raise RuntimeError("secret internals: http://10.0.0.1/v1 said no")

        with session_factory() as session:
            run_id = seed_run(session)
        with session_factory() as session:
            job, _ = create_job(
                session, pipeline_run_id=run_id, target_language="es",
                settings=make_settings(),
            )
            session.commit()
            jid = job.id
        execute_job(session_factory, jid, settings=make_settings(), llm=ExplodingLLM())
        with session_factory() as session:
            row = session.get(TranslationJob, jid)
            assert row.status == TranslationJobStatus.FAILED.value
            # Closed vocabulary: the classification, never the message text.
            assert row.error == "unexpected error (RuntimeError) — see worker logs"

    def test_cancel_during_the_only_batch_wins_at_finalize(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)

        def cancel_mid_call() -> None:
            with session_factory() as inner:
                jid = inner.execute(select(TranslationJob.id)).scalar_one()
                assert request_cancel(inner, jid)
                inner.commit()

        with session_factory() as session:
            job, _ = create_job(
                session, pipeline_run_id=run_id, target_language="es",
                settings=make_settings(),
            )
            session.commit()
            jid = job.id
        # One batch only: the between-batch check never fires, so the
        # finalize-time cancel check must catch the flag instead.
        execute_job(
            session_factory,
            jid,
            settings=make_settings(),
            llm=FakeLLM(on_first_call=cancel_mid_call),
        )
        with session_factory() as session:
            row = session.get(TranslationJob, jid)
            assert row.status == TranslationJobStatus.CANCELLED.value
            assert current_translation(session, run_id, "es") is None

    def test_stale_running_job_is_force_cancelled(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from sqlalchemy import text as sql_text

        with session_factory() as session:
            run_id = seed_run(session)
        job_id = self._claimed_job(session_factory, run_id)
        with session_factory() as session:
            # Backdate BOTH stamps (started_at >= created_at CHECK) past the
            # attempts x timeout + grace bound.
            session.execute(
                sql_text(
                    "UPDATE translation_jobs SET"
                    " created_at = now() - interval '1 day',"
                    " started_at = now() - interval '1 day'"
                    " WHERE id = :id"
                ),
                {"id": str(job_id)},
            )
            session.commit()
        with session_factory() as session:
            assert request_cancel(session, job_id) is True
            session.commit()
        with session_factory() as session:
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.CANCELLED.value
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
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.RUNNING.value  # live executor owns it
            assert job.cancel_requested is True

    def test_cancel_on_terminal_job_returns_false(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            job, _ = create_job(
                session, pipeline_run_id=run_id, target_language="es",
                settings=make_settings(),
            )
            assert job is not None
            session.commit()
            job_id = job.id
        execute_job(session_factory, job_id, settings=make_settings(), llm=FakeLLM())
        with session_factory() as session:
            assert request_cancel(session, job_id) is False
