"""The #758 clean-up job lifecycle over real Postgres (slice 3).

Creation gates (LLM enablement, D6 language, an active job), the guarded
claim, cancel before, during and after the model, the broken-model and
malformed-line rules, the source-changed race guard, snapshot validation and
the finalize stamp that sets ``status`` and ``cleanup_id`` together.
"""

import uuid
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_cleanup_writer import seed
from tests.integration.test_translation_jobs import make_settings
from voxint.adjudication.review_state import set_correction
from voxint.clients.llm import ChatMessage, LLMError, LLMReplyError
from voxint.db.models import (
    AppSettings,
    CleanupJob,
    CleanupJobStatus,
    MediaItem,
    PipelineRun,
    RunCleanup,
    RunStatus,
    TranscriptSegment,
)
from voxint.enrichment.cleanup_jobs import (
    SNAPSHOT_INVALID_ERROR,
    SOURCE_CHANGED_ERROR,
    CleanupJobError,
    active_or_last_job,
    claim_job,
    create_job,
    execute_job,
    language_refusal,
    request_cancel,
    stale_queued_job_ids,
)
from voxint.enrichment.cleanups import CleanupError, current_cleanup
from voxint.enrichment.producers.cleanup_llm import CLEANUP_PROMPT_VERSION
from voxint.export.filler_lists import PRESET_VERSION

# Line index -> what the fake model proposes; unlisted lines come back as sent.
PROPOSALS = {
    0: "I think it's fine.",
    1: "so we went",
    2: "to Paris, yesterday.",
}


class FakeCleanupLLM:
    """Answers each numbered batch; ``respond`` may override per call."""

    def __init__(
        self,
        proposals: dict[int, str] | None = None,
        *,
        respond: Callable[[dict[int, str]], object] | None = None,
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self._proposals = PROPOSALS if proposals is None else proposals
        self._respond = respond
        self._on_call = on_call
        self.calls = 0

    def chat_json(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        self.calls += 1
        if self._on_call is not None:
            self._on_call(self.calls)
        lines: dict[int, str] = {}
        for row in messages[-1].content.splitlines():
            if row.startswith("[") and "] " in row:
                index, _, rest = row.partition("] ")
                lines[int(index[1:])] = rest
        if self._respond is not None:
            reply = self._respond(lines)
            if isinstance(reply, Exception):
                raise reply
            assert isinstance(reply, dict)
            return reply
        return {
            "lines": [
                {"i": i, "text": self._proposals.get(i, source)} for i, source in lines.items()
            ]
        }


def _job(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, **settings: object
) -> uuid.UUID:
    with session_factory() as session:
        job, already = create_job(
            session, pipeline_run_id=run_id, settings=make_settings(**settings)
        )
        assert job is not None and not already
        session.commit()
        return job.id


def _row(session_factory: sessionmaker[Session], job_id: uuid.UUID) -> CleanupJob:
    with session_factory() as session:
        row = session.get(CleanupJob, job_id)
        assert row is not None
        session.expunge(row)
        return row


def _set_language(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, code: str | None
) -> None:
    with session_factory() as session:
        session.execute(
            update(PipelineRun).where(PipelineRun.id == run_id).values(detected_language=code)
        )
        session.commit()


def _edit_first_segment(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as inner:
        segment = inner.execute(
            select(TranscriptSegment).where(
                TranscriptSegment.pipeline_run_id == run_id,
                TranscriptSegment.segment_index == 0,
            )
        ).scalar_one()
        set_correction(inner, segment=segment, text="Edited while cleaning.")
        inner.commit()


@pytest.fixture
def run_id(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        return seed(session)


# ------------------------------------------------------------------ creation


def test_create_snapshots_the_endpoint_knobs_and_filler_list(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(
        session_factory, run_id, llm_model="m-1", llm_batch_max_segments=3,
        voxint_fillers_add="you know",
    )
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.QUEUED.value
    assert row.config["model"] == "m-1"
    assert row.config["llm_batch_max_segments"] == 3
    knobs = {"base_url", "llm_timeout_seconds", "llm_attempts_per_batch", "llm_batch_max_chars"}
    assert knobs <= set(row.config)
    assert row.config["llm_disable_thinking"] is False
    fillers = row.config["filler_list"]
    assert fillers["preset_version"] == PRESET_VERSION
    assert "you know" in fillers["phrases"]
    assert len(row.source_content_hash) == 64


def test_create_refuses_when_the_llm_is_off(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    off = make_settings(llm_enabled=False)
    with session_factory() as session, pytest.raises(CleanupJobError, match="model enabled"):
        create_job(session, pipeline_run_id=run_id, settings=off)


def test_create_refuses_a_run_without_a_transcript(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        bare = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(bare)
        session.flush()
        with pytest.raises(CleanupJobError, match="nothing to clean up"):
            create_job(session, pipeline_run_id=bare.id, settings=make_settings())


def test_create_refuses_a_non_english_run_and_names_the_limit(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    _set_language(session_factory, run_id, "fr")
    with session_factory() as session, pytest.raises(CleanupJobError) as exc:
        create_job(session, pipeline_run_id=run_id, settings=make_settings())
    assert str(exc.value) == (
        "clean-up works on English transcripts only, and this run was detected as French (fr)"
    )


@pytest.mark.parametrize("code", [None, "en", " EN "])
def test_create_allows_english_or_undetected(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, code: str | None
) -> None:
    _set_language(session_factory, run_id, code)
    assert _job(session_factory, run_id)
    assert language_refusal(code) is None


def test_create_refuses_an_invalid_saved_filler_list(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    with session_factory() as session:
        row = session.get(AppSettings, 1) or AppSettings(id=1)
        row.llm_enabled = True
        row.fillers_add = {"en": "not a list"}
        session.add(row)
        session.commit()
        with pytest.raises(CleanupJobError, match="filler word list is not valid"):
            create_job(session, pipeline_run_id=run_id, settings=make_settings())


def test_create_skips_while_a_job_is_active(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    first = _job(session_factory, run_id)
    with session_factory() as session:
        assert create_job(session, pipeline_run_id=run_id, settings=make_settings()) == (
            None, True,
        )
    with session_factory() as session:
        assert request_cancel(session, first)
        session.commit()
    assert _job(session_factory, run_id) != first


# ------------------------------------------------------------------ claim and cancel


def test_claim_is_exactly_once_and_a_queued_cancel_wins(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    with session_factory() as session:
        assert claim_job(session, job_id) is not None
        assert claim_job(session, job_id) is None
    other = seed_other(session_factory)
    queued = _job(session_factory, other)
    with session_factory() as session:
        assert request_cancel(session, queued)
        session.commit()
        assert claim_job(session, queued) is None
    assert _row(session_factory, queued).status == CleanupJobStatus.CANCELLED.value


def seed_other(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        return seed(session)


def test_cancel_on_a_terminal_job_returns_false(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM())
    with session_factory() as session:
        assert request_cancel(session, job_id) is False


def test_a_stale_running_job_is_force_cancelled_and_a_fresh_one_is_flagged(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    stale = _job(session_factory, run_id)
    with session_factory() as session:
        assert claim_job(session, stale) is not None
        session.execute(
            text(
                "UPDATE cleanup_jobs SET created_at = now() - interval '1 day',"
                " started_at = now() - interval '1 day' WHERE id = :id"
            ),
            {"id": str(stale)},
        )
        session.commit()
        assert request_cancel(session, stale)
        session.commit()
    row = _row(session_factory, stale)
    assert row.status == CleanupJobStatus.CANCELLED.value and row.finished_at is not None

    fresh = _job(session_factory, run_id)
    with session_factory() as session:
        assert claim_job(session, fresh) is not None
        assert request_cancel(session, fresh)
        session.commit()
    row = _row(session_factory, fresh)
    assert row.status == CleanupJobStatus.RUNNING.value and row.cancel_requested


def test_cancel_during_a_call_discards_the_reply_and_stores_nothing(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)

    def cancel(_call: int) -> None:
        with session_factory() as inner:
            assert request_cancel(inner, job_id)
            inner.commit()

    llm = FakeCleanupLLM(on_call=cancel)
    execute_job(
        session_factory, job_id, settings=make_settings(llm_batch_max_segments=1), llm=llm
    )
    assert llm.calls == 1
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.CANCELLED.value and row.cleanup_id is None
    with session_factory() as session:
        assert current_cleanup(session, run_id) is None


def test_cancel_after_the_last_call_wins_at_finalize(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxint.enrichment import cleanup_jobs

    job_id = _job(session_factory, run_id)
    real = cleanup_jobs.propose_cleanup

    def propose_then_cancel(*args: Any, **kwargs: Any) -> dict[int, str | None]:
        out = real(*args, **kwargs)
        with session_factory() as inner:
            assert request_cancel(inner, job_id)
            inner.commit()
        return out

    monkeypatch.setattr(cleanup_jobs, "propose_cleanup", propose_then_cancel)
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM())
    assert _row(session_factory, job_id).status == CleanupJobStatus.CANCELLED.value
    with session_factory() as session:
        assert current_cleanup(session, run_id) is None


# ------------------------------------------------------------------ execution


def test_success_stores_a_generation_and_links_it_in_one_stamp(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id, llm_model="m-1")
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM())
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.SUCCEEDED.value and row.error is None
    with session_factory() as session:
        head = current_cleanup(session, run_id)
        assert head is not None and head.id == row.cleanup_id
        assert [line["text"] for line in head.lines] == [
            "I think it's fine.",
            # The filler pass capitalises the new line start, as for #757 omits.
            "So we went",
            "to Paris, yesterday.",
            "Okay then.",
            "So -- we go.",
        ]
        assert head.counts["lines_changed"] == 3
        assert head.model == "m-1"
        assert head.producer == "cleanup.llm"
        assert head.idempotency_key == str(job_id)
        assert head.config["prompt_version"] == CLEANUP_PROMPT_VERSION
        assert head.config["filler_list"] == row.config["filler_list"]
        assert head.config["llm_batch_max_segments"] == row.config["llm_batch_max_segments"]
        assert head.started_at == row.started_at
        assert head.source_content_hash == row.source_content_hash


def test_a_second_job_supersedes_the_first(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    first = _job(session_factory, run_id)
    execute_job(session_factory, first, settings=make_settings(), llm=FakeCleanupLLM())
    second = _job(session_factory, run_id)
    execute_job(session_factory, second, settings=make_settings(), llm=FakeCleanupLLM({}))
    with session_factory() as session:
        head = current_cleanup(session, run_id)
        assert head is not None and head.generation == 2
        assert head.id == _row(session_factory, second).cleanup_id
        assert head.counts["lines_changed"] == 0


def test_a_duplicate_delivery_noops(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    llm = FakeCleanupLLM()
    execute_job(session_factory, job_id, settings=make_settings(), llm=llm)
    calls = llm.calls
    execute_job(session_factory, job_id, settings=make_settings(), llm=llm)
    assert llm.calls == calls
    with session_factory() as session:
        generations = select(RunCleanup.generation).where(RunCleanup.pipeline_run_id == run_id)
        assert session.scalars(generations).all() == [1]


def test_executes_from_the_snapshot_not_current_settings(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id, llm_batch_max_segments=1, llm_model="snap")
    llm = FakeCleanupLLM()
    execute_job(
        session_factory, job_id, settings=make_settings(llm_batch_max_segments=50, llm_model="now"),
        llm=llm,
    )
    # Five lines, one per batch, as the snapshot said.
    assert llm.calls == 5
    with session_factory() as session:
        head = current_cleanup(session, run_id)
        assert head is not None and head.model == "snap"


def test_malformed_lines_are_counted_and_the_rest_stored(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    def respond(lines: dict[int, str]) -> object:
        if 1 in lines:
            return LLMReplyError("completion content is not valid JSON")
        return {"lines": [{"i": i, "text": PROPOSALS.get(i, s)} for i, s in lines.items()]}

    job_id = _job(session_factory, run_id, llm_batch_max_segments=1)
    llm = FakeCleanupLLM(respond=respond)
    execute_job(session_factory, job_id, settings=make_settings(), llm=llm)
    assert _row(session_factory, job_id).status == CleanupJobStatus.SUCCEEDED.value
    with session_factory() as session:
        head = current_cleanup(session, run_id)
        assert head is not None
        assert head.counts["rejected"]["malformed"] == 1
        assert head.lines[1]["outcome"] == "rejected" and head.lines[1]["text"] == "Um, so we went"


def test_a_broken_model_fails_honestly_and_stores_nothing(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    execute_job(
        session_factory, job_id, settings=make_settings(),
        llm=FakeCleanupLLM(respond=lambda _lines: {"cleaned": []}),
    )
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error == "the model gave no usable reply for any line"
    assert row.cleanup_id is None
    with session_factory() as session:
        assert current_cleanup(session, run_id) is None


def test_a_dead_endpoint_fails_without_leaking_its_reply(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    execute_job(
        session_factory, job_id, settings=make_settings(),
        llm=FakeCleanupLLM(respond=lambda _lines: LLMError("body: secret-token")),
    )
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error is not None and "request failed" in row.error
    assert "secret-token" not in row.error


def test_an_unexpected_error_is_bounded(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    execute_job(
        session_factory, job_id, settings=make_settings(),
        llm=FakeCleanupLLM(respond=lambda _lines: RuntimeError("http://192.0.2.1 said no")),
    )
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error == "unexpected error (RuntimeError); see worker logs"


def test_a_malformed_base_url_fails_instead_of_sticking_running(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    bad = make_settings(llm_base_url="http://[::1")
    with session_factory() as session:
        job, _ = create_job(session, pipeline_run_id=run_id, settings=bad)
        assert job is not None
        session.commit()
        job_id = job.id
    execute_job(session_factory, job_id, settings=bad, llm=None)
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error is not None and row.error.startswith("LLM endpoint could not be initialized")


def test_the_llm_disabled_after_enqueue_fails_the_job(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    llm = FakeCleanupLLM()
    execute_job(session_factory, job_id, settings=make_settings(llm_enabled=False), llm=llm)
    assert llm.calls == 0
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error == "the language model was disabled after this clean-up was queued"


def test_the_language_is_rechecked_at_claim(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    job_id = _job(session_factory, run_id)
    _set_language(session_factory, run_id, "de")
    llm = FakeCleanupLLM()
    execute_job(session_factory, job_id, settings=make_settings(), llm=llm)
    assert llm.calls == 0
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error is not None and row.error.endswith("detected as German (de)")


@pytest.mark.parametrize(
    "key, value",
    [
        ("llm_batch_max_segments", 0),
        ("llm_batch_max_segments", True),
        ("llm_attempts_per_batch", "2"),
        ("llm_batch_max_chars", None),
        ("llm_timeout_seconds", -1),
        ("model", ""),
        ("llm_disable_thinking", "yes"),
        ("base_url", 3),
        ("filler_list", {"preset_version": "en-1", "words": "um", "phrases": [], "kept": []}),
        ("filler_list", None),
    ],
)
def test_an_invalid_snapshot_fails_the_job(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, key: str, value: object
) -> None:
    job_id = _job(session_factory, run_id)
    with session_factory() as session:
        job = session.get(CleanupJob, job_id)
        assert job is not None
        job.config = {**job.config, key: value}
        session.commit()
    llm = FakeCleanupLLM()
    execute_job(session_factory, job_id, settings=make_settings(), llm=llm)
    assert llm.calls == 0
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value and row.error == SNAPSHOT_INVALID_ERROR


def test_a_source_change_mid_job_fails_and_keeps_the_previous_generation(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    first = _job(session_factory, run_id)
    execute_job(session_factory, first, settings=make_settings(), llm=FakeCleanupLLM())
    second = _job(session_factory, run_id)
    execute_job(
        session_factory, second, settings=make_settings(),
        llm=FakeCleanupLLM({}, on_call=lambda _n: _edit_first_segment(session_factory, run_id)),
    )
    row = _row(session_factory, second)
    assert row.status == CleanupJobStatus.FAILED.value and row.error == SOURCE_CHANGED_ERROR
    with session_factory() as session:
        head = current_cleanup(session, run_id)
        assert head is not None and head.generation == 1
        assert head.id == _row(session_factory, first).cleanup_id


# ------------------------------------------------------------------ read helpers


def test_active_or_last_and_stale_queued(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    with session_factory() as session:
        assert active_or_last_job(session, run_id) is None
    done = _job(session_factory, run_id)
    execute_job(session_factory, done, settings=make_settings(), llm=FakeCleanupLLM())
    with session_factory() as session:
        last = active_or_last_job(session, run_id)
        assert last is not None and last.id == done
    queued = _job(session_factory, run_id)
    with session_factory() as session:
        session.execute(
            text("UPDATE cleanup_jobs SET created_at = now() - interval '1 hour' WHERE id = :id"),
            {"id": str(done)},
        )
        session.commit()
        active = active_or_last_job(session, run_id)
        assert active is not None and active.id == queued
        cutoff = session.scalar(select(text("now() + interval '1 minute'")))
        assert stale_queued_job_ids(session, cutoff=cutoff) == [queued]
        assert stale_queued_job_ids(session, cutoff=cutoff, limit=0) == []
        old = session.scalar(select(text("now() - interval '1 minute'")))
        assert stale_queued_job_ids(session, cutoff=old) == []


# ------------------------------------------------------------------ finalize failures


def test_a_cancel_between_the_last_check_and_the_stamp_rolls_back(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxint.enrichment import cleanup_jobs

    job_id = _job(session_factory, run_id)
    real = cleanup_jobs.record_cleanup

    def record_then_cancel(*args: Any, **kwargs: Any) -> RunCleanup:
        row = real(*args, **kwargs)
        with session_factory() as inner:
            assert request_cancel(inner, job_id)
            inner.commit()
        return row

    monkeypatch.setattr(cleanup_jobs, "record_cleanup", record_then_cancel)
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM())
    assert _row(session_factory, job_id).status == CleanupJobStatus.CANCELLED.value
    with session_factory() as session:
        assert current_cleanup(session, run_id) is None


@pytest.mark.parametrize(
    "error, expected",
    [
        (CleanupError("no line received a usable reply from the model"),
         "no line received a usable reply from the model"),
        (RuntimeError("internal detail"), "unexpected error (RuntimeError); see worker logs"),
    ],
)
def test_a_writer_failure_lands_on_the_row(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: str,
) -> None:
    from voxint.enrichment import cleanup_jobs

    def refuse(*_args: Any, **_kwargs: Any) -> RunCleanup:
        raise error

    job_id = _job(session_factory, run_id)
    monkeypatch.setattr(cleanup_jobs, "record_cleanup", refuse)
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM())
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value and row.error == expected
    assert row.cleanup_id is None


def test_the_worker_client_reports_an_unreachable_endpoint(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    # Port 9 (discard) refuses connections, so the real client fails fast.
    unreachable = make_settings(
        llm_base_url="http://127.0.0.1:9/v1", llm_attempts_per_batch=1, llm_timeout_seconds=2
    )
    with session_factory() as session:
        job, _ = create_job(session, pipeline_run_id=run_id, settings=unreachable)
        assert job is not None
        session.commit()
        job_id = job.id
    execute_job(session_factory, job_id, settings=unreachable, llm=None)
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.FAILED.value
    assert row.error is not None and row.error.startswith("the language model request failed")


def test_completion_stamps_follow_the_model_not_the_transaction_start(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    import time

    job_id = _job(session_factory, run_id)
    execute_job(
        session_factory, job_id, settings=make_settings(),
        llm=FakeCleanupLLM(on_call=lambda _n: time.sleep(1.1)),
    )
    row = _row(session_factory, job_id)
    with session_factory() as session:
        head = current_cleanup(session, run_id)
        assert head is not None
        assert (head.completed_at - head.started_at).total_seconds() >= 1
    assert row.finished_at is not None and row.finished_at >= head.completed_at


def test_a_failure_that_loses_to_a_cancel_drops_its_error(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxint.enrichment import cleanup_jobs

    job_id = _job(session_factory, run_id)

    def cancel_then_refuse(*_args: Any, **_kwargs: Any) -> RunCleanup:
        with session_factory() as inner:
            assert request_cancel(inner, job_id)
            inner.commit()
        raise CleanupError("refused")

    monkeypatch.setattr(cleanup_jobs, "record_cleanup", cancel_then_refuse)
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeCleanupLLM())
    row = _row(session_factory, job_id)
    assert row.status == CleanupJobStatus.CANCELLED.value and row.error is None


@pytest.mark.parametrize("config", [None, [], "x"])
def test_a_non_object_snapshot_is_refused(config: object) -> None:
    from voxint.enrichment.cleanup_jobs import _settings_from_snapshot

    with pytest.raises(CleanupJobError, match="saved settings are not valid"):
        _settings_from_snapshot(make_settings(), config)  # type: ignore[arg-type]
