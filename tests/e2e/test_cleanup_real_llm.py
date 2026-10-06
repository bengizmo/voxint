"""Real-LLM E2E for the #758 clean-up variant: gate the invariants, not the edits.

Drives one clean-up job through the real ``HttpLLMClient`` (``llm=None``, so
the worker builds its own client from the job snapshot) against the operator's
configured endpoint, over a short seeded English transcript full of fillers.

It asserts only what must hold whatever the model proposes:

1. the job succeeds and links its generation;
2. every stored line is a pure deletion of its source words: the kept keys are
   a subsequence of the source keys, no protected word was deleted, and a
   rejected or unchanged line is stored verbatim;
3. the counts add up;
4. the generation is fresh, and one operator correction stales it.

It never asserts which words were deleted: a real model's choices are not a
contract. The per-reason reject rate is printed for the maintainer (the plan's
"Rollout / risks" measurement), never blocked on.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import set_correction
from voxint.app_settings import complete_onboarding
from voxint.config import Settings
from voxint.db.models import (
    CleanupJob,
    CleanupJobStatus,
    MediaItem,
    PipelineRun,
    RunStatus,
    TranscriptSegment,
)
from voxint.enrichment.cleanup import proposal_keys
from voxint.enrichment.cleanup_jobs import create_job, execute_job
from voxint.enrichment.cleanups import current_cleanup, load_cleanup_source
from voxint.enrichment.translations import load_translation_source, translation_source_hash

from .conftest import LLMConfig

_TRANSCRIPT = (
    "Um, so, I mean, we finished the three reports yesterday.",
    "You know, the client said no to the, uh, the second option.",
    "Okay so like Maria will, um, call them on Friday.",
    "I think we're, we're basically on track, right?",
    "Uh, yeah, we can't move the 4 o'clock meeting.",
)


def _seed(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        complete_onboarding(session, llm_enabled_default=True)
        media = MediaItem(source_path=f"e2e/{uuid.uuid4().hex}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id, status=RunStatus.COMPLETED.value, detected_language="en"
        )
        session.add(run)
        session.flush()
        for index, text in enumerate(_TRANSCRIPT):
            start = float(index * 5)
            session.add(
                TranscriptSegment(
                    pipeline_run_id=run.id,
                    segment_index=index,
                    start_seconds=start,
                    end_seconds=start + 5,
                    raw_text=text,
                    diarization_label=f"S{index % 2}",
                    # Word timings give every word a #757 anchor.
                    words=[
                        {
                            "word": (" " if i else "") + word,
                            "start": start + i * 0.2,
                            "end": start + i * 0.2 + 0.2,
                        }
                        for i, word in enumerate(text.split())
                    ],
                )
            )
        session.commit()
        return run.id


def _is_subsequence(kept: list[str], source: list[str]) -> bool:
    remaining = iter(source)
    return all(key in remaining for key in kept)


def test_real_llm_cleanup_chain(
    session_factory: sessionmaker[Session],
    settings: Settings,
    cleanup_llm_config: LLMConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = _seed(session_factory)
    with session_factory() as session:
        job, _ = create_job(session, pipeline_run_id=run_id, settings=settings)
        assert job is not None
        session.commit()
        job_id = job.id

    execute_job(session_factory, job_id, settings=settings, llm=None)

    with session_factory() as session:
        row = session.get(CleanupJob, job_id)
        assert row is not None
        assert row.status == CleanupJobStatus.SUCCEEDED.value, (
            f"clean-up job did not succeed against {cleanup_llm_config.resolved_identity}:"
            f" status={row.status} error={row.error!r}"
        )
        head = current_cleanup(session, run_id)
        assert head is not None and head.id == row.cleanup_id

        source = load_cleanup_source(session, run_id)
        assert len(head.lines) == len(source.lines)
        removed = 0
        for line, frozen in zip(head.lines, source.lines, strict=True):
            assert line["source"] == frozen.line.text
            if line["outcome"] != "changed":
                assert line["text"] == line["source"] and line["deleted"] == []
                continue
            assert _is_subsequence(proposal_keys(line["text"]), proposal_keys(line["source"]))
            deleted = {(start, end) for start, end, _ls, _le in line["deleted"]}
            for word in frozen.words.words:
                if (word.anchor.token_start, word.anchor.token_end) in deleted:
                    assert not word.protected, f"protected word deleted: {word.text!r}"
                    removed += len(word.keys)
        counts = head.counts
        rejected = counts["rejected"]
        outcomes = [line["outcome"] for line in head.lines]
        assert counts["lines"] == len(head.lines)
        assert counts["lines_changed"] == outcomes.count("changed")
        assert sum(rejected.values()) == outcomes.count("rejected")
        assert counts["words_removed"] == removed

        assert head.source_content_hash == translation_source_hash(
            load_translation_source(session, run_id)
        )
        lines = [(line["outcome"], line["reason"], line["text"]) for line in head.lines]

    with capsys.disabled():
        print(
            f"\n[characterization] model={cleanup_llm_config.model}"
            f" resolved={cleanup_llm_config.resolved_identity}"
            f"\n[characterization] changed={counts['lines_changed']}/{counts['lines']}"
            f" words_removed={counts['words_removed']} rejected={rejected}"
        )
        for outcome, reason, text in lines:
            print(f"[characterization] {outcome:9} {reason or '':12} {text}")

    with session_factory() as session:
        segment = session.query(TranscriptSegment).filter_by(
            pipeline_run_id=run_id, segment_index=0
        ).one()
        set_correction(session, segment=segment, text="We finished the three reports yesterday.")
        session.commit()
        head = current_cleanup(session, run_id)
        assert head is not None
        assert head.source_content_hash != translation_source_hash(
            load_translation_source(session, run_id)
        ), "an operator correction did not stale the clean-up"
