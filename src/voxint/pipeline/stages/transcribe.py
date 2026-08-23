"""Transcribe stage: ASR over the normalized audio, raw text preserved forever."""

import uuid
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from voxint.clients.base import TranscriptionSegment, TranscriptionWord
from voxint.db.models import PipelineRun, TranscriptSegment
from voxint.pipeline.stages.context import StageContext, normalized_audio_path

# Whisper biases toward — and can hallucinate from — an over-long initial_prompt,
# so bound it. Terms are dropped whole (never mid-word).
INITIAL_PROMPT_MAX_CHARS = 2000


def _initial_prompt(vocabulary: tuple[str, ...]) -> str | None:
    """Render the run's vocabulary as a bounded whisper ``initial_prompt``.

    ``vocabulary`` is already deduped/ordered upstream; this joins terms with
    ", " up to the char cap (whole terms only) and returns None when empty. A
    single over-cap term is skipped rather than aborting the rest, so one
    pathological entry can't starve every later term.
    """
    parts: list[str] = []
    length = 0
    for term in vocabulary:
        addition = len(term) + (2 if parts else 0)  # ", " separator between terms
        if length + addition > INITIAL_PROMPT_MAX_CHARS:
            continue  # this term doesn't fit; a shorter later term still might
        parts.append(term)
        length += addition
    return ", ".join(parts) if parts else None


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the temporal intersection of two intervals (0 if disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def bucket_words(
    segments: tuple[TranscriptionSegment, ...],
    words: tuple[TranscriptionWord, ...],
) -> list[list[TranscriptionWord]]:
    """Assign each flat word to exactly one segment, returning a per-segment list
    in transcript order.

    A word joins the segment it overlaps most in time. Ties (equal overlap,
    including the common zero-overlap case for a word that falls in a gap between
    segments) break toward the **nearest** segment, then the **earlier** index —
    fully deterministic, so the same transcription always buckets identically.
    Every word lands somewhere; none are dropped. Words with no segments to land
    in are discarded (there is nothing to attach them to)."""
    buckets: list[list[TranscriptionWord]] = [[] for _ in segments]
    if not segments:
        return buckets
    for word in words:
        best_index = 0
        best_overlap = -1.0
        best_gap = float("inf")
        for index, seg in enumerate(segments):
            overlap = _overlap(
                word.start_seconds, word.end_seconds, seg.start_seconds, seg.end_seconds
            )
            # Distance from the word to this segment's interval; 0 when they touch
            # or overlap. Only the tie-breaker among equal-overlap candidates.
            gap = max(
                0.0,
                seg.start_seconds - word.end_seconds,
                word.start_seconds - seg.end_seconds,
            )
            if overlap > best_overlap or (overlap == best_overlap and gap < best_gap):
                best_index, best_overlap, best_gap = index, overlap, gap
        buckets[best_index].append(word)
    return buckets


def _word_payload(words: list[TranscriptionWord]) -> list[dict[str, Any]]:
    """The JSONB array stored on a segment — one object per word, in order.

    Always a list (possibly empty): the NULL-vs-array decision is made per run at
    the call site, so ``words IS NULL`` cleanly means "this run had no word
    timing" and never "this segment happened to bucket zero words"."""
    return [
        {
            "start": w.start_seconds,
            "end": w.end_seconds,
            "word": w.word,
            "confidence": w.confidence,
        }
        for w in words
    ]


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    audio = normalized_audio_path(session, run_id, ctx.media_root)
    # Render the prompt once, decode, then stamp what whisper actually saw: the
    # operator's live-unioned glossary is captured nowhere else (issue #123).
    # Stamping AFTER the decode keeps provenance structurally honest — a run that
    # never transcribed records no prompt, independent of the engine's rollback.
    # Idempotent under at-least-once stage re-runs: the same effective vocabulary
    # renders the same prompt, and a re-run reflects the final decode.
    prompt = _initial_prompt(ctx.vocabulary)
    result = ctx.asr.transcribe(audio, initial_prompt=prompt)
    pipeline_run = session.get(PipelineRun, run_id)
    if pipeline_run is not None:
        pipeline_run.initial_prompt = prompt
        # Detected-language provenance (issue #124): stamp what whisper actually
        # reported, after a successful decode, in the transcript's transaction —
        # same honesty rule as initial_prompt above. A re-run reflects its own
        # final decode; a failed decode never reaches this line, so a committed
        # stamp survives a later failed attempt's rollback.
        pipeline_run.detected_language = result.language
        pipeline_run.detected_language_probability = result.language_probability
    session.execute(
        delete(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
    )
    word_buckets = bucket_words(result.segments, result.words)
    # A run either has word timing (every segment stores an array, empty if none
    # bucketed there) or it doesn't (every segment stores SQL NULL) — decided
    # once here, not per segment, so NULL never ambiguously means "empty bucket".
    run_has_words = bool(result.words)
    for index, segment in enumerate(result.segments):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run_id,
                segment_index=index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                raw_text=segment.text,
                suspect=segment.suspect,
                confidence=segment.confidence,
                words=_word_payload(word_buckets[index]) if run_has_words else None,
            )
        )
