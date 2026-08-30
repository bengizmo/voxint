"""Speaker-level statistics for corpus visualization (issue #335).

Pure computation: log-odds distinctive vocabulary (Monroe/Colaresi/Quinn
"Fightin' Words"), ego transition network, and words-per-minute. No DB or
framework imports.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass

from voxint.api.term_stats import tokenize


@dataclass
class SpeakerTermStat:
    term: str
    count: int
    log_odds: float
    z_score: float


@dataclass
class TransitionEdge:
    from_speaker_id: uuid.UUID
    to_speaker_id: uuid.UUID
    count: int


def compute_log_odds(
    target_counts: Counter[str],
    background_counts: Counter[str],
    *,
    min_count: int = 3,
    top_n: int = 30,
    alpha_sum: float = 1.0,
) -> list[SpeakerTermStat]:
    """Weighted log-odds with informative Dirichlet prior.

    Implements Monroe, Colaresi & Quinn (2008) "Fightin' Words": each prior
    weight alpha_w is proportional to the term's background frequency, not a
    flat constant. This avoids over-shrinking genuine differences for small
    speakers and under-regularizing rare words.

    Returns the top_n terms by positive z-score (distinctive TO the target
    speaker), filtered to terms with at least ``min_count`` occurrences in the
    target text.
    """
    if not target_counts or not background_counts:
        return []

    n_target = sum(target_counts.values())
    n_background = sum(background_counts.values())
    if n_target == 0 or n_background == 0:
        return []

    vocabulary = set(target_counts) | set(background_counts)
    if len(vocabulary) < 2:
        return []

    corpus_total = n_target + n_background
    results: list[SpeakerTermStat] = []

    for term in vocabulary:
        y_i = target_counts.get(term, 0)
        if y_i < min_count:
            continue
        y_j = background_counts.get(term, 0)

        corpus_freq = (y_i + y_j) / corpus_total
        alpha_w = max(alpha_sum * corpus_freq, 1e-10)
        alpha_total = alpha_sum

        n_i_plus = n_target + alpha_total
        n_j_plus = n_background + alpha_total

        denom_i = n_i_plus - y_i - alpha_w
        denom_j = n_j_plus - y_j - alpha_w
        if denom_i <= 0 or denom_j <= 0:
            continue

        log_odds_i = math.log((y_i + alpha_w) / denom_i)
        log_odds_j = math.log((y_j + alpha_w) / denom_j)
        delta = log_odds_i - log_odds_j

        sigma = math.sqrt(1.0 / (y_i + alpha_w) + 1.0 / (y_j + alpha_w))
        z = delta / sigma

        if z > 0:
            results.append(SpeakerTermStat(
                term=term,
                count=y_i,
                log_odds=round(delta, 6),
                z_score=round(z, 4),
            ))

    results.sort(key=lambda s: (-s.z_score, s.term))
    return results[:top_n]


def compute_ego_transitions(
    interval_sequences: list[list[str | None]],
    target_speaker_id: str,
) -> tuple[list[TransitionEdge], list[TransitionEdge]]:
    """Ego transition network from attributed-interval speaker sequences.

    Each sequence is one run's ordered list of canonical speaker IDs (as
    strings). ``None`` entries represent unattributed/excluded intervals and
    break the adjacency chain.

    Returns (transitions_in, transitions_out):
      - transitions_in:  edges where another speaker precedes the target
      - transitions_out: edges where the target precedes another speaker
    """
    in_counts: Counter[str] = Counter()
    out_counts: Counter[str] = Counter()

    for sequence in interval_sequences:
        collapsed = _collapse_consecutive(sequence)
        for i in range(len(collapsed) - 1):
            a, b = collapsed[i], collapsed[i + 1]
            if a is None or b is None:
                continue
            if a == target_speaker_id and b != target_speaker_id:
                out_counts[b] += 1
            elif b == target_speaker_id and a != target_speaker_id:
                in_counts[a] += 1

    target_uuid = uuid.UUID(target_speaker_id)

    transitions_in = sorted(
        [
            TransitionEdge(
                from_speaker_id=uuid.UUID(sid),
                to_speaker_id=target_uuid,
                count=count,
            )
            for sid, count in in_counts.items()
        ],
        key=lambda e: (-e.count, str(e.from_speaker_id)),
    )

    transitions_out = sorted(
        [
            TransitionEdge(
                from_speaker_id=target_uuid,
                to_speaker_id=uuid.UUID(sid),
                count=count,
            )
            for sid, count in out_counts.items()
        ],
        key=lambda e: (-e.count, str(e.to_speaker_id)),
    )

    return transitions_in, transitions_out


def _collapse_consecutive(sequence: list[str | None]) -> list[str | None]:
    """Collapse consecutive identical speaker IDs into one entry."""
    if not sequence:
        return []
    collapsed: list[str | None] = [sequence[0]]
    for item in sequence[1:]:
        if item != collapsed[-1]:
            collapsed.append(item)
    return collapsed


def compute_wpm(
    word_counts: list[int],
    durations: list[float],
    *,
    min_timed_seconds: float = 60.0,
) -> tuple[float | None, int, int]:
    """Words per minute from per-segment word counts and durations.

    Each pair (word_counts[i], durations[i]) represents one segment's word
    count and speaking duration (end_seconds - start_seconds). Zero-duration
    segments are skipped.

    Returns (wpm, timed_segments, total_segments). Returns (None, timed, N)
    when fewer than ``min_timed_seconds`` of timed speech are available.
    """
    total_segments = len(word_counts)
    total_words = 0
    total_seconds = 0.0
    timed = 0

    for wc, dur in zip(word_counts, durations, strict=True):
        if dur <= 0 or wc <= 0:
            continue
        total_words += wc
        total_seconds += dur
        timed += 1

    if total_seconds < min_timed_seconds:
        return None, timed, total_segments

    wpm = (total_words / total_seconds) * 60.0
    return round(wpm, 1), timed, total_segments


def tokenize_text(text: str) -> Counter[str]:
    """Tokenize text into a term frequency counter, reusing the corpus tokenizer."""
    return Counter(tokenize(text))
