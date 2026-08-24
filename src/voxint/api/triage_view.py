"""View-layer triage assembly shared by the review workbench and speaker research.

Fuses one enrichment candidate's signals (producer score, evidence, voice
support, cross-producer agreement) into the explainable review priority the
console renders, and builds the representative NAME-suggestion lists for the
workbench. Pure reads; the scoring math itself lives in
:mod:`voxint.enrichment.triage`.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151): the
speakers router and the review workbench both need these, and router modules
never import ``app``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import AssignmentMethod, ClaimField, Speaker, SpeakerAssignment
from voxint.enrichment.queries import (
    CandidateState,
    CandidateView,
    candidates_for_run,
)
from voxint.enrichment.triage import (
    EvidenceRef,
    TriageInputs,
    TriageScore,
    VoiceSignal,
)
from voxint.enrichment.triage import (
    score as triage_score,
)

# Rendering precedence inside one (target, value) suggestion group: a human
# decision is history and outranks a fresh proposed duplicate from a rerun
# (decided candidates are terminal and never superseded), which outranks
# superseded leftovers.
_HINT_STATE_PRECEDENCE = {
    CandidateState.ACCEPTED: 0,
    CandidateState.REJECTED: 1,
    CandidateState.PROPOSED: 2,
    CandidateState.SUPERSEDED: 3,
}


@dataclass(frozen=True)
class _VoiceRow:
    """Per-label grounded cosine facts for triage voice-support."""

    name_norm: str
    confidence: float | None
    grounded: bool


@dataclass(frozen=True)
class _HintTriage:
    """The triage a template renders beside one representative suggestion."""

    priority: float
    components: dict[str, float]
    # Proposed candidates for the same (label, value) hidden behind this
    # representative — so a decision on one producer's claim never silently
    # buries another producer's still-open proposal (#42).
    unresolved_peers: int


def _name_match_key(value: str) -> str:
    """The one normalization for matching a name candidate to its peers, to a
    voice assignment, and for representative grouping: strip + casefold.

    Deliberately NOT ``roster.normalize_display_name`` — that raises above 120
    chars, and a NAME candidate ``value`` may be longer (the column allows 4000);
    calling it in this read path would 500 the workbench. Using one key
    everywhere keeps a representative card and its agreement/voice signals about
    the same set of candidates.
    """
    return value.strip().casefold()


def _voice_by_label(session: Session, run_id: uuid.UUID) -> dict[str, _VoiceRow]:
    """Grounded cosine facts per diarization label (one cosine row per label).

    Only ``method='cosine'`` carries a roster speaker + confidence + grounding;
    ``llm_hint`` has none. **Active roster identities only** — a since-merged or
    archived speaker's stale display name must not drive voice matching (it would
    invert the signal: a false conflict against the merge target, or false
    support for a tombstone). ``UNIQUE(run, label, method)`` gives one row per
    label; the ORDER BY only makes the dict-build deterministic if that ever
    changes.
    """
    rows = session.execute(
        select(
            SpeakerAssignment.diarization_label,
            Speaker.display_name,
            SpeakerAssignment.confidence,
            SpeakerAssignment.grounded,
        )
        .join(Speaker, Speaker.id == SpeakerAssignment.speaker_id)
        .where(
            SpeakerAssignment.pipeline_run_id == run_id,
            SpeakerAssignment.method == AssignmentMethod.COSINE.value,
            Speaker.merged_into_id.is_(None),
            Speaker.deleted_at.is_(None),
        )
        .order_by(SpeakerAssignment.id)
    ).all()
    return {
        label: _VoiceRow(
            name_norm=_name_match_key(name),
            confidence=confidence,
            grounded=grounded,
        )
        for label, name, confidence, grounded in rows
    }


def _name_peer_counts(views: Sequence[CandidateView]) -> dict[tuple[str | None, str], int]:
    """Distinct producers proposing the same (label, normalized name) across
    ACTIVE candidates (proposed or accepted). Rejected/superseded never
    corroborate — computed over all views, before representative collapsing."""
    producers: dict[tuple[str | None, str], set[str]] = {}
    for view in views:
        if view.state not in (CandidateState.PROPOSED, CandidateState.ACCEPTED):
            continue
        key = (view.candidate.diarization_label, _name_match_key(view.candidate.value))
        producers.setdefault(key, set()).add(view.candidate.producer_run.producer)
    return {key: len(names) for key, names in producers.items()}


def _triage_for(
    view: CandidateView,
    *,
    voice: _VoiceRow | None,
    peer_count: int,
    authority: frozenset[str],
) -> TriageScore:
    """Fuse one candidate's signals into an explainable review priority."""
    candidate = view.candidate
    voice_signal: VoiceSignal | None = None
    if voice is not None:
        voice_signal = VoiceSignal(
            matches_value=_name_match_key(candidate.value) == voice.name_norm,
            grounded=voice.grounded,
            confidence=voice.confidence,
        )
    return triage_score(
        TriageInputs(
            field=candidate.field,
            producer=candidate.producer_run.producer,
            producer_score=candidate.score,
            producer_components=candidate.score_components or {},
            evidence=tuple(EvidenceRef(kind=e.kind, url=e.url) for e in view.evidence),
            voice=voice_signal,
            peer_producer_count=peer_count,
            authority_domains=authority,
        )
    )


def _name_suggestions(
    session: Session, run_id: uuid.UUID
) -> tuple[list[CandidateView], dict[str, list[CandidateView]], dict[uuid.UUID, _HintTriage]]:
    """Representative NAME suggestions for the workbench: run-level + per-label,
    triage-ordered, with a per-representative triage map.

    Each (target, normalized value) group renders one representative so rerun
    duplicates beside decided history are never presented as new suggestions.
    Cross-producer facts (voice support, agreement) and each candidate's triage
    priority are computed over all active candidates BEFORE collapsing.
    """
    views = [
        view
        for view in candidates_for_run(session, run_id)
        if view.candidate.field == ClaimField.NAME.value
    ]
    voice_map = _voice_by_label(session, run_id)
    peer_counts = _name_peer_counts(views)

    def _score(view: CandidateView) -> TriageScore:
        label = view.candidate.diarization_label
        peer_key = (label, _name_match_key(view.candidate.value))
        return _triage_for(
            view,
            voice=voice_map.get(label) if label is not None else None,
            peer_count=peer_counts.get(peer_key, 1),
            authority=frozenset(),  # name candidates carry no URL evidence
        )

    scores: dict[uuid.UUID, TriageScore] = {v.candidate.id: _score(v) for v in views}

    def _order(view: CandidateView) -> tuple[int, float, str, str]:
        # Decided history first (a decided value is never re-shown as new), then
        # higher triage PRIORITY — never a raw cross-producer score — then a
        # stable tiebreak. Same key selects representatives and orders the lists.
        return (
            _HINT_STATE_PRECEDENCE[view.state],
            -scores[view.candidate.id].priority,
            view.candidate.value.casefold(),
            str(view.candidate.id),
        )

    groups: dict[tuple[str | None, str], CandidateView] = {}
    proposed_counts: dict[tuple[str | None, str], int] = {}
    for view in views:
        key = (view.candidate.diarization_label, _name_match_key(view.candidate.value))
        if view.state is CandidateState.PROPOSED:
            proposed_counts[key] = proposed_counts.get(key, 0) + 1
        current = groups.get(key)
        if current is None or _order(view) < _order(current):
            groups[key] = view

    triage: dict[uuid.UUID, _HintTriage] = {}
    for key, view in groups.items():
        score = scores[view.candidate.id]
        # Proposed peers hidden behind this representative — never below zero.
        rep_is_proposed = 1 if view.state is CandidateState.PROPOSED else 0
        hidden = max(0, proposed_counts.get(key, 0) - rep_is_proposed)
        triage[view.candidate.id] = _HintTriage(
            priority=score.priority, components=score.components, unresolved_peers=hidden
        )

    run_level = sorted((view for (label, _), view in groups.items() if label is None), key=_order)
    per_label: dict[str, list[CandidateView]] = {}
    for (label, _), view in groups.items():
        if label is not None:
            per_label.setdefault(label, []).append(view)
    for label_views in per_label.values():
        label_views.sort(key=_order)
    return run_level, per_label, triage


