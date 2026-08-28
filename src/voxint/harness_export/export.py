"""Pure DB-reading exporters behind :mod:`voxint.harness_export`.

Every function reads a SQLAlchemy session and returns plain Python
dicts/lists — no filesystem, no clock, no process lookups (``exported_at`` and
``git_sha`` are injected). Serialization and file IO belong to the (deferred)
driver; keeping the core pure keeps it byte-for-byte testable and lets the same
mapping be round-tripped through the real ``voxint score`` parsers in tests.

Determinism: runs are emitted in the caller's order (deduped, validated),
labels/slots/speaker-ids/source-ids are always sorted, and every embedding is
validated (finite, non-zero, exactly ``EMBEDDING_DIM``) so a downstream
``allow_nan=False`` dump can never surprise us.
"""

import enum
import uuid
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint import __version__
from voxint.adjudication.resolver import LabelState, label_states
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MatchCandidate,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.matching import (
    MatchingGates,
    _unit,
    eligible_label_vectors,
    gates_from_settings,
    label_centroid,
    roster_centroids,
)
from voxint.speakers.roster import active_speaker_clause, canonicalize, merge_map

# Ground-truth sentinels the name-accuracy scorer understands (docs/harness.md):
# ``__ABSTAIN__`` = no name should be assigned; ``__NEITHER_DETERMINABLE__`` =
# unscoreable, excluded from the metrics.
ABSTAIN = "__ABSTAIN__"
NEITHER_DETERMINABLE = "__NEITHER_DETERMINABLE__"

# The name-accuracy items contract is versioned by its command, not per record;
# the agreement slots stream likewise. Only the enrollment JSON document carries
# an explicit schema_version (checked by the CLI on load).
ENROLLMENT_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1

KIND_CURATED = "curated"
KIND_NEGATIVE_CONTROL = "negative_control"


class ExportError(Exception):
    """A selection/state problem that must fail the export before any output.

    Raised for a caller mistake (duplicate/empty run ids, a curated run with no
    host, a requested speaker with no usable centroid) or an un-exportable DB
    state (a run whose turns span more than one embedding space). Fail-closed:
    never emit a partial or silently-narrowed artifact, which would bias a
    baseline without anyone noticing.
    """


class TruthAnchoring(enum.StrEnum):
    """How the ground truth relates to the machine proposal (anchoring guard).

    The database cannot prove whether an annotation preceded the proposal, so the
    caller declares it. ``INDEPENDENT`` = truth was fixed without seeing Voxint's
    proposal (a professionally annotated corpus); ``POST_PROPOSAL`` = the operator
    ruled in the console after seeing the proposal (their own material), which can
    anchor a human toward the machine's guess.
    """

    INDEPENDENT = "independent"
    POST_PROPOSAL = "post_proposal"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _ordered_unique_runs(run_ids: Iterable[uuid.UUID]) -> list[uuid.UUID]:
    """Preserve caller order, reject duplicates and an empty selection."""
    seen: set[uuid.UUID] = set()
    ordered: list[uuid.UUID] = []
    for run_id in run_ids:
        if run_id in seen:
            raise ExportError(f"duplicate run id in selection: {run_id}")
        seen.add(run_id)
        ordered.append(run_id)
    if not ordered:
        raise ExportError("no runs selected for export")
    return ordered


def _resolve_names(
    session: Session, speaker_ids: Iterable[uuid.UUID | None]
) -> dict[uuid.UUID, str]:
    """Canonical display names for a set of (possibly merged) speaker ids.

    Match-candidate rows keep the historical ``top_speaker_id`` (an id only, and
    possibly a since-merged one), so an offline consumer cannot recover a name
    from the export alone. Resolve it here, through the merge map, exactly as the
    read-time resolver does, so a rejected near-miss carries a human-readable
    candidate name into its provenance.
    """
    ids = list(speaker_ids)  # may be a generator; we iterate it twice
    tombstones = merge_map(session)
    canonical: set[uuid.UUID] = set()
    for sid in ids:
        if sid is None:
            continue
        c = canonicalize(sid, tombstones)
        if c is not None:
            canonical.add(c)
    if not canonical:
        return {}
    rows = session.execute(
        select(Speaker.id, Speaker.display_name).where(Speaker.id.in_(canonical))
    ).tuples()
    by_canonical = {sid: name for sid, name in rows}
    # Key the result by the ORIGINAL id so callers can look up a raw
    # top_speaker_id directly and still get the canonical name.
    result: dict[uuid.UUID, str] = {}
    for sid in ids:
        if sid is None:
            continue
        c = canonicalize(sid, tombstones)
        if c is not None and c in by_canonical:
            result[sid] = by_canonical[c]
    return result


def _vector_json(vector: np.ndarray, *, dims: int, where: str) -> list[float]:
    """Validate a centroid and render it as a JSON-safe float list.

    Mirrors the harness's own vector guard (finite, non-zero, exact dims) so an
    exported file can never fail the scorer's ``tagged_vector`` check downstream.
    """
    values = [float(x) for x in vector.tolist()]
    if len(values) != dims:
        raise ExportError(f"{where}: embedding has {len(values)} dims, expected {dims}")
    if not all(np.isfinite(values)):
        raise ExportError(f"{where}: embedding has non-finite components")
    if not any(values):
        raise ExportError(f"{where}: embedding is a zero vector")
    return values


def _single_embedding_space(session: Session, run_ids: Sequence[uuid.UUID]) -> str:
    """The one embedding space present across the selected runs' turns.

    The agreement contract is one space per file; a run whose turns span two
    spaces is a pipeline bug. Reject the whole batch rather than silently pick
    one and bias the baseline.
    """
    spaces = {
        space
        for space in session.execute(
            select(DiarizationTurn.embedding_space)
            .where(
                DiarizationTurn.pipeline_run_id.in_(run_ids),
                DiarizationTurn.embedding_space.is_not(None),
            )
            .distinct()
        )
        .scalars()
        .all()
        if space is not None
    }
    if not spaces:
        raise ExportError("selected runs carry no embeddings in any embedding space")
    if len(spaces) > 1:
        raise ExportError(
            f"selected runs span multiple embedding spaces {sorted(spaces)}; "
            "the agreement contract is one space per file"
        )
    return spaces.pop()


def _run_total_speech(session: Session, run_id: uuid.UUID) -> float:
    """Interval-union duration of a run's diarization turns (overlap counted once).

    Feeds the agreement ``total_speech`` field, which gates confident-absence
    calls. Summing per-turn durations would double-count overlapping speech and
    overstate coverage, so merge the intervals first.
    """
    intervals = session.execute(
        select(DiarizationTurn.start_seconds, DiarizationTurn.end_seconds)
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .order_by(DiarizationTurn.start_seconds, DiarizationTurn.end_seconds)
    ).all()
    total = 0.0
    cur_start: float | None = None
    cur_end: float | None = None
    for start, end in intervals:
        if cur_end is None or start > cur_end:
            if cur_start is not None and cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    if cur_start is not None and cur_end is not None:
        total += cur_end - cur_start
    return total


# --------------------------------------------------------------------------- #
# name-accuracy family
# --------------------------------------------------------------------------- #
def _match_provenance(
    mc: MatchCandidate, names: Mapping[uuid.UUID, str]
) -> dict[str, object]:
    """Serialize one match-candidate row as scorer-ignored provenance.

    Carries the full label decision (accepted / rejected / ineligible) plus the
    resolved candidate name so a later confidence-policy pass can re-derive any
    band from a single export. ``margin`` is serialized as ``null`` for a
    single-speaker roster (never ``Infinity`` — the harness rejects non-finite
    numbers, and null keeps the provenance schema stable).
    """
    return {
        "decision": mc.decision,
        "reason": mc.reason,
        "embedding_space": mc.embedding_space,
        "top_speaker_id": str(mc.top_speaker_id) if mc.top_speaker_id else None,
        "top_speaker_name": (
            names.get(mc.top_speaker_id) if mc.top_speaker_id else None
        ),
        "similarity": mc.similarity,
        "margin": mc.margin,
        "vote_agreement": mc.vote_agreement,
        "grounded": mc.grounded,
        "eligible_turns": mc.eligible_turns,
        "eligible_seconds": mc.eligible_seconds,
        "roster_size": mc.roster_size,
    }


def _truth_from_state(state: LabelState) -> str | None:
    """The human ground truth for a label, or None when there is none.

    A human ruling is the only truth source: ASSIGN gives the assigned name,
    EXCLUDE means no name should be assigned (``__ABSTAIN__``), UNKNOWN is
    unscoreable (``__NEITHER_DETERMINABLE__``). Absent a ruling — including a
    label the matcher auto-attributed but no human ever confirmed — there is no
    truth, so the slot is excluded (None) rather than scored against a guess.
    """
    decision = state.effective_decision
    if decision is None:
        return None
    if decision.decision == Decision.AUTO_ENROLL.value:
        return None
    if decision.decision == Decision.ASSIGN.value:
        return state.speaker_name
    if decision.decision == Decision.EXCLUDE.value:
        return ABSTAIN
    return NEITHER_DETERMINABLE  # UNKNOWN (INHERIT is segment-scope, never here)


def name_accuracy_items(
    session: Session,
    run_ids: Iterable[uuid.UUID],
    *,
    truth_anchoring: TruthAnchoring,
) -> list[dict[str, object]]:
    """Render selected runs into ``score name-accuracy`` item records.

    One item per run, one slot per diarization label. ``assigned_name`` is the
    matcher's auto-attribution — the grounded cosine proposal's speaker — read
    from the machine evidence INDEPENDENTLY of the read-time resolution, so a
    label a human later adjudicated still reports what the machine would have
    shown. ``confidence`` accompanies a name only when one is assigned (an
    abstention must not be ranked by leftover proposal confidence). ``truth`` is
    the human ruling. The full match decision rides along under ``match`` for
    later policy work; the scorer ignores every field but the four it parses.
    """
    ordered = _ordered_unique_runs(run_ids)
    items: list[dict[str, object]] = []
    for run_id in ordered:
        states = label_states(session, run_id)
        candidates = {
            mc.diarization_label: mc
            for mc in session.execute(
                select(MatchCandidate).where(
                    MatchCandidate.pipeline_run_id == run_id
                )
            ).scalars()
        }
        names = _resolve_names(
            session, (mc.top_speaker_id for mc in candidates.values())
        )
        slots: dict[str, object] = {}
        for state in states:
            assigned = state.cosine_speaker_name if state.cosine_grounded else None
            slot: dict[str, object] = {
                "assigned_name": assigned,
                "truth": _truth_from_state(state),
                # Confidence is descriptive-only and belongs to an assigned name;
                # leave it off an abstention so risk-coverage never ranks it.
                "confidence": state.cosine_confidence if assigned is not None else None,
                "duration": state.total_seconds,
            }
            mc = candidates.get(state.label)
            if mc is not None:
                slot["match"] = _match_provenance(mc, names)
            slots[state.label] = slot
        items.append(
            {
                "item_id": str(run_id),
                "truth_anchoring": truth_anchoring.value,
                "slots": {label: slots[label] for label in sorted(slots)},
            }
        )
    return items


# --------------------------------------------------------------------------- #
# agreement family
# --------------------------------------------------------------------------- #
def agreement_enrollment(
    session: Session,
    embedding_space: str,
    *,
    roster_speaker_ids: Iterable[uuid.UUID] | None = None,
    dims: int = EMBEDDING_DIM,
) -> dict[str, object]:
    """Render the active roster into a ``score agreement`` enrollment document.

    Each voiceprint is the EXACT centroid production matching compares (reused
    from :func:`voxint.speakers.matching.roster_centroids`, over stored
    enrollment vectors — never recomputed). ``enrollment_items`` counts only the
    rows that contributed a valid (non-zero) vector to that centroid.
    ``source_item_ids`` lists the runs the enrollment came from; ``held_out`` is
    true only when EVERY contributing row records its source run — a missing
    provenance means we cannot attest the voiceprint did not see the item being
    scored, so we fail the attestation (the CLI then abstains on every use).
    """
    centroids = roster_centroids(session, embedding_space)
    rows = session.execute(
        select(
            SpeakerEmbedding.speaker_id,
            SpeakerEmbedding.embedding,
            SpeakerEmbedding.source_pipeline_run_id,
        )
        .join(Speaker, Speaker.id == SpeakerEmbedding.speaker_id)
        .where(SpeakerEmbedding.embedding_space == embedding_space, active_speaker_clause())
    ).all()

    # Per speaker, the provenance of the rows that actually built the centroid.
    contributed: dict[uuid.UUID, list[uuid.UUID | None]] = {}
    for speaker_id, embedding, source_run_id in rows:
        if _unit(np.asarray(embedding, dtype=np.float64)) is None:
            continue  # zero vector — excluded from the centroid, so not counted
        contributed.setdefault(speaker_id, []).append(source_run_id)

    if roster_speaker_ids is not None:
        wanted = list(roster_speaker_ids)
        missing = [sid for sid in wanted if sid not in centroids]
        if missing:
            raise ExportError(
                f"requested roster speakers have no usable centroid in "
                f"{embedding_space!r}: {sorted(str(s) for s in missing)}"
            )
        selected = set(wanted)
    else:
        selected = set(centroids)

    voiceprints: dict[str, object] = {}
    for speaker_id in sorted(selected, key=str):
        centroid = centroids[speaker_id]
        sources = contributed.get(speaker_id, [])
        held_out = bool(sources) and all(src is not None for src in sources)
        source_item_ids = sorted({str(src) for src in sources if src is not None})
        voiceprints[str(speaker_id)] = {
            "embedding": _vector_json(
                centroid, dims=dims, where=f"speaker {speaker_id}"
            ),
            "enrollment_items": len(sources),
            "held_out": held_out,
            "source_item_ids": source_item_ids,
        }
    if not voiceprints:
        raise ExportError(
            f"no active roster voiceprints in embedding space {embedding_space!r}"
        )
    return {
        "schema_version": ENROLLMENT_SCHEMA_VERSION,
        "embedding_space": embedding_space,
        "dims": dims,
        "voiceprints": voiceprints,
    }


def agreement_slots(
    session: Session,
    run_ids: Iterable[uuid.UUID],
    *,
    kind_by_run: Mapping[uuid.UUID, str],
    host_by_run: Mapping[uuid.UUID, uuid.UUID] | None = None,
    gates: MatchingGates,
    embedding_space: str | None = None,
    dims: int = EMBEDDING_DIM,
) -> list[dict[str, object]]:
    """Render selected runs into ``score agreement`` slot records.

    One record per run; one slot per diarization label with a usable centroid,
    built by the matcher's own eligibility + centroid helpers over stored
    per-turn vectors (the exact vector production compared). ``duration`` is the
    label's usable non-overlap speech and ``segments`` its eligible turn count.
    ``kind_by_run`` marks each run curated or negative-control; a curated run
    must name its reference speaker in ``host_by_run``.
    """
    ordered = _ordered_unique_runs(run_ids)
    space = embedding_space or _single_embedding_space(session, ordered)
    host_by_run = host_by_run or {}

    records: list[dict[str, object]] = []
    for run_id in ordered:
        kind = kind_by_run.get(run_id)
        if kind not in (KIND_CURATED, KIND_NEGATIVE_CONTROL):
            raise ExportError(
                f"run {run_id}: kind must be {KIND_CURATED!r} or "
                f"{KIND_NEGATIVE_CONTROL!r}, got {kind!r}"
            )
        by_label = eligible_label_vectors(session, run_id, gates)
        slots: dict[str, object] = {}
        for label in sorted(by_label):
            label_space, entries = by_label[label]
            if label_space != space:
                raise ExportError(
                    f"run {run_id} label {label!r} is in embedding space "
                    f"{label_space!r}, not {space!r}"
                )
            centroid = label_centroid(entries, gates.turn_weight_cap_seconds)
            if centroid is None:
                continue  # degenerate centroid — no usable slot
            slots[label] = {
                "embedding": _vector_json(
                    centroid, dims=dims, where=f"run {run_id} label {label!r}"
                ),
                "duration": sum(usable for _, usable in entries),
                "segments": len(entries),
            }
        if not slots:
            continue  # no eligible speech in this run — nothing to score

        record: dict[str, object] = {
            "item_id": str(run_id),
            "kind": kind,
            "embedding_space": space,
            "total_speech": _run_total_speech(session, run_id),
            "slots": slots,
        }
        if kind == KIND_CURATED:
            host = host_by_run.get(run_id)
            if host is None:
                raise ExportError(f"curated run {run_id} has no host speaker id")
            record["host_id"] = str(host)
        records.append(record)
    if not records:
        raise ExportError("no runs produced any usable agreement slots")
    return records


# --------------------------------------------------------------------------- #
# evidence snapshot
# --------------------------------------------------------------------------- #
def _roster_digest(session: Session, embedding_space: str) -> list[dict[str, str]]:
    """A stable per-speaker centroid fingerprint for the active roster.

    Sorted by speaker id; each centroid hashed over its canonical float list so
    the digest changes iff the roster's matching-relevant vectors change.
    """
    import hashlib
    import json

    centroids = roster_centroids(session, embedding_space)
    digest: list[dict[str, str]] = []
    for speaker_id in sorted(centroids, key=str):
        payload = json.dumps(
            [float(x) for x in centroids[speaker_id].tolist()], sort_keys=True
        )
        digest.append(
            {
                "speaker_id": str(speaker_id),
                "centroid_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    return digest


def evidence_snapshot(
    session: Session,
    settings: Settings,
    run_ids: Iterable[uuid.UUID],
    *,
    exported_at: str,
    git_sha: str | None = None,
    embedding_space: str | None = None,
) -> dict[str, object]:
    """Record the export-time provenance that a baseline must be read against.

    Captures the code version, the matching gates and roster as they are AT
    EXPORT, the embedding-space identity, the selected runs, and an injected
    export time. It is deliberately not a historical replay: the database does
    not retain the gates or roster centroids used when each run was matched, so
    the gates are labelled ``gates_at_export`` and the roster digest is an
    export-time fingerprint. A baseline whose snapshot no longer matches the
    live roster/gates is stale and must be regenerated.
    """
    ordered = _ordered_unique_runs(run_ids)
    space = embedding_space or _single_embedding_space(session, ordered)
    gates = gates_from_settings(settings)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": "match_evidence_snapshot",
        "exported_at": exported_at,
        "code": {"version": __version__, "git_sha": git_sha},
        "embedding": {
            "space": space,
            "dims": EMBEDDING_DIM,
        },
        "gates_at_export": {
            "max_overlap_ratio": gates.max_overlap_ratio,
            "turn_weight_cap_seconds": gates.turn_weight_cap_seconds,
            "min_turns": gates.min_turns,
            "min_seconds": gates.min_seconds,
            "min_cosine": gates.min_cosine,
            "min_margin": gates.min_margin,
            "min_vote_agreement": gates.min_vote_agreement,
            "grounded_min_turns": gates.grounded_min_turns,
            "grounded_min_seconds": gates.grounded_min_seconds,
            "grounded_min_cosine": gates.grounded_min_cosine,
            "grounded_min_margin": gates.grounded_min_margin,
            "grounded_min_vote_agreement": gates.grounded_min_vote_agreement,
        },
        "roster_digest_at_export": _roster_digest(session, space),
        "run_ids": sorted(str(run_id) for run_id in ordered),
    }
