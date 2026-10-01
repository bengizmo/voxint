"""Idempotently seed the bundled guided-tutorial run — the shared fixture builder.

Both ``voxint tutorial seed`` and the test suite call :func:`seed_tutorial_run`.
It creates ONE genuine COMPLETED three-speaker run exhibiting the three states the
tutorial teaches, verified through :func:`voxint.adjudication.resolver.label_states`:

* a **grounded cosine** label — matched to a roster speaker, ``grounded=True``;
* a **heard name** label — an ``llm_hint`` proposal, surfaced but never attribution
  (stays ``UNRESOLVED`` with ``llm_hint_name`` set);
* a fully **unresolved** label — turns but no cosine match and no name.

Design notes:

* SpeakerAssignments are written only through
  :func:`voxint.speakers.matching.replace_run_proposals` (the one validated writer).
  The grounded proposal is authored directly rather than recomputed by
  ``match_speakers`` so the fixture is deterministic and immune to future gate-
  threshold drift; the turn embeddings are still geometrically honest (the grounded
  label sits at cosine 0.95 to the roster centroid, the others orthogonal), so a
  hypothetical re-match would reproduce the same outcome.
* The roster speaker is created directly (no ``AdjudicationDecision``) — a decision
  would resolve the label as ``HUMAN_ASSIGN`` instead of ``GROUNDED_COSINE``.
* The run also carries authored ``match_candidates`` evidence (#675), mirroring
  what ``evaluate_run`` records for this geometry against the one-speaker
  tutorial roster: accepted for the grounded label, ``below_cosine`` rejections
  for the other two. Without it the speaker rail reports "Voice matching did not
  run" beside a voice it lists as matched.
* Idempotency is keyed on ``app_settings.tutorial_run_id`` (FK ``ON DELETE SET
  NULL``): a second seed returns the existing run untouched; a deleted run is
  rebuilt with a fresh id; a missing WAV file (media_root wiped) or missing
  match evidence (a run seeded before #675) is repaired in place. The bundled
  ``MediaItem`` is reused across rebuilds so its UNIQUE ``source_path`` never
  collides. A permanently deleted (purged) tutorial is not repaired: the operator
  asked to start over, so the item is revived, its old run archived, and a fresh
  run built (#676).
* The caller owns the transaction — this module flushes, never commits. The WAV is
  copied to media_root before any DB write and cannot be rolled back with the
  transaction; a rolled-back seed leaves a harmless orphan artifact directory.
* The seeded run is a normal ``COMPLETED`` run with two unresolved labels, so it
  intentionally appears in ``resolver.adjudication_queue`` — that IS the tutorial:
  the user practices claiming and adjudicating it. There is no ``is_tutorial`` flag
  on ``PipelineRun``; the tutorial UX (a later slice) keys off
  ``app_settings.tutorial_run_id`` to frame it, and completing the walkthrough
  resolves the labels so it leaves the queue.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import Resolution, label_states
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    STAGE_ORDER,
    ArtifactKind,
    AssignmentMethod,
    AudioArtifact,
    DiarizationTurn,
    MatchCandidate,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
    StageRun,
    StageStatus,
    TranscriptSegment,
)
from voxint.domain_packs.base import load_default
from voxint.ingest.service import archive_run
from voxint.media.operations import has_active_operation
from voxint.pipeline.stages.context import normalized_audio_path
from voxint.speakers.matching import (
    DECISION_ACCEPTED,
    DECISION_REJECTED,
    REASON_ACCEPTED,
    REASON_BELOW_COSINE,
    CosineProposal,
    LabelDecision,
    NameHintProposal,
    replace_run_match_candidates,
    replace_run_proposals,
)
from voxint.speakers.roster import canonicalize, merge_map
from voxint.tutorial import resources

logger = logging.getLogger(__name__)

# Stable identity of the bundled sample under MEDIA_ROOT — a reserved sentinel,
# reused across rebuilds so its UNIQUE source_path never collides after a run
# deletion. The seeded run is already COMPLETED, so ACQUIRE/PREPARE never read it,
# but the Media library does (#646): it flags an item whose live file is absent,
# and trash/restore/archive move that file. So the bundled WAV is written here as
# well as to the normalized artifact below (see _ensure_source_present).
TUTORIAL_SOURCE_PATH = "tutorial/sample-3speaker.wav"
# Mirrors pipeline.stages.prepare._ARTIFACT_TEMPLATE so the tutorial run's artifact
# looks identical to a real prepared run; only this row and the physical copy must
# agree, since normalized_audio_path serves whatever path is stored.
_ARTIFACT_TEMPLATE = "artifacts/{run_id}/normalized.wav"
# Cosine of the grounded label's turns to the roster centroid: >= grounded gate 0.70.
_GROUNDED_SIMILARITY = 0.95
# Nominal terminal revision — a completed run has advanced well past 0; the tutorial
# run never re-runs the pipeline, and review-claim CAS only needs a consistent start.
_TERMINAL_REVISION = 2 * len(STAGE_ORDER)
# Stable, arbitrary 64-bit key for the transaction advisory lock that serializes
# concurrent seeds (fits signed bigint). Every seed entry point — the CLI and both
# web routes (#75) — funnels through seed_tutorial_run, so this one lock closes the
# CLI-vs-web race too: without it two concurrent seeds each build a separate
# PipelineRun (the loser is a completed run with unresolved labels that pollutes the
# review queue). The lock is held for the caller's transaction and released on
# commit/rollback; a second waiter, once admitted, reads the winner's committed
# tutorial_run_id and returns it idempotently.
_SEED_ADVISORY_LOCK_KEY = 0x766F78696E747574


class TutorialSeedError(RuntimeError):
    """The seeded run does not exhibit the expected three states — a builder bug."""


def _axis(index: int) -> list[float]:
    """A 192-dim unit basis vector e_index."""
    vec = [0.0] * EMBEDDING_DIM
    vec[index] = 1.0
    return vec


# Roster centroid = e0. Grounded label's turns sit at cosine 0.95 to it (unit).
_ROSTER_EMBEDDING = _axis(0)
_GROUNDED_TURN_EMBEDDING = [
    _GROUNDED_SIMILARITY,
    math.sqrt(1.0 - _GROUNDED_SIMILARITY**2),
    *([0.0] * (EMBEDDING_DIM - 2)),
]


def _label_embeddings(layout: dict[str, Any]) -> dict[str, list[float]]:
    """Per-label turn embedding: grounded label near the roster, others orthogonal."""
    roster_label = layout["roster_speaker"]["label"]
    labels = sorted({utt["label"] for utt in layout["utterances"]})
    embeddings: dict[str, list[float]] = {}
    axis = 1  # e1, e2, ... for the non-grounded labels (e0 is the roster centroid)
    for label in labels:
        if label == roster_label:
            embeddings[label] = _GROUNDED_TURN_EMBEDDING
        else:
            embeddings[label] = _axis(axis)
            axis += 1
    return embeddings


def _atomic_copy(wav_bytes: bytes, dest: Path) -> None:
    """Write ``wav_bytes`` to ``dest`` atomically (temp in the same dir + rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(wav_bytes)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _revive_purged_media(session: Session, media: MediaItem) -> None:
    """Bring a permanently deleted tutorial item back for a fresh run (#676).

    Purge keeps the ``MediaItem`` (its UNIQUE ``source_path`` is the tutorial's
    reserved identity) and the old run's rows, but deletes every artifact and
    file. Reached only from an explicit set-up (CLI or Settings), so reviving is
    the operator's choice, not a silent undo of "Delete permanently". Every old
    run on the item is soft-archived so it leaves the review queue and /runs;
    the caller then builds a fresh run and rewrites the original file.

    The columns are rewritten directly, not through a media operation, so the
    item's completed PURGE row stays in ``media_operations`` as history of the
    earlier deletion. Nothing reads a completed operation as current state, and
    a later trash or purge of the revived item plans its own operations.
    """
    # Lock AND repopulate: a plain FOR UPDATE would return the instance already
    # in the identity map with its pre-lock values (see reconcile._process_one).
    session.refresh(media, with_for_update=True)
    if media.purged_at is None:
        return
    if has_active_operation(session, media.id):
        raise TutorialSeedError("tutorial media item has a media operation in progress")
    old_runs = session.execute(
        select(PipelineRun.id).where(
            PipelineRun.media_item_id == media.id,
            PipelineRun.archived_at.is_(None),
        )
    ).scalars()
    for run_id in old_runs:
        archive_run(session, run_id)
    media.purged_at = None
    media.trashed_at = None
    media.current_path = media.source_path
    session.flush()


def _get_or_create_media(
    session: Session, provenance: dict[str, Any], size_bytes: int
) -> MediaItem:
    """The reserved tutorial MediaItem (UNIQUE source_path), race-safe on insert.

    Mirrors ``ingest.service._get_or_create_media`` / ``app_settings.get_or_create``:
    two concurrent seeds can both observe no row and both insert; the SAVEPOINT
    rolls back only the loser's insert — not the caller's transaction — so we
    re-read and adopt the winner's row instead of crashing on UNIQUE. A purged
    row is revived (see :func:`_revive_purged_media`).
    """
    media = session.execute(
        select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
    ).scalar_one_or_none()
    if media is not None:
        if media.purged_at is not None:
            _revive_purged_media(session, media)
        return media
    media = MediaItem(
        source_path=TUTORIAL_SOURCE_PATH,
        media_type="audio/wav",
        duration_seconds=float(provenance["duration_seconds"]),
        size_bytes=size_bytes,
        sha256=str(provenance["wav_sha256"]),
    )
    try:
        with session.begin_nested():
            session.add(media)
            session.flush()
    except IntegrityError:
        adopted = session.execute(
            select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one_or_none()
        if adopted is None:
            raise  # not the expected UNIQUE(source_path) race — surface it
        return adopted
    return media


def _get_or_create_roster_speaker(
    session: Session, name: str, space: str, embedding: list[float]
) -> Speaker:
    """The grounded roster identity + its enrollment centroid (created once).

    ``name`` is a reserved tutorial display name (see ``utterance.json``), scoped
    like the other tutorial anchors (``tutorial/…`` source_path, ``voxint-tutorial``
    embedding space) so adopting a same-named row only ever re-adopts the tutorial's
    own speaker — never a real user's roster identity. The insert is SAVEPOINT-
    guarded for the same concurrent-seed race (``display_name`` is UNIQUE).
    """
    speaker = session.execute(
        select(Speaker).where(Speaker.display_name == name)
    ).scalar_one_or_none()
    if speaker is None:
        speaker = Speaker(display_name=name)
        try:
            with session.begin_nested():
                session.add(speaker)
                session.flush()
        except IntegrityError:
            speaker = session.execute(
                select(Speaker).where(Speaker.display_name == name)
            ).scalar_one_or_none()
            if speaker is None:
                raise  # not the expected UNIQUE(display_name) race — surface it
    # Roster curation (issue #7) may have merged or archived a previously seeded
    # tutorial speaker; the tutorial needs its grounded anchor active again. A
    # merge is followed to the canonical identity (the tutorial centroid moved
    # with it), but auto-restoring is reserved for the tutorial's own identity —
    # un-archiving a REAL speaker the operator deliberately archived, just
    # because the tutorial anchor was merged into it, would silently undo their
    # curation. In that case the seed proceeds without touching deleted_at and
    # says so; the operator restores by hand if they want the tutorial grounded.
    if speaker.merged_into_id is not None:
        canonical = session.get(Speaker, canonicalize(speaker.id, merge_map(session)))
        if canonical is None:  # FK makes this unreachable; fail loud, not silent
            raise TutorialSeedError("tutorial speaker merge target missing")
        speaker = canonical
    if speaker.deleted_at is not None:
        if speaker.display_name == name:
            speaker.deleted_at = None
        else:
            logger.warning(
                "tutorial anchor was merged into archived speaker %r;"
                " leaving it archived — restore it via /speakers to re-ground"
                " the tutorial",
                speaker.display_name,
            )
    has_embedding = session.execute(
        select(SpeakerEmbedding.id).where(
            SpeakerEmbedding.speaker_id == speaker.id,
            SpeakerEmbedding.embedding_space == space,
        )
    ).first()
    if has_embedding is None:
        session.add(
            SpeakerEmbedding(
                speaker_id=speaker.id,
                embedding_space=space,
                embedding=embedding,
                source_pipeline_run_id=None,
                source_diarization_label=None,
                source_adjudication_decision_id=None,
            )
        )
        session.flush()
    return speaker


def _grounded_proposal(label: str, speaker_id: uuid.UUID) -> CosineProposal:
    return CosineProposal(
        diarization_label=label,
        speaker_id=speaker_id,
        similarity=_GROUNDED_SIMILARITY,
        margin=math.inf,  # single-speaker roster: no runner-up
        vote_agreement=1.0,
        grounded=True,
    )


def _match_decisions(
    layout: dict[str, Any], speaker_id: uuid.UUID
) -> tuple[LabelDecision, ...]:
    """The evidence ``evaluate_run`` records for the tutorial geometry (#675).

    One-speaker roster, so margin is stored as None and every turn votes for the
    only candidate (vote agreement 1.0). The grounded label sits at cosine 0.95;
    the other two are orthogonal (cosine 0.0), so they fail the cosine gate.
    Authored rather than recomputed for the same reason as the proposals: an
    operator's earlier tutorial rulings grow the tutorial roster, and a live
    re-match would then disagree with the authored proposals.
    """
    roster_label = layout["roster_speaker"]["label"]
    space = layout["embedding_space"]
    durations: dict[str, list[float]] = {}
    for utt in layout["utterances"]:
        durations.setdefault(utt["label"], []).append(
            float(utt["end"]) - float(utt["start"])
        )
    decisions: list[LabelDecision] = []
    for label in sorted(durations):
        accepted = label == roster_label
        decisions.append(
            LabelDecision(
                diarization_label=label,
                decision=DECISION_ACCEPTED if accepted else DECISION_REJECTED,
                reason=REASON_ACCEPTED if accepted else REASON_BELOW_COSINE,
                embedding_space=space,
                top_speaker_id=speaker_id,
                similarity=_GROUNDED_SIMILARITY if accepted else 0.0,
                margin=None,
                vote_agreement=1.0,
                grounded=True if accepted else None,
                eligible_turns=len(durations[label]),
                eligible_seconds=sum(durations[label]),
                roster_size=1,
                proposal=_grounded_proposal(label, speaker_id) if accepted else None,
            )
        )
    return tuple(decisions)


def _backfill_match_candidates(
    session: Session, run_id: uuid.UUID, layout: dict[str, Any]
) -> None:
    """Add the #675 match evidence to a tutorial run seeded without it."""
    has_rows = session.execute(
        select(MatchCandidate.id).where(MatchCandidate.pipeline_run_id == run_id)
    ).first()
    if has_rows is not None:
        return
    speaker_id = session.execute(
        select(SpeakerAssignment.speaker_id).where(
            SpeakerAssignment.pipeline_run_id == run_id,
            SpeakerAssignment.diarization_label == layout["roster_speaker"]["label"],
            SpeakerAssignment.method == AssignmentMethod.COSINE.value,
        )
    ).scalar_one_or_none()
    if speaker_id is None:
        raise TutorialSeedError(
            f"tutorial run {run_id} has no grounded cosine assignment to backfill from"
        )
    replace_run_match_candidates(session, run_id, _match_decisions(layout, speaker_id))
    session.flush()


def _build_run(
    session: Session,
    *,
    media: MediaItem,
    layout: dict[str, Any],
    media_root: Path,
    wav_bytes: bytes,
    provenance: dict[str, Any],
) -> uuid.UUID:
    run_id = uuid.uuid4()
    relative = _ARTIFACT_TEMPLATE.format(run_id=run_id)
    # Copy the audio BEFORE the DB rows: a rolled-back transaction then leaves only
    # an orphan artifact dir, never a run row pointing at a missing file.
    _atomic_copy(wav_bytes, media_root / relative)

    session.add(
        PipelineRun(
            id=run_id,
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            current_stage=None,
            revision=_TERMINAL_REVISION,
            error=None,
            # The tutorial is a fixed teaching artifact — always the bundled
            # generic pack, frozen like any other run (issue #11).
            domain_pack=load_default().to_mapping(),
        )
    )
    session.flush()  # parent row must exist before its FK-bearing children insert
    finished_at = datetime.now(tz=UTC)
    for stage in STAGE_ORDER:
        session.add(
            StageRun(
                pipeline_run_id=run_id,
                stage=stage.value,
                status=StageStatus.COMPLETED.value,
                attempt=1,
                finished_at=finished_at,
            )
        )
    session.add(
        AudioArtifact(
            pipeline_run_id=run_id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=relative,
            meta={
                "sample_rate": provenance["sample_rate"],
                "channels": provenance["channels"],
                "codec": provenance["encoding"],
                "duration_seconds": provenance["duration_seconds"],
            },
        )
    )

    embeddings = _label_embeddings(layout)
    space = layout["embedding_space"]
    for utt in sorted(layout["utterances"], key=lambda u: u["index"]):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run_id,
                turn_index=utt["index"],
                start_seconds=float(utt["start"]),
                end_seconds=float(utt["end"]),
                label=utt["label"],
                overlap=False,
                overlap_seconds=0.0,
                snr_db=None,
                skip_reason=None,
                embedding=embeddings[utt["label"]],
                embedding_space=space,
            )
        )
        session.add(
            TranscriptSegment(
                pipeline_run_id=run_id,
                segment_index=utt["index"],
                start_seconds=float(utt["start"]),
                end_seconds=float(utt["end"]),
                raw_text=utt["text"],
                enhanced_text=None,
                diarization_label=utt["label"],
                suspect=False,
            )
        )
    session.flush()  # turns must be visible before replace_run_proposals validates labels

    roster = layout["roster_speaker"]
    speaker = _get_or_create_roster_speaker(
        session, roster["display_name"], space, _ROSTER_EMBEDDING
    )
    heard = layout["heard_name"]
    replace_run_proposals(
        session,
        run_id,
        (_grounded_proposal(roster["label"], speaker.id),),
        (
            NameHintProposal(
                diarization_label=heard["label"],
                proposed_name=heard["name"],
            ),
        ),
    )
    replace_run_match_candidates(session, run_id, _match_decisions(layout, speaker.id))
    session.flush()
    return run_id


def _verify_states(session: Session, run_id: uuid.UUID, layout: dict[str, Any]) -> None:
    """Assert the seeded run resolves to the three tutorial states (fail loudly)."""
    states = {s.label: s for s in label_states(session, run_id)}
    roster_label = layout["roster_speaker"]["label"]
    heard = layout["heard_name"]
    unresolved_label = layout["unresolved_label"]

    grounded = states.get(roster_label)
    if grounded is None or grounded.resolution is not Resolution.GROUNDED_COSINE:
        raise TutorialSeedError(
            f"label {roster_label!r} is not GROUNDED_COSINE: "
            f"{grounded.resolution if grounded else 'missing'}"
        )
    hint = states.get(heard["label"])
    if (
        hint is None
        or hint.resolution is not Resolution.UNRESOLVED
        or hint.llm_hint_name != heard["name"]
    ):
        raise TutorialSeedError(
            f"label {heard['label']!r} is not an unresolved heard-name for "
            f"{heard['name']!r}"
        )
    plain = states.get(unresolved_label)
    if (
        plain is None
        or plain.resolution is not Resolution.UNRESOLVED
        or plain.llm_hint_name is not None
    ):
        raise TutorialSeedError(f"label {unresolved_label!r} is not purely unresolved")
    # #675: the speaker rail reads "matching did not run" unless every label
    # carries match evidence that agrees with its resolution.
    expected = {
        label: DECISION_ACCEPTED if label == roster_label else DECISION_REJECTED
        for label in states
    }
    recorded = {label: state.match_decision for label, state in states.items()}
    if recorded != expected:
        raise TutorialSeedError(
            f"match evidence {recorded!r} does not match the tutorial states {expected!r}"
        )


def _ensure_wav_present(session: Session, run_id: uuid.UUID, media_root: Path) -> None:
    """Repair the served WAV if media_root was wiped since the run was seeded."""
    path = normalized_audio_path(session, run_id, media_root)
    if not path.is_file():
        _atomic_copy(resources.load_sample_wav_bytes(), path)


def _ensure_source_present(media: MediaItem, media_root: Path) -> None:
    """Write the tutorial item's original where the Media library looks for it.

    Follows the live pointer (``current_path``, else ``source_path``), so a
    trashed or archived tutorial is repaired in place rather than resurrected at
    its old path. A purged item is never written here: only
    :func:`_revive_purged_media` brings one back, as an explicit decision.
    """
    if media.purged_at is not None:
        return
    live = media.current_path if media.current_path is not None else media.source_path
    root = media_root.resolve()
    dest = (media_root / live).resolve()
    if not dest.is_relative_to(root):
        raise TutorialSeedError(f"tutorial media path escapes MEDIA_ROOT: {live!r}")
    if not dest.is_file():
        _atomic_copy(resources.load_sample_wav_bytes(), dest)


def seed_tutorial_run(
    session: Session, *, media_root: Path, settings: Settings
) -> uuid.UUID:
    """Seed (or return the existing) bundled tutorial run. Caller commits.

    Returns the run id referenced by ``app_settings.tutorial_run_id``.
    """
    # Serialize concurrent seeds so the idempotency read below is a genuine
    # post-lock re-check (see _SEED_ADVISORY_LOCK_KEY). No-op on any non-Postgres
    # harness — the production/test app is Postgres-only by doctrine (pgvector).
    # Correctness of the re-check assumes READ COMMITTED (the Postgres default):
    # the admitted waiter takes a fresh snapshot and sees the winner's committed
    # tutorial_run_id. Under REPEATABLE READ the loser would instead attempt a
    # rebuild — we do not run that isolation level.
    if session.get_bind().dialect.name == "postgresql":
        session.execute(select(func.pg_advisory_xact_lock(_SEED_ADVISORY_LOCK_KEY)))

    layout = resources.load_layout()
    provenance = resources.load_provenance()

    existing = get_app_settings(session)
    if existing is not None and existing.tutorial_run_id is not None:
        run = session.get(PipelineRun, existing.tutorial_run_id)
        # A purged run has lost its audio for good; fall through to a fresh
        # build, which revives the item and archives this run (#676).
        if run is not None and run.media_item.purged_at is None:
            _ensure_wav_present(session, run.id, media_root)
            _ensure_source_present(run.media_item, media_root)
            _backfill_match_candidates(session, run.id, layout)
            return run.id

    wav_bytes = resources.load_sample_wav_bytes()
    media = _get_or_create_media(session, provenance, len(wav_bytes))
    run_id = _build_run(
        session,
        media=media,
        layout=layout,
        media_root=media_root,
        wav_bytes=wav_bytes,
        provenance=provenance,
    )
    _verify_states(session, run_id, layout)

    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.tutorial_run_id = run_id
    session.flush()
    # Last, after every step that can fail: a rolled-back seed must not leave the
    # original behind with no MediaItem row, where a folder scan would ingest it.
    _ensure_source_present(media, media_root)
    return run_id
