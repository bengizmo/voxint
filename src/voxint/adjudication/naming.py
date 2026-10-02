"""Name a voice from the command line through the console's own ledger (#741).

One transaction per call. The run row is locked and every live review claim
is refused, the caller's own included, because ``claim_run`` rotates the
token on a same-reviewer re-claim and would log a console tab out of its
review. No claim is written here, so a crash leaves nothing behind.

Branches, in order: the name belongs to an active roster speaker (assign it;
a no-op when the label already resolves to that speaker); the name belongs to
a merged or archived speaker (refuse with the roster's own wording); the
label's speaker is an auto-enrolled placeholder, a ``Voice N`` created by an
``AUTO_ENROLL`` ruling that nobody renamed (rename it and add a human ASSIGN);
otherwise enroll a new speaker from the voice's audio. A fresh nonce per call,
so A then B then A records three rulings. The caller owns commit and rollback.
"""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, aliased

from voxint.activity import record_speaker_identified
from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import LabelState, label_states, newest_in_scope
from voxint.adjudication.slots import ClaimUnavailableError, require_unclaimed
from voxint.adjudication.transcript import label_display_name
from voxint.db.models import AdjudicationDecision, Decision, PipelineRun, Speaker, SpeakerAssignment
from voxint.speakers.auto_enroll import NAME_PREFIX
from voxint.speakers.matching import MatchingGates
from voxint.speakers.roster import (
    RosterError,
    describe_name_owner,
    is_active,
    normalize_display_name,
    rename_speaker,
)


class NamingError(ValueError):
    """An operator-visible naming refusal."""


@dataclass(frozen=True)
class VoiceRow:
    label: str
    name: str
    resolution: str
    talk_time: str


@dataclass(frozen=True)
class NamingResult:
    effects: tuple[str, ...]


def list_voices(session: Session, run_id: uuid.UUID) -> list[VoiceRow]:
    if session.get(PipelineRun, run_id) is None:
        raise NamingError(f"no run {run_id}")
    rows = []
    for state in sorted(label_states(session, run_id), key=lambda s: s.label):
        minutes, seconds = divmod(int(state.total_seconds), 60)
        rows.append(
            VoiceRow(
                state.label,
                label_display_name(state, state.label),
                state.resolution.value,
                f"{minutes}:{seconds:02}",
            )
        )
    return rows


def _resolve(states: list[LabelState], voice: str) -> LabelState:
    for state in states:
        if state.label == voice:
            return state
    matches = [s for s in states if label_display_name(s, s.label) == voice]
    if not matches:
        matches = [
            s for s in states if label_display_name(s, s.label).casefold() == voice.casefold()
        ]
    if len(matches) == 1:
        return matches[0]
    voices = ", ".join(f'{s.label} "{label_display_name(s, s.label)}"' for s in states)
    reason = "ambiguous" if matches else "unknown"
    raise NamingError(f"{reason} voice {voice!r}; voices: {voices}")


def _other_runs(session: Session, run_id: uuid.UUID, speaker_id: uuid.UUID) -> int:
    ruling = aliased(AdjudicationDecision)
    revoke = aliased(AdjudicationDecision)
    live = select(ruling.pipeline_run_id).where(
        ruling.speaker_id == speaker_id,
        ruling.pipeline_run_id != run_id,
        ruling.detached_at.is_(None),
        or_(ruling.transcript_segment_id.is_(None), newest_in_scope(ruling)),
        ~select(revoke.id).where(revoke.voids_decision_id == ruling.id).exists(),
    )
    machine = select(SpeakerAssignment.pipeline_run_id).where(
        SpeakerAssignment.speaker_id == speaker_id,
        SpeakerAssignment.pipeline_run_id != run_id,
    )
    return len(session.scalars(live.union(machine)).all())


def name_voice(
    session: Session,
    run_id: uuid.UUID,
    voice: str,
    name: str,
    *,
    operator: str,
    gates: MatchingGates,
    activity_enabled: bool,
) -> NamingResult:
    """Name one voice under the run lock without taking a review claim."""
    try:
        require_unclaimed(session, run_id)
        state = _resolve(label_states(session, run_id), voice)
        try:
            name = normalize_display_name(name)
        except ValueError as exc:
            raise NamingError(str(exc)) from exc
        if operator == "system:auto_enroll":
            raise NamingError("configure VOXINT_USER as a human operator identity")
        nonce = str(uuid.uuid4())
        owner = session.scalar(
            select(Speaker)
            .where(Speaker.display_name == name)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        effects = []
        speaker: Speaker | None
        decision_id: uuid.UUID | None = None
        if owner is not None:
            if not is_active(owner):
                raise NamingError(describe_name_owner(owner))
            if state.speaker_id == owner.id:
                return NamingResult((f'{state.label} already assigned to "{name}"',))
            speaker = owner
        else:
            # Re-read under the roster lock before deciding whether this is still
            # a generated name. Concurrent roster renames must not be overwritten.
            speaker = session.scalar(
                select(Speaker)
                .where(Speaker.id == state.speaker_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            placeholder = (
                state.effective_decision is not None
                and state.effective_decision.decision == Decision.AUTO_ENROLL.value
                and speaker is not None
                and re.fullmatch(re.escape(NAME_PREFIX) + r"[0-9]+", speaker.display_name)
            )
            if placeholder and speaker is not None:
                old_name = speaker.display_name
                count = _other_runs(session, run_id, speaker.id)
                speaker = rename_speaker(session, speaker.id, name)
                noun = "recording" if count == 1 else "recordings"
                effects.append(
                    f'renamed "{old_name}" -> "{name}" (also appears in {count} other {noun})'
                )
            else:
                enrolled = enroll_new_speaker(
                    session,
                    run_id=run_id,
                    diarization_label=state.label,
                    display_name=name,
                    operator=operator,
                    idempotency_key=nonce,
                    gates=gates,
                )
                speaker = session.get(Speaker, enrolled.speaker_id)
                if speaker is None:
                    raise RuntimeError(f"enrolled speaker {enrolled.speaker_id} not found")
                decision_id = enrolled.decision_id
                effects.append(f'enrolled "{name}" as a new speaker')
        if speaker is None:
            raise RuntimeError("naming resolved no speaker")
        if decision_id is None:
            row = record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label=state.label,
                decision=Decision.ASSIGN,
                speaker_id=speaker.id,
                operator=operator,
                idempotency_key=nonce,
            )
            decision_id = row.id
        effects.append(f'assigned {state.label} -> "{name}"')
        if activity_enabled and speaker.id != state.speaker_id:
            record_speaker_identified(
                session,
                run_id=run_id,
                decision_id=decision_id,
                speaker_name=speaker.display_name,
                speaker_id=speaker.id,
            )
        return NamingResult(tuple(effects))
    except (ClaimUnavailableError, EnrollmentError, RosterError, NamingError) as exc:
        raise NamingError(str(exc).replace("\u2014", ";")) from exc
