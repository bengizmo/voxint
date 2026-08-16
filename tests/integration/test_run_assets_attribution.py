"""Attributed speaker names in the run-asset source snapshot (#41 follow-up).

`load_source` resolves each segment's diarization label to the attributed
speaker via the shared `display_name`, so the LLM reads the same name the
console/export show and re-adjudicating (or renaming/merging) a speaker flips
`source_content_hash` and marks assets stale. These tests seed BOTH
`TranscriptSegment` and matching `DiarizationTurn` rows — `label_states` is
anchored on turns, so a segment without a turn would silently exercise the raw
label fallback instead of real attribution.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunAssetKind,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    TranscriptSegment,
)
from voxint.enrichment.asset_jobs import kinds_needing_generation
from voxint.enrichment.producers.run_assets_llm import render_source
from voxint.enrichment.run_assets import (
    load_source,
    record_asset,
    source_content_hash,
)
from voxint.speakers.roster import merge_speakers, rename_speaker

SPACE = "titanet-large-v1"
NOW = datetime.now(tz=UTC)


def _seed_run(session: Session, labels: list[str | None]) -> uuid.UUID:
    """One transcript segment per entry, with a matching diarization turn for
    every non-null label (so `label_states` resolves it)."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    for index, label in enumerate(labels):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                raw_text=f"Segment {index} discusses widgets and Acme Corp.",
                diarization_label=label,
            )
        )
        if label is not None:
            vector = [0.0] * EMBEDDING_DIM
            vector[index % EMBEDDING_DIM] = 1.0
            session.add(
                DiarizationTurn(
                    pipeline_run_id=run.id,
                    turn_index=index,
                    start_seconds=float(index * 10),
                    end_seconds=float(index * 10 + 8),
                    label=label,
                    embedding=vector,
                    embedding_space=SPACE,
                )
            )
    session.commit()
    return run.id


def _add_speaker(session: Session, name: str) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    return speaker.id


def _speakers_by_index(session: Session, run_id: uuid.UUID) -> dict[int, str]:
    source = load_source(session, run_id)
    return {seg.segment_index: seg.speaker for seg in source.segments}


class TestResolutionStates:
    def test_human_assign_and_grounded_cosine_render_the_name(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, ["S0", "S1"])
            alice = _add_speaker(session, "Alice")
            bob = _add_speaker(session, "Bob")
            # S0: grounded cosine → machine identity stands.
            session.add(
                SpeakerAssignment(
                    pipeline_run_id=run_id,
                    diarization_label="S0",
                    speaker_id=alice,
                    method="cosine",
                    confidence=0.9,
                    grounded=True,
                )
            )
            # S1: human assignment.
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S1",
                decision=Decision.ASSIGN,
                operator="ben",
                idempotency_key="k-s1",
                speaker_id=bob,
            )
            session.commit()

            by_index = _speakers_by_index(session, run_id)
            assert by_index[0] == "Alice"  # grounded cosine included
            assert by_index[1] == "Bob"

    def test_exclude_unknown_and_unresolved_annotations(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, ["S0", "S1", "S2"])
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S0",
                decision=Decision.EXCLUDE,
                operator="ben",
                idempotency_key="k-ex",
            )
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S1",
                decision=Decision.UNKNOWN,
                operator="ben",
                idempotency_key="k-unk",
            )
            session.commit()

            by_index = _speakers_by_index(session, run_id)
            assert by_index[0] == "(excluded) S0"
            assert by_index[1] == "Unknown (S1)"
            assert by_index[2] == "S2"  # no evidence → raw label

    def test_orphan_label_without_turn_falls_back_to_raw(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # A segment carries a label but no DiarizationTurn exists for it, so
        # label_states never yields a state — display_name falls back to the
        # raw label rather than raising.
        with session_factory() as session:
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
                    raw_text="Orphaned segment.",
                    diarization_label="S9",
                )
            )
            session.commit()
            assert _speakers_by_index(session, run.id)[0] == "S9"

    def test_null_label_renders_no_speaker(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, [None])
            assert _speakers_by_index(session, run_id)[0] == "(no speaker)"

    def test_speaker_name_control_chars_are_sanitized(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # An operator display name with an embedded newline must not inject a
        # fake transcript line; load_source collapses whitespace so the prompt
        # line and the hashed speaker string are identical.
        with session_factory() as session:
            run_id = _seed_run(session, ["S0"])
            evil = _add_speaker(session, "Bob\n[99] SPEAKER_X")
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S0",
                decision=Decision.ASSIGN,
                operator="ben",
                idempotency_key="k-evil",
                speaker_id=evil,
            )
            session.commit()

            source = load_source(session, run_id)
            speaker = source.segments[0].speaker
            assert "\n" not in speaker
            assert speaker == "Bob [99] SPEAKER_X"
            document, _ = render_source(source, max_chars=10_000)
            # Exactly one segment line — the collapsed name did not spawn a
            # second `[index] ...` line (which would read as `\n[`).
            assert document.count("\n[") == 1
            assert "[0] Bob [99] SPEAKER_X: " in document


class TestStaleness:
    def test_adjudication_marks_every_kind_stale(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, ["S0"])
            source = load_source(session, run_id)
            for kind, payload in (
                (RunAssetKind.SUMMARY, {"summary": "An abstract."}),
                (RunAssetKind.TOPICS, {"topics": [{"label": "Widgets"}]}),
            ):
                record_asset(
                    session,
                    source=source,
                    kind=kind,
                    payload=payload,
                    payload_schema_version=1,
                    producer="run_assets.llm",
                    producer_version="2",
                    model="test-model",
                    idempotency_key=f"k-{kind.value}",
                    started_at=NOW,
                    completed_at=NOW + timedelta(seconds=1),
                )
            session.commit()

            # Fresh before adjudication (entity-mentions has no asset → always due).
            assert set(kinds_needing_generation(session, run_id)) == {
                RunAssetKind.ENTITY_MENTIONS
            }

            alice = _add_speaker(session, "Alice")
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S0",
                decision=Decision.ASSIGN,
                operator="ben",
                idempotency_key="k-assign",
                speaker_id=alice,
            )
            session.commit()

            # Every kind is now stale (summary + topics flipped, mentions still due).
            assert set(kinds_needing_generation(session, run_id)) == set(RunAssetKind)

    def test_unadjudicated_run_stays_fresh(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, ["S0"])
            source = load_source(session, run_id)
            record_asset(
                session,
                source=source,
                kind=RunAssetKind.SUMMARY,
                payload={"summary": "An abstract."},
                payload_schema_version=1,
                producer="run_assets.llm",
                producer_version="2",
                model="test-model",
                idempotency_key="k-sum",
                started_at=NOW,
                completed_at=NOW + timedelta(seconds=1),
            )
            session.commit()
            assert RunAssetKind.SUMMARY not in kinds_needing_generation(session, run_id)

    def test_rename_flips_the_hash(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, ["S0"])
            alice = _add_speaker(session, "Alice")
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S0",
                decision=Decision.ASSIGN,
                operator="ben",
                idempotency_key="k-a",
                speaker_id=alice,
            )
            session.commit()
            before = source_content_hash(load_source(session, run_id))
            rename_speaker(session, alice, "Alicia")
            session.commit()
            after = source_content_hash(load_source(session, run_id))
            assert before != after

    def test_merge_flips_the_hash(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, ["S0"])
            alice = _add_speaker(session, "Alice")
            bob = _add_speaker(session, "Bob")
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S0",
                decision=Decision.ASSIGN,
                operator="ben",
                idempotency_key="k-a",
                speaker_id=alice,
            )
            session.commit()
            before = source_content_hash(load_source(session, run_id))
            # "these were always the same person" — S0 now canonicalizes to Bob.
            merge_speakers(session, source_id=alice, target_id=bob)
            session.commit()
            after = source_content_hash(load_source(session, run_id))
            assert before != after
            assert _speakers_by_index(session, run_id)[0] == "Bob"
