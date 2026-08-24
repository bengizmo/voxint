"""Draft-triage ordering over the real app read paths (issue #42).

Seeds enrichment candidates through the sanctioned draft writer plus a cosine
``speaker_assignments`` row, then drives the two read surfaces
(``_name_suggestions`` for the workbench, ``_research_state`` for research) and
asserts triage priority ordering and the explainable component values — voice
value-matching, cross-producer agreement (and its exclusion of rejected
claims), independent domains, and source authority.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.routers.speakers import _research_state
from voxint.api.triage_view import _name_suggestions
from voxint.config import Settings
from voxint.db.models import (
    ClaimField,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    ProfileDecision,
    ProfileReviewDecision,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)
from voxint.enrichment.drafts import (
    CandidateDraft,
    EnrichmentScope,
    MetadataEvidence,
    UrlEvidence,
    record_producer_run,
)

NOW = datetime.now(tz=UTC)


def _run_with_metadata(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    meta = MediaSourceMetadata(
        media_item_id=media.id,
        source_kind="ytdlp",
        title="Interview with Jane Doe",
        raw_schema_version=1,
        acquired_at=NOW,
    )
    session.add(meta)
    session.flush()
    return run.id, meta.id


def _name_run(
    session: Session,
    *,
    run_id: uuid.UUID,
    meta_id: uuid.UUID,
    producer: str,
    claims: tuple[tuple[str, str, float, dict[str, float]], ...],
) -> dict[tuple[str, str], uuid.UUID]:
    """Record ONE producer invocation carrying all its run_label name claims
    (a producer covers the whole run in a single generation, so splitting one
    producer across calls would supersede its own earlier candidates)."""
    drafts = tuple(
        CandidateDraft(
            target=EnrichmentScope.run_label(run_id, label),
            field=ClaimField.NAME,
            value=value,
            evidence=(
                MetadataEvidence(source_metadata_id=meta_id, source_field="title", snippet=value),
            ),
            score=score,
            score_components=components,
        )
        for label, value, score, components in claims
    )
    run = record_producer_run(
        session,
        producer=producer,
        producer_version="1",
        scope=EnrichmentScope.run(run_id),
        covered_fields=(ClaimField.NAME,),
        candidates=drafts,
        idempotency_key=f"{producer}:{run_id}",
        started_at=NOW,
        completed_at=NOW,
    )
    return {
        (c.diarization_label or "", c.value): c.id for c in run.candidates
    }


def _cosine(
    session: Session, *, run_id: uuid.UUID, label: str, name: str
) -> None:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run_id,
            diarization_label=label,
            speaker_id=speaker.id,
            method="cosine",
            confidence=0.9,
            grounded=True,
        )
    )
    session.flush()


def test_name_triage_orders_and_scores(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, meta_id = _run_with_metadata(session)
        # One offline invocation covers the whole run: "Jane Doe" at SPEAKER_00
        # and "Bob Smith" at SPEAKER_01.
        _name_run(
            session, run_id=run_id, meta_id=meta_id, producer="names.offline",
            claims=(
                ("SPEAKER_00", "Jane Doe", 0.85, {"base": 0.8}),
                ("SPEAKER_01", "Bob Smith", 0.7, {"base": 0.7}),
            ),
        )
        # The LLM pass also proposes "Jane Doe" at SPEAKER_00 (agreement).
        _name_run(
            session, run_id=run_id, meta_id=meta_id, producer="names.llm",
            claims=(("SPEAKER_00", "Jane Doe", 0.5, {"llm": 1.0}),),
        )
        # A grounded cosine names the same "Jane Doe" (voice support), and a
        # different identity for SPEAKER_01 (voice conflict).
        _cosine(session, run_id=run_id, label="SPEAKER_00", name="Jane Doe")
        _cosine(session, run_id=run_id, label="SPEAKER_01", name="Alice Jones")
        session.commit()

        _, per_label, triage = _name_suggestions(session, run_id)

        jane = per_label["SPEAKER_00"][0]
        jt = triage[jane.candidate.id]
        # name_match uses the offline `base` (0.8), voice matches (0.9),
        # 2 producers agree (0.5). priority = 0.95*(0.6*0.8 + 0.25*0.9 + 0.15*0.5).
        assert jt.components["name_match"] == 0.8
        assert jt.components["voice_support"] == 0.9
        assert jt.components["voice_conflict"] == 0.0
        assert jt.components["cross_source_agreement"] == 0.5
        assert jt.components["peer_producer_count"] == 2.0
        assert jt.priority == round(0.95 * (0.6 * 0.8 + 0.25 * 0.9 + 0.15 * 0.5), 4)

        bob = per_label["SPEAKER_01"][0]
        bt = triage[bob.candidate.id]
        assert bt.components["voice_conflict"] == 1.0
        assert bt.components["voice_support"] == 0.0


def test_merged_speaker_voice_is_excluded(session_factory: sessionmaker[Session]) -> None:
    """A cosine assignment to a since-merged speaker must NOT drive voice
    matching — its stale tombstone name would otherwise invert the signal."""
    with session_factory() as session:
        run_id, meta_id = _run_with_metadata(session)
        _name_run(
            session, run_id=run_id, meta_id=meta_id, producer="names.offline",
            claims=(("SPEAKER_00", "Jane Doe", 0.8, {"base": 0.8}),),
        )
        jane = Speaker(display_name="Jane Doe")
        canonical = Speaker(display_name="Jane Q. Doe")
        session.add_all([jane, canonical])
        session.flush()
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id, diarization_label="SPEAKER_00",
                speaker_id=jane.id, method="cosine", confidence=0.9, grounded=True,
            )
        )
        session.flush()
        jane.merged_into_id = canonical.id  # Jane is now a tombstone
        jane.merged_at = NOW  # constraint: merged_into_id and merged_at set together
        session.commit()

        _, per_label, triage = _name_suggestions(session, run_id)
        rep = per_label["SPEAKER_00"][0]
        # No voice row for the merged speaker → neither support nor conflict.
        assert triage[rep.candidate.id].components["voice_support"] == 0.0
        assert triage[rep.candidate.id].components["voice_conflict"] == 0.0


def test_rejected_claim_does_not_inflate_agreement(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, meta_id = _run_with_metadata(session)
        _name_run(
            session, run_id=run_id, meta_id=meta_id, producer="names.offline",
            claims=(("SPEAKER_00", "Jane Doe", 0.8, {"base": 0.8}),),
        )
        llm_ids = _name_run(
            session, run_id=run_id, meta_id=meta_id, producer="names.llm",
            claims=(("SPEAKER_00", "Jane Doe", 0.5, {"llm": 1.0}),),
        )
        llm_id = llm_ids[("SPEAKER_00", "Jane Doe")]
        # Reject the llm claim: it must no longer corroborate the offline one.
        session.add(
            ProfileReviewDecision(
                candidate_id=llm_id,
                decision=ProfileDecision.REJECT.value,
                operator="reviewer",
                idempotency_key=f"reject:{llm_id}",
            )
        )
        session.commit()

        _, per_label, triage = _name_suggestions(session, run_id)
        # The proposed offline claim represents; only it is active.
        rep = per_label["SPEAKER_00"][0]
        rt = triage[rep.candidate.id]
        assert rt.components["cross_source_agreement"] == 0.0
        assert rt.components["peer_producer_count"] == 1.0


def test_profile_triage_orders_by_priority_and_authority(
    session_factory: sessionmaker[Session],
) -> None:
    settings = Settings(_env_file=None, source_authority_domains="a.com, b.org")
    with session_factory() as session:
        seed_onboarded(session_factory)
        speaker = Speaker(display_name="Jane Doe")
        session.add(speaker)
        session.flush()
        # A strong claim: 3 distinct domains, 2 authority-listed.
        record_producer_run(
            session,
            producer="web_researcher",
            producer_version="1",
            scope=EnrichmentScope.speaker(speaker.id),
            covered_fields=(ClaimField.BIO, ClaimField.AFFILIATION, ClaimField.LINK),
            candidates=(
                CandidateDraft(
                    target=EnrichmentScope.speaker(speaker.id),
                    field=ClaimField.AFFILIATION,
                    value="Acme Corporation (chief scientist)",
                    evidence=(
                        UrlEvidence(url="https://a.com/jane", snippet="Jane at Acme, scientist"),
                        UrlEvidence(url="https://b.org/jane", snippet="Jane, chief scientist"),
                        UrlEvidence(url="https://c.net/jane", snippet="Acme names Jane lead"),
                    ),
                    score=0.5,
                    score_components={"web": 1.0},
                ),
                # A weak claim: a single, non-authority domain.
                CandidateDraft(
                    target=EnrichmentScope.speaker(speaker.id),
                    field=ClaimField.BIO,
                    value="Speaker on building science",
                    evidence=(
                        UrlEvidence(url="https://d.io/jane", snippet="Jane speaks on building"),
                    ),
                    score=0.5,
                    score_components={"web": 1.0},
                ),
            ),
            idempotency_key=f"web_researcher:{speaker.id}:job1",
            started_at=NOW,
            completed_at=NOW,
        )
        session.commit()

        state = _research_state(session, settings, speaker)
        proposed = state["proposed"]
        triage = state["triage"]
        # The 3-domain, authority-backed affiliation outranks the 1-domain bio.
        assert proposed[0].candidate.field == ClaimField.AFFILIATION.value
        strong = triage[proposed[0].candidate.id]
        assert strong.components["distinct_domains_count"] == 3.0
        assert strong.components["independent_domains"] == 1.0
        assert strong.components["source_authority"] == round(2 / 3, 4)
        assert strong.components["corroborated"] == 1.0
        assert strong.priority == round(0.95 * (0.5 * 1.0 + 0.5 * (2 / 3)), 4)

        weak = triage[proposed[1].candidate.id]
        assert weak.components["distinct_domains_count"] == 1.0
        assert weak.components["source_authority"] == 0.0
        assert weak.components["corroborated"] == 0.0
