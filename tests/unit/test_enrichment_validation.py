"""DB-free validation behavior of the enrichment drafts writer (issue #37).

The writer validates everything before touching the database and fails
closed. These tests pin the caps, the XOR scope shapes, the scope-containment
rule, and the structural URL policy.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from voxint.db.models import ClaimField
from voxint.enrichment.drafts import (
    MAX_EVIDENCE_ROWS,
    CandidateDraft,
    EnrichmentDraftError,
    EnrichmentScope,
    MetadataEvidence,
    TranscriptEvidence,
    UrlEvidence,
    _validate_candidate,
    _validate_invocation,
)

SPEAKER = uuid.uuid4()
RUN = uuid.uuid4()
META = uuid.uuid4()
SEGMENT = uuid.uuid4()
NOW = datetime.now(tz=UTC)

URL_EVIDENCE = (UrlEvidence(url="https://example.com/about"),)


def _speaker_draft(**overrides: object) -> CandidateDraft:
    defaults: dict[str, object] = {
        "target": EnrichmentScope.speaker(SPEAKER),
        "field": ClaimField.BIO,
        "value": "Host of an interview podcast.",
        "evidence": URL_EVIDENCE,
    }
    defaults.update(overrides)
    return CandidateDraft(**defaults)  # type: ignore[arg-type]


def _validate(draft: CandidateDraft, scope: EnrichmentScope | None = None) -> None:
    _validate_candidate(
        draft, scope or EnrichmentScope.speaker(SPEAKER), (ClaimField.BIO, ClaimField.NAME)
    )


class TestScope:
    def test_shapes_validate(self) -> None:
        EnrichmentScope.speaker(SPEAKER).validate()
        EnrichmentScope.run(RUN).validate()
        EnrichmentScope.run_label(RUN, "SPEAKER_00").validate()

    def test_malformed_shapes_refused(self) -> None:
        from voxint.db.models import EnrichmentTargetKind

        bad = EnrichmentScope(
            EnrichmentTargetKind.SPEAKER, speaker_id=SPEAKER, pipeline_run_id=RUN
        )
        with pytest.raises(EnrichmentDraftError, match="shape mismatch"):
            bad.validate()
        with pytest.raises(EnrichmentDraftError, match="non-empty"):
            EnrichmentScope.run_label(RUN, "   ").validate()

    def test_containment(self) -> None:
        run_scope = EnrichmentScope.run(RUN)
        assert run_scope.contains(EnrichmentScope.run(RUN))
        # a run-scope invocation may emit label-level candidates for that run
        assert run_scope.contains(EnrichmentScope.run_label(RUN, "SPEAKER_00"))
        assert not run_scope.contains(EnrichmentScope.run(uuid.uuid4()))
        assert not run_scope.contains(EnrichmentScope.run_label(uuid.uuid4(), "X"))
        assert not run_scope.contains(EnrichmentScope.speaker(SPEAKER))
        # narrower scopes admit only their exact target
        label_scope = EnrichmentScope.run_label(RUN, "SPEAKER_00")
        assert label_scope.contains(EnrichmentScope.run_label(RUN, "SPEAKER_00"))
        assert not label_scope.contains(EnrichmentScope.run_label(RUN, "SPEAKER_01"))
        assert not label_scope.contains(EnrichmentScope.run(RUN))
        speaker_scope = EnrichmentScope.speaker(SPEAKER)
        assert speaker_scope.contains(EnrichmentScope.speaker(SPEAKER))
        assert not speaker_scope.contains(EnrichmentScope.speaker(uuid.uuid4()))

    def test_lock_key_distinguishes_scopes(self) -> None:
        keys = {
            EnrichmentScope.speaker(SPEAKER).lock_key(),
            EnrichmentScope.run(RUN).lock_key(),
            EnrichmentScope.run_label(RUN, "SPEAKER_00").lock_key(),
            EnrichmentScope.run_label(RUN, "SPEAKER_01").lock_key(),
        }
        assert len(keys) == 4


class TestCandidateValidation:
    def test_valid_draft_passes(self) -> None:
        _validate(
            _speaker_draft(
                evidence=(
                    MetadataEvidence(source_metadata_id=META, source_field="title"),
                    TranscriptEvidence(
                        transcript_segment_id=SEGMENT,
                        timestamp_seconds=12.5,
                        snippet="I'm Jane and welcome to the show",
                    ),
                    UrlEvidence(url="https://example.com/about", retrieved_at=NOW),
                ),
                score=0.75,
                score_components={"name_match": 0.9, "source_diversity": 0.5},
            )
        )

    def test_target_outside_scope_refused(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="outside invocation scope"):
            _validate(_speaker_draft(target=EnrichmentScope.speaker(uuid.uuid4())))

    def test_field_not_covered_refused(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="not in covered_fields"):
            _validate(_speaker_draft(field=ClaimField.LINK))

    @pytest.mark.parametrize("value", ["", "   ", "x" * 4001])
    def test_bad_value_refused(self, value: str) -> None:
        with pytest.raises(EnrichmentDraftError, match="value"):
            _validate(_speaker_draft(value=value))

    def test_name_value_cap_tighter(self) -> None:
        _validate(_speaker_draft(field=ClaimField.NAME, value="J" * 120))
        with pytest.raises(EnrichmentDraftError, match="name value"):
            _validate(_speaker_draft(field=ClaimField.NAME, value="J" * 121))

    def test_evidence_count_bounds(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="evidence rows"):
            _validate(_speaker_draft(evidence=()))
        too_many = tuple(
            UrlEvidence(url=f"https://example.com/{i}")
            for i in range(MAX_EVIDENCE_ROWS + 1)
        )
        with pytest.raises(EnrichmentDraftError, match="evidence rows"):
            _validate(_speaker_draft(evidence=too_many))

    @pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
    def test_bad_score_refused(self, score: float) -> None:
        with pytest.raises(EnrichmentDraftError, match="score"):
            _validate(_speaker_draft(score=score))

    def test_bad_score_components_refused(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="component key"):
            _validate(_speaker_draft(score_components={"  ": 0.5}))
        with pytest.raises(EnrichmentDraftError, match="must be a number"):
            _validate(_speaker_draft(score_components={"flag": True}))
        with pytest.raises(EnrichmentDraftError, match="must be finite"):
            _validate(_speaker_draft(score_components={"x": float("nan")}))
        with pytest.raises(EnrichmentDraftError, match="score components"):
            _validate(
                _speaker_draft(score_components={f"k{i}": 0.1 for i in range(33)})
            )


class TestEvidenceValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "",
            "ftp://example.com/x",
            "https://user:pass@example.com/",
            "https://example.com/with space",
            "javascript:alert(1)",
            "https://e.com/" + "x" * 2048,
        ],
    )
    def test_structural_url_policy(self, url: str) -> None:
        with pytest.raises(EnrichmentDraftError, match="url"):
            _validate(_speaker_draft(evidence=(UrlEvidence(url=url),)))

    def test_snippet_bounds(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="snippet"):
            _validate(
                _speaker_draft(
                    evidence=(
                        UrlEvidence(url="https://example.com", snippet="x" * 1001),
                    )
                )
            )

    def test_detail_pairing(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="detail"):
            _validate(
                _speaker_draft(
                    evidence=(UrlEvidence(url="https://example.com", detail={"a": 1}),)
                )
            )
        with pytest.raises(EnrichmentDraftError, match="detail_schema_version"):
            _validate(
                _speaker_draft(
                    evidence=(
                        UrlEvidence(
                            url="https://example.com",
                            detail={"a": 1},
                            detail_schema_version=0,
                        ),
                    )
                )
            )

    def test_bad_transcript_timestamp_refused(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="timestamp_seconds"):
            _validate(
                _speaker_draft(
                    evidence=(
                        TranscriptEvidence(
                            transcript_segment_id=SEGMENT, timestamp_seconds=-1.0
                        ),
                    )
                )
            )

    def test_empty_source_field_refused(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="source_field"):
            _validate(
                _speaker_draft(
                    evidence=(
                        MetadataEvidence(source_metadata_id=META, source_field="  "),
                    )
                )
            )


class TestInvocationValidation:
    def _validate(self, **overrides: object) -> None:
        defaults: dict[str, object] = {
            "producer": "name_miner",
            "producer_version": "1.0",
            "scope": EnrichmentScope.speaker(SPEAKER),
            "covered": (ClaimField.NAME,),
            "idempotency_key": "k1",
            "started_at": NOW,
            "completed_at": NOW,
            "config": None,
            "config_schema_version": None,
        }
        defaults.update(overrides)
        _validate_invocation(*defaults.values())  # type: ignore[arg-type]

    def test_valid_invocation_passes(self) -> None:
        self._validate()
        self._validate(config={"budget": 3}, config_schema_version=1)

    @pytest.mark.parametrize("producer", ["", "   ", "p" * 201])
    def test_bad_producer_refused(self, producer: str) -> None:
        with pytest.raises(EnrichmentDraftError, match="producer"):
            self._validate(producer=producer)

    def test_covered_fields_rules(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="covered_fields"):
            self._validate(covered=())
        with pytest.raises(EnrichmentDraftError, match="duplicate"):
            self._validate(covered=(ClaimField.NAME, ClaimField.NAME))

    def test_timestamps_rules(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="timezone-aware"):
            self._validate(started_at=NOW.replace(tzinfo=None))
        with pytest.raises(EnrichmentDraftError, match="precedes"):
            self._validate(completed_at=NOW - timedelta(seconds=1))

    def test_config_rules(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="set together"):
            self._validate(config={"a": 1})
        with pytest.raises(EnrichmentDraftError, match=">= 1"):
            self._validate(config={"a": 1}, config_schema_version=0)
        with pytest.raises(EnrichmentDraftError, match="not JSON-serializable"):
            self._validate(config={"a": object()}, config_schema_version=1)
        with pytest.raises(EnrichmentDraftError, match="bytes"):
            self._validate(
                config={"blob": "x" * 17_000}, config_schema_version=1
            )

    def test_empty_idempotency_key_refused(self) -> None:
        with pytest.raises(EnrichmentDraftError, match="idempotency_key"):
            self._validate(idempotency_key="  ")


class TestReviewValidation:
    """record_profile_decision validates before touching the session, so a
    sentinel session object proves these paths never reach the database."""

    def _record(self, **overrides: object) -> None:
        from typing import cast

        from sqlalchemy.orm import Session

        from voxint.enrichment.review import record_profile_decision

        defaults: dict[str, object] = {
            "candidate_id": uuid.uuid4(),
            "decision": ClaimField.NAME,  # never reached on validation errors
            "operator": "ben",
            "idempotency_key": "k1",
            "note": None,
        }
        defaults.update(overrides)
        record_profile_decision(cast(Session, object()), **defaults)  # type: ignore[arg-type]

    @pytest.mark.parametrize("operator", ["", "   ", "o" * 201])
    def test_bad_operator_refused(self, operator: str) -> None:
        with pytest.raises(ValueError, match="operator"):
            self._record(operator=operator)

    @pytest.mark.parametrize("note", ["", "   ", "n" * 2001])
    def test_bad_note_refused(self, note: str) -> None:
        with pytest.raises(ValueError, match="note"):
            self._record(note=note)

    def test_empty_idempotency_key_refused(self) -> None:
        with pytest.raises(ValueError, match="idempotency_key"):
            self._record(idempotency_key="  ")
