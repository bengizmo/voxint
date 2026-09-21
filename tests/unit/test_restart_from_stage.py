"""Unit tests for restart-from-stage (#506 Slice B).

Tests the stage-aware blocker matrix (restart_impact), eager downstream
invalidation (_eager_invalidate_downstream), prerequisite validation
(_validate_prerequisites), and the extended restart_run with from_stage.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from voxint.db.models import RunStatus, Stage
from voxint.ingest.service import (
    RestartImpact,
    RestartPrerequisiteError,
    RunRestartLabelRiskError,
    RunRestartVoidRequiredError,
    _eager_invalidate_downstream,
    _validate_prerequisites,
    restart_impact,
    restart_run,
)

# ---------------------------------------------------------------------------
# restart_impact: stage-aware blocker matrix
# ---------------------------------------------------------------------------


class TestRestartImpactStageAware:
    """restart_impact(from_stage=...) gates differ by stage band."""

    def _session_with_counts(
        self,
        label_scope: int = 0,
        segment_scope: int = 0,
        evidence: int = 0,
        derived_embeddings: int = 0,
        corrections: int = 0,
        verifications: int = 0,
    ) -> MagicMock:
        """Return a mock session whose scalar() returns the given counts in
        the order restart_impact queries them."""
        session = MagicMock()
        session.scalar.side_effect = [
            label_scope,
            segment_scope,
            evidence,
            derived_embeddings,
            corrections,
            verifications,
        ]
        return session

    def test_enhance_match_always_clean(self) -> None:
        session = self._session_with_counts(label_scope=5, segment_scope=3, evidence=2)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.ENHANCE_MATCH)
        assert impact == RestartImpact(0, 0, 0)
        assert not impact.requires_void
        session.scalar.assert_not_called()

    def test_finalize_always_clean(self) -> None:
        session = self._session_with_counts()
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.FINALIZE)
        assert impact == RestartImpact(0, 0, 0)
        session.scalar.assert_not_called()

    def test_diarize_embed_drops_segment_blockers(self) -> None:
        session = MagicMock()
        session.scalar.side_effect = [7]  # only label_scope queried
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.DIARIZE_EMBED)
        assert impact.label_scope_decisions == 7
        assert impact.segment_scope_decisions == 0
        assert impact.enrichment_evidence == 0
        assert not impact.requires_void

    def test_diarize_embed_zero_labels(self) -> None:
        session = MagicMock()
        session.scalar.side_effect = [0]
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.DIARIZE_EMBED)
        assert impact == RestartImpact(0, 0, 0)

    def test_transcribe_full_counts(self) -> None:
        session = self._session_with_counts(
            label_scope=2,
            segment_scope=3,
            evidence=1,
            derived_embeddings=1,
        )
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.TRANSCRIBE)
        assert impact.label_scope_decisions == 2
        assert impact.segment_scope_decisions == 3
        assert impact.enrichment_evidence == 1
        assert impact.requires_void

    def test_prepare_requires_void(self) -> None:
        session = self._session_with_counts(label_scope=0, segment_scope=5, evidence=0)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.PREPARE)
        assert impact.requires_void

    def test_acquire_requires_void(self) -> None:
        session = self._session_with_counts(label_scope=0, segment_scope=0, evidence=3)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.ACQUIRE)
        assert impact.requires_void

    def test_none_from_stage_is_full_restart(self) -> None:
        session = self._session_with_counts(label_scope=1, segment_scope=2, evidence=3)
        impact = restart_impact(session, uuid.uuid4(), from_stage=None)
        assert impact.label_scope_decisions == 1
        assert impact.segment_scope_decisions == 2
        assert impact.enrichment_evidence == 3


# ---------------------------------------------------------------------------
# _eager_invalidate_downstream: per-stage delete sets
# ---------------------------------------------------------------------------


@patch("voxint.ingest.service._invalidate_run_embeddings", return_value=0)
@patch("voxint.ingest.service._void_run_decisions", return_value=0)
class TestEagerInvalidateDownstream:
    """Verify the right delete/update statements are issued per stage."""

    def _make_session(self) -> MagicMock:
        session = MagicMock()
        session.execute.return_value = MagicMock()
        return session

    def _execute_count(self, session: MagicMock) -> int:
        return session.execute.call_count

    def test_finalize_clears_only_review_claims(
        self,
        mock_void: MagicMock,
        mock_inv: MagicMock,
    ) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.FINALIZE)
        assert self._execute_count(session) == 1
        session.flush.assert_called_once()

    def test_enhance_match_clears_enhancement_and_match(
        self,
        mock_void: MagicMock,
        mock_inv: MagicMock,
    ) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.ENHANCE_MATCH)
        assert self._execute_count(session) == 4
        session.flush.assert_called_once()

    def test_diarize_embed_includes_enhance_match(
        self,
        mock_void: MagicMock,
        mock_inv: MagicMock,
    ) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.DIARIZE_EMBED)
        assert self._execute_count(session) == 7

    def test_transcribe_includes_diarize(
        self,
        mock_void: MagicMock,
        mock_inv: MagicMock,
    ) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.TRANSCRIBE)
        # Segments will be deleted, so enhancement UPDATE and label-null UPDATE
        # are skipped. assignments delete + candidates delete + synthdetect delete
        # + turns delete + segments delete + run metadata clear + review claim = 7
        assert self._execute_count(session) == 7
        mock_void.assert_called_once()
        mock_inv.assert_called_once()

    def test_prepare_includes_transcribe(
        self,
        mock_void: MagicMock,
        mock_inv: MagicMock,
    ) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.PREPARE)
        # transcribe (6) + artifacts delete + chunks delete + review claim = 9
        assert self._execute_count(session) == 9

    def test_acquire_same_as_prepare(
        self,
        mock_void: MagicMock,
        mock_inv: MagicMock,
    ) -> None:
        session = self._make_session()
        rid = uuid.uuid4()
        _eager_invalidate_downstream(session, rid, Stage.ACQUIRE)
        session_p = self._make_session()
        _eager_invalidate_downstream(session_p, rid, Stage.PREPARE)
        assert self._execute_count(session) == self._execute_count(session_p)


# ---------------------------------------------------------------------------
# _validate_prerequisites
# ---------------------------------------------------------------------------


class TestValidatePrerequisites:
    def test_acquire_checks_media_not_trashed(self) -> None:
        session = MagicMock()
        run = MagicMock()
        run.media_item_id = uuid.uuid4()
        session.get.side_effect = lambda model, id: (
            run if model.__name__ == "PipelineRun" else MagicMock(trashed_at=None, purged_at=None)
        )
        _validate_prerequisites(session, uuid.uuid4(), Stage.ACQUIRE)

    def test_acquire_rejects_trashed_media(self) -> None:
        session = MagicMock()
        run = MagicMock()
        run.media_item_id = uuid.uuid4()
        trashed_item = MagicMock()
        trashed_item.trashed_at = datetime.now(UTC)
        trashed_item.purged_at = None

        def get_side_effect(model: Any, id: Any) -> Any:
            if model.__name__ == "PipelineRun":
                return run
            return trashed_item

        session.get.side_effect = get_side_effect
        with pytest.raises(RestartPrerequisiteError, match="trashed or purged"):
            _validate_prerequisites(session, uuid.uuid4(), Stage.ACQUIRE)

    def test_non_acquire_requires_upstream_completed(self) -> None:
        session = MagicMock()
        session.scalar.return_value = 0  # no completed upstream stage
        with pytest.raises(RestartPrerequisiteError, match="has not completed"):
            _validate_prerequisites(session, uuid.uuid4(), Stage.ENHANCE_MATCH)

    def test_transcribe_requires_preprocessed_audio(self) -> None:
        session = MagicMock()
        # upstream completed (1), no audio artifact (0) -- raises before
        # _find_earliest_viable because we provide earliest_viable=Stage.PREPARE
        session.scalar.side_effect = [1, 0]
        with pytest.raises(RestartPrerequisiteError, match="preprocessed audio"):
            _validate_prerequisites(session, uuid.uuid4(), Stage.TRANSCRIBE)

    def test_transcribe_passes_with_audio(self, tmp_path: Path) -> None:
        session = MagicMock()
        (tmp_path / "normalized.wav").touch()
        session.scalar.side_effect = [1, "normalized.wav"]
        with patch("voxint.ingest.service.get_settings") as settings:
            settings.return_value.media_root = tmp_path
            _validate_prerequisites(session, uuid.uuid4(), Stage.TRANSCRIBE)

    def test_transcribe_rejects_deleted_audio_file(self, tmp_path: Path) -> None:
        session = MagicMock()
        session.scalar.side_effect = [1, "normalized.wav"]
        with patch("voxint.ingest.service.get_settings") as settings:
            settings.return_value.media_root = tmp_path
            with pytest.raises(RestartPrerequisiteError, match="preprocessed audio") as exc:
                _validate_prerequisites(session, uuid.uuid4(), Stage.TRANSCRIBE)
        assert exc.value.earliest_viable == Stage.PREPARE

    @pytest.mark.parametrize("exists", [False, True])
    @pytest.mark.parametrize("current_path", [None, "moved.wav"])
    def test_prepare_requires_media_file(
        self, tmp_path: Path, exists: bool, current_path: str | None
    ) -> None:
        session = MagicMock()
        session.scalar.return_value = 1  # historical ACQUIRE completion
        session.get.side_effect = [
            MagicMock(media_item_id=uuid.uuid4()),
            MagicMock(current_path=current_path, source_path="source.wav"),
        ]
        if exists:
            (tmp_path / (current_path or "source.wav")).touch()
        with patch("voxint.ingest.service.get_settings") as settings:
            settings.return_value.media_root = tmp_path
            if exists:
                _validate_prerequisites(session, uuid.uuid4(), Stage.PREPARE)
            else:
                with pytest.raises(RestartPrerequisiteError, match="media file") as exc:
                    _validate_prerequisites(session, uuid.uuid4(), Stage.PREPARE)
                assert exc.value.earliest_viable == Stage.ACQUIRE

    @pytest.mark.parametrize("segments", [0, 1])
    def test_enhance_restart_requires_segments_not_assignments(self, segments: int) -> None:
        session = MagicMock()
        session.scalar.side_effect = [1, segments]
        if segments:
            _validate_prerequisites(session, uuid.uuid4(), Stage.ENHANCE_MATCH)
        else:
            with pytest.raises(RestartPrerequisiteError, match="transcript segments") as exc:
                _validate_prerequisites(session, uuid.uuid4(), Stage.ENHANCE_MATCH)
            assert exc.value.earliest_viable == Stage.TRANSCRIBE
        query = str(session.scalar.call_args.args[0])
        assert "transcript_segments" in query
        assert "speaker_assignments" not in query

    def test_finalize_restart_allows_zero_proposals(self) -> None:
        session = MagicMock()
        session.scalar.side_effect = [1]  # only upstream completion is required
        _validate_prerequisites(session, uuid.uuid4(), Stage.FINALIZE)
        session.scalar.assert_called_once()

    def test_diarize_embed_requires_segments(self) -> None:
        session = MagicMock()
        # upstream completed (1), no segments (0)
        session.scalar.side_effect = [1, 0]
        with pytest.raises(RestartPrerequisiteError, match="no transcript segments"):
            _validate_prerequisites(session, uuid.uuid4(), Stage.DIARIZE_EMBED)

    def test_diarize_embed_passes_with_segments(self) -> None:
        session = MagicMock()
        # upstream completed (1), segments exist (5)
        session.scalar.side_effect = [1, 5]
        _validate_prerequisites(session, uuid.uuid4(), Stage.DIARIZE_EMBED)


# ---------------------------------------------------------------------------
# restart_run with from_stage
# ---------------------------------------------------------------------------


class TestRestartRunFromStage:
    def _mock_run(
        self,
        *,
        status: str = RunStatus.COMPLETED.value,
        archived: bool = False,
        revision: int = 5,
        cycle: int = 1,
    ) -> MagicMock:
        run = MagicMock()
        run.id = uuid.uuid4()
        run.status = status
        run.current_stage = Stage.FINALIZE.value
        run.revision = revision
        run.processing_cycle = cycle
        run.archived_at = datetime.now(UTC) if archived else None
        return run

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_from_stage_bumps_cycle(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(0, 0, 0)
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run(cycle=2)
        session.get.return_value = run

        restart_run(session, run.id, from_stage=Stage.ENHANCE_MATCH)

        mock_cas.assert_called_once()
        _, kwargs = mock_cas.call_args
        assert kwargs["processing_cycle"] == 3
        assert kwargs["current_stage"] is Stage.ENHANCE_MATCH
        assert kwargs["status"] is RunStatus.QUEUED

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_full_restart_also_bumps_cycle(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(0, 0, 0)
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run(cycle=1)
        session.get.return_value = run

        restart_run(session, run.id, from_stage=None)

        _, kwargs = mock_cas.call_args
        assert kwargs["processing_cycle"] == 2
        assert kwargs["current_stage"] is None

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_prerequisites_checked_for_from_stage(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(0, 0, 0)
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(session, run.id, from_stage=Stage.DIARIZE_EMBED)
        mock_prereq.assert_called_once_with(session, run.id, Stage.DIARIZE_EMBED)

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_full_restart_validates_acquire_prerequisites(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(0, 0, 0)
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(session, run.id, from_stage=None)
        mock_prereq.assert_called_once_with(session, run.id, Stage.ACQUIRE)

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    def test_blocked_by_segment_scope_raises(
        self,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(
            label_scope_decisions=0,
            segment_scope_decisions=5,
            enrichment_evidence=0,
        )
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        with pytest.raises(RunRestartVoidRequiredError):
            restart_run(session, run.id, from_stage=Stage.TRANSCRIBE)

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    def test_label_risk_without_ack_raises(
        self,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(
            label_scope_decisions=3,
            segment_scope_decisions=0,
            enrichment_evidence=0,
        )
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        with pytest.raises(RunRestartLabelRiskError):
            restart_run(session, run.id, from_stage=Stage.DIARIZE_EMBED)

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_label_risk_with_ack_proceeds(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(
            label_scope_decisions=3,
            segment_scope_decisions=0,
            enrichment_evidence=0,
        )
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(
            session,
            run.id,
            from_stage=Stage.DIARIZE_EMBED,
            acknowledge_label_risk=True,
        )
        mock_cas.assert_called_once()

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_eager_invalidation_always_called(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(0, 0, 0)
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(session, run.id, from_stage=Stage.ENHANCE_MATCH)
        mock_invalidate.assert_called_once_with(session, run.id, Stage.ENHANCE_MATCH)

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_full_restart_invalidates_from_acquire(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(0, 0, 0)
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(session, run.id, from_stage=None)
        mock_invalidate.assert_called_once_with(session, run.id, Stage.ACQUIRE)

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    def test_void_required_without_ack_raises(
        self,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(
            label_scope_decisions=2,
            segment_scope_decisions=5,
            enrichment_evidence=1,
        )
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        with pytest.raises(RunRestartVoidRequiredError):
            restart_run(session, run.id, from_stage=Stage.TRANSCRIBE)
        mock_invalidate.assert_not_called()

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_void_required_with_ack_proceeds(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        mock_impact.return_value = RestartImpact(
            label_scope_decisions=2,
            segment_scope_decisions=5,
            enrichment_evidence=1,
        )
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(
            session,
            run.id,
            from_stage=Stage.TRANSCRIBE,
            acknowledge_void=True,
        )
        mock_cas.assert_called_once()
        mock_invalidate.assert_called_once()

    @patch("voxint.ingest.service._eager_invalidate_downstream")
    @patch("voxint.ingest.service._validate_prerequisites")
    @patch("voxint.ingest.service.restart_impact")
    @patch("voxint.ingest.service.cas_update_run")
    def test_void_subsumes_label_risk(
        self,
        mock_cas: MagicMock,
        mock_impact: MagicMock,
        mock_prereq: MagicMock,
        mock_invalidate: MagicMock,
    ) -> None:
        """When acknowledge_void is set, label risk is subsumed."""
        mock_impact.return_value = RestartImpact(
            label_scope_decisions=5,
            segment_scope_decisions=3,
            enrichment_evidence=1,
        )
        mock_cas.return_value = MagicMock()
        session = MagicMock()
        run = self._mock_run()
        session.get.return_value = run

        restart_run(
            session,
            run.id,
            from_stage=Stage.TRANSCRIBE,
            acknowledge_void=True,
            acknowledge_label_risk=False,
        )
        mock_cas.assert_called_once()


# ---------------------------------------------------------------------------
# RestartPrerequisiteError shape
# ---------------------------------------------------------------------------


class TestRestartPrerequisiteError:
    def test_message_includes_stage_and_reason(self) -> None:
        err = RestartPrerequisiteError(
            uuid.uuid4(), Stage.TRANSCRIBE, "preprocessed audio is missing"
        )
        assert "transcribe" in str(err)
        assert "preprocessed audio is missing" in str(err)

    def test_message_includes_earliest_viable_hint(self) -> None:
        err = RestartPrerequisiteError(
            uuid.uuid4(),
            Stage.DIARIZE_EMBED,
            "upstream not completed",
            earliest_viable=Stage.PREPARE,
        )
        assert "prepare" in str(err)

    def test_no_hint_when_earliest_viable_is_none(self) -> None:
        err = RestartPrerequisiteError(uuid.uuid4(), Stage.ENHANCE_MATCH, "not ready")
        assert "earliest" not in str(err)


# ---------------------------------------------------------------------------
# RestartImpact gate profiles: parametric coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_stage",
    [Stage.ENHANCE_MATCH, Stage.FINALIZE],
    ids=["enhance_match", "finalize"],
)
def test_post_segment_stages_have_no_blockers(from_stage: Stage) -> None:
    session = MagicMock()
    impact = restart_impact(session, uuid.uuid4(), from_stage=from_stage)
    assert impact == RestartImpact(0, 0, 0)
    assert not impact.requires_void


@pytest.mark.parametrize(
    "from_stage",
    [Stage.ACQUIRE, Stage.PREPARE, Stage.TRANSCRIBE],
    ids=["acquire", "prepare", "transcribe"],
)
def test_pre_diarize_stages_query_all_counts(from_stage: Stage) -> None:
    session = MagicMock()
    session.scalar.side_effect = [1, 2, 3, 4, 5, 6]
    impact = restart_impact(session, uuid.uuid4(), from_stage=from_stage)
    assert impact.label_scope_decisions == 1
    assert impact.segment_scope_decisions == 2
    assert impact.enrichment_evidence == 3
    assert impact.derived_embeddings == 4
    assert impact.corrections == 5
    assert impact.verifications == 6
    assert session.scalar.call_count == 6


# ---------------------------------------------------------------------------
# Slice C: restart stage profiles and CLI (#506)
# ---------------------------------------------------------------------------


class TestRestartStageProfiles:
    def test_all_clean(self) -> None:
        from voxint.ingest.service import restart_stage_profiles

        profiles = restart_stage_profiles(RestartImpact(0, 0, 0))
        for profile in profiles:
            assert profile["safe"] is True
            assert profile["requires_void"] is False
            assert profile["label_risk"] is False
            assert profile["label_count"] == 0

    def test_with_label_scope_only(self) -> None:
        from voxint.ingest.service import restart_stage_profiles

        profiles = restart_stage_profiles(RestartImpact(3, 0, 0))
        for profile in profiles[:4]:
            assert profile["requires_void"] is False
            assert profile["label_risk"] is True
            assert profile["label_count"] == 3
            assert profile["safe"] is False
        for profile in profiles[4:]:
            assert profile["requires_void"] is False
            assert profile["label_risk"] is False
            assert profile["label_count"] == 0
            assert profile["safe"] is True

    def test_with_segment_scope_blockers(self) -> None:
        from voxint.ingest.service import restart_stage_profiles

        profiles = restart_stage_profiles(RestartImpact(2, 5, 1))
        for profile in profiles[:3]:
            assert profile["requires_void"] is True
            assert profile["label_risk"] is False
            assert profile["label_count"] == 2
            assert profile["safe"] is False
        assert profiles[3]["requires_void"] is False
        assert profiles[3]["label_risk"] is True
        assert profiles[3]["label_count"] == 2
        assert profiles[3]["safe"] is False
        for profile in profiles[4:]:
            assert profile["requires_void"] is False
            assert profile["label_risk"] is False
            assert profile["label_count"] == 0
            assert profile["safe"] is True

    def test_returns_correct_stage_values(self) -> None:
        from voxint.db.models import STAGE_ORDER
        from voxint.ingest.service import restart_stage_profiles

        profiles = restart_stage_profiles(RestartImpact(0, 0, 0))
        assert [profile["stage"] for profile in profiles] == [stage.value for stage in STAGE_ORDER]
        for profile in profiles:
            assert set(profile) == {
                "stage",
                "label",
                "requires_void",
                "label_risk",
                "label_count",
                "corrections",
                "verifications",
                "safe",
            }
            assert isinstance(profile["label"], str)
            assert profile["label"]

    def test_returns_six_profiles(self) -> None:
        from voxint.ingest.service import restart_stage_profiles

        assert len(restart_stage_profiles(RestartImpact(0, 0, 0))) == 6


def _parse_and_run(*args: str) -> int:
    from voxint.cli import build_parser

    ns = build_parser().parse_args(args)
    return ns.fn(ns)


@patch("voxint.db.session.build_engine")
@patch("voxint.db.session.build_session_factory")
@patch("voxint.db.session.session_scope")
@patch("voxint.ingest.restart_impact")
@patch("voxint.ingest.restart_run")
@patch("voxint.cli._publish_or_defer")
class TestRestartCLI:
    def test_restart_cli_invalid_uuid(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        assert _parse_and_run("restart", "bogus", "--yes") == 2
        assert "invalid run UUID" in capsys.readouterr().out
        mock_engine.assert_not_called()
        mock_scope.assert_not_called()
        mock_impact.assert_not_called()
        mock_restart.assert_not_called()
        mock_publish.assert_not_called()

    def test_restart_cli_run_not_found(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from voxint.db.models import PipelineRun

        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        session.get.return_value = None
        assert _parse_and_run("restart", str(run_id), "--yes") == 2
        assert "no run" in capsys.readouterr().out
        session.get.assert_called_once_with(PipelineRun, run_id)
        mock_impact.assert_not_called()
        mock_restart.assert_not_called()
        mock_publish.assert_not_called()

    def test_restart_cli_void_required(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        mock_impact.return_value = RestartImpact(2, 5, 1)
        assert _parse_and_run("restart", str(run_id), "--yes") == 2
        assert "--acknowledge-void" in capsys.readouterr().out
        mock_impact.assert_called_once_with(session, run_id, from_stage=None)
        mock_restart.assert_not_called()
        mock_publish.assert_not_called()

    def test_restart_cli_success_with_yes(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        def publish_after_commit(*args: object, **kwargs: object) -> bool:
            mock_scope.return_value.__exit__.assert_called_once_with(None, None, None)
            return True

        mock_publish.side_effect = publish_after_commit

        assert _parse_and_run("restart", str(run_id), "--yes") == 0
        assert f"restarted {run_id} (full restart)" in capsys.readouterr().out
        mock_engine.assert_called_once_with()
        mock_factory.assert_called_once_with(mock_engine.return_value)
        mock_scope.assert_called_once_with(mock_factory.return_value)
        mock_impact.assert_called_once_with(session, run_id, from_stage=None)
        mock_restart.assert_called_once_with(
            session,
            run_id,
            from_stage=None,
            expected_revision=1,
            acknowledge_label_risk=False,
            acknowledge_void=False,
            acknowledge_editorial=False,
        )
        mock_publish.assert_called_once_with(run_id, stage=None)

    def test_restart_cli_from_stage_with_yes(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        def publish_after_commit(*args: object, **kwargs: object) -> bool:
            mock_scope.return_value.__exit__.assert_called_once_with(None, None, None)
            return True

        mock_publish.side_effect = publish_after_commit

        assert _parse_and_run("restart", str(run_id), "--from-stage", "enhance_match", "--yes") == 0
        assert f"restarted {run_id} from enhance_match" in capsys.readouterr().out
        mock_engine.assert_called_once_with()
        mock_factory.assert_called_once_with(mock_engine.return_value)
        mock_scope.assert_called_once_with(mock_factory.return_value)
        mock_impact.assert_called_once_with(session, run_id, from_stage=Stage.ENHANCE_MATCH)
        mock_restart.assert_called_once_with(
            session,
            run_id,
            from_stage=Stage.ENHANCE_MATCH,
            expected_revision=1,
            acknowledge_label_risk=False,
            acknowledge_void=False,
            acknowledge_editorial=False,
        )
        mock_publish.assert_called_once_with(run_id, stage=Stage.ENHANCE_MATCH)

    def test_restart_cli_label_risk_needs_ack(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        mock_impact.return_value = RestartImpact(3, 0, 0)
        assert _parse_and_run("restart", str(run_id), "--yes") == 2
        assert "--acknowledge-label-risk" in capsys.readouterr().out
        mock_restart.assert_not_called()
        mock_publish.assert_not_called()

    def test_restart_cli_eof_on_stdin(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        with patch("builtins.input", side_effect=EOFError) as mock_input:
            assert _parse_and_run("restart", str(run_id)) == 2

        mock_input.assert_called_once_with("Restart? [y/N] ")
        mock_restart.assert_not_called()
        mock_publish.assert_not_called()

    def test_restart_cli_interactive_decline(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(0, 0, 0)

        with patch("builtins.input", return_value="n") as mock_input:
            assert _parse_and_run("restart", str(run_id)) == 2

        mock_input.assert_called_once_with("Restart? [y/N] ")
        mock_restart.assert_not_called()
        mock_publish.assert_not_called()

    def test_restart_cli_yes_with_label_ack_succeeds(
        self,
        mock_publish: MagicMock,
        mock_restart: MagicMock,
        mock_impact: MagicMock,
        mock_scope: MagicMock,
        mock_factory: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        run_id = uuid.uuid4()
        session = mock_scope.return_value.__enter__.return_value
        session.get.return_value = MagicMock(revision=1, status="completed", archived_at=None)
        mock_engine.return_value = object()
        mock_factory.return_value = object()
        mock_impact.return_value = RestartImpact(3, 0, 0)

        assert _parse_and_run("restart", str(run_id), "--yes", "--acknowledge-label-risk") == 0

        mock_restart.assert_called_once_with(
            session,
            run_id,
            from_stage=None,
            expected_revision=1,
            acknowledge_label_risk=True,
            acknowledge_void=False,
            acknowledge_editorial=False,
        )
        mock_publish.assert_called_once_with(run_id, stage=None)


@pytest.mark.parametrize(
    "flags, answers, expected",
    [
        (["--yes"], [], 2),
        (["--yes", "--acknowledge-editorial"], [], 0),
        ([], ["yes", "yes"], 0),
        ([], ["no"], 2),
        ([], [EOFError()], 2),
    ],
)
def test_cli_editorial_acknowledgement(monkeypatch, flags, answers, expected):
    import voxint.cli as cli
    import voxint.db.session as db_session
    import voxint.ingest as ingest

    run_id = uuid.uuid4()
    scope = MagicMock()
    scope.return_value.__enter__.return_value.get.return_value = MagicMock(
        revision=1, status="completed", archived_at=None
    )
    monkeypatch.setattr(db_session, "build_engine", MagicMock())
    monkeypatch.setattr(db_session, "build_session_factory", MagicMock())
    monkeypatch.setattr(db_session, "session_scope", scope)
    monkeypatch.setattr(
        ingest, "restart_impact", MagicMock(return_value=RestartImpact(0, 0, 0, corrections=1))
    )
    restart = MagicMock()
    publish = MagicMock()
    monkeypatch.setattr(ingest, "restart_run", restart)
    monkeypatch.setattr(cli, "_publish_or_defer", publish)
    prompt = MagicMock(side_effect=answers)
    monkeypatch.setattr("builtins.input", prompt)

    assert cli.main(["restart", str(run_id), *flags]) == expected
    if expected == 0:
        assert restart.call_args.kwargs["acknowledge_editorial"] is True
        publish.assert_called_once()
    else:
        restart.assert_not_called()
        publish.assert_not_called()
    assert prompt.call_count == len(answers)
