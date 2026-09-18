"""Unit tests for restart-from-stage (#506 Slice B).

Tests the stage-aware blocker matrix (restart_impact), eager downstream
invalidation (_eager_invalidate_downstream), prerequisite validation
(_validate_prerequisites), and the extended restart_run with from_stage.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from voxint.db.models import RunStatus, Stage
from voxint.ingest.service import (
    RestartImpact,
    RestartPrerequisiteError,
    RunRestartBlockedError,
    RunRestartLabelRiskError,
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
    ) -> MagicMock:
        """Return a mock session whose scalar() returns the given counts in
        the order restart_impact queries them."""
        session = MagicMock()
        session.scalar.side_effect = [label_scope, segment_scope, evidence]
        return session

    def test_enhance_match_always_clean(self) -> None:
        session = self._session_with_counts(label_scope=5, segment_scope=3, evidence=2)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.ENHANCE_MATCH)
        assert impact == RestartImpact(0, 0, 0)
        assert not impact.has_blockers
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
        assert not impact.has_blockers

    def test_diarize_embed_zero_labels(self) -> None:
        session = MagicMock()
        session.scalar.side_effect = [0]
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.DIARIZE_EMBED)
        assert impact == RestartImpact(0, 0, 0)

    def test_transcribe_full_blockers(self) -> None:
        session = self._session_with_counts(label_scope=2, segment_scope=3, evidence=1)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.TRANSCRIBE)
        assert impact.label_scope_decisions == 2
        assert impact.segment_scope_decisions == 3
        assert impact.enrichment_evidence == 1
        assert impact.has_blockers

    def test_prepare_full_blockers(self) -> None:
        session = self._session_with_counts(label_scope=0, segment_scope=5, evidence=0)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.PREPARE)
        assert impact.has_blockers

    def test_acquire_full_blockers(self) -> None:
        session = self._session_with_counts(label_scope=0, segment_scope=0, evidence=3)
        impact = restart_impact(session, uuid.uuid4(), from_stage=Stage.ACQUIRE)
        assert impact.has_blockers

    def test_none_from_stage_is_full_restart(self) -> None:
        session = self._session_with_counts(label_scope=1, segment_scope=2, evidence=3)
        impact = restart_impact(session, uuid.uuid4(), from_stage=None)
        assert impact.label_scope_decisions == 1
        assert impact.segment_scope_decisions == 2
        assert impact.enrichment_evidence == 3


# ---------------------------------------------------------------------------
# _eager_invalidate_downstream: per-stage delete sets
# ---------------------------------------------------------------------------


class TestEagerInvalidateDownstream:
    """Verify the right delete/update statements are issued per stage."""

    def _make_session(self) -> MagicMock:
        session = MagicMock()
        session.execute.return_value = MagicMock()
        return session

    def _execute_count(self, session: MagicMock) -> int:
        return session.execute.call_count

    def test_finalize_clears_only_review_claims(self) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.FINALIZE)
        # Only the review-claim clear + flush
        assert self._execute_count(session) == 1
        session.flush.assert_called_once()

    def test_enhance_match_clears_enhancement_and_match(self) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.ENHANCE_MATCH)
        # enhanced_text update + speaker_assignments delete + match_candidates delete
        # + review claim clear = 4
        assert self._execute_count(session) == 4
        session.flush.assert_called_once()

    def test_diarize_embed_includes_enhance_match(self) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.DIARIZE_EMBED)
        # enhance_match (3) + synthdetect delete + turns delete + label null + review claim = 7
        assert self._execute_count(session) == 7

    def test_transcribe_includes_diarize(self) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.TRANSCRIBE)
        # Segments will be deleted, so enhancement UPDATE and label-null UPDATE
        # are skipped. assignments delete + candidates delete + synthdetect delete
        # + turns delete + segments delete + run metadata clear + review claim = 7
        assert self._execute_count(session) == 7

    def test_prepare_includes_transcribe(self) -> None:
        session = self._make_session()
        _eager_invalidate_downstream(session, uuid.uuid4(), Stage.PREPARE)
        # transcribe (6) + artifacts delete + chunks delete + review claim = 9
        assert self._execute_count(session) == 9

    def test_acquire_same_as_prepare(self) -> None:
        session = self._make_session()
        rid = uuid.uuid4()
        _eager_invalidate_downstream(session, rid, Stage.ACQUIRE)
        # ACQUIRE has no extra outputs beyond PREPARE
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

    def test_transcribe_passes_with_audio(self) -> None:
        session = MagicMock()
        session.scalar.side_effect = [1, 1]  # upstream completed + audio exists
        _validate_prerequisites(session, uuid.uuid4(), Stage.TRANSCRIBE)

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

        with pytest.raises(RunRestartBlockedError):
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
    assert not impact.has_blockers


@pytest.mark.parametrize(
    "from_stage",
    [Stage.ACQUIRE, Stage.PREPARE, Stage.TRANSCRIBE],
    ids=["acquire", "prepare", "transcribe"],
)
def test_pre_diarize_stages_query_all_three_counts(from_stage: Stage) -> None:
    session = MagicMock()
    session.scalar.side_effect = [1, 2, 3]
    impact = restart_impact(session, uuid.uuid4(), from_stage=from_stage)
    assert impact.label_scope_decisions == 1
    assert impact.segment_scope_decisions == 2
    assert impact.enrichment_evidence == 3
    assert session.scalar.call_count == 3
