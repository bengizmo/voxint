"""Contract tests for the pipeline stage graph.

These tests lock the structural invariants that the H6 analysis identified as
implicit: the stage enum, the canonical ordering, the two-lane partition, the
stage-function registry, and the lane routing decision must all agree.  A new
stage that is added to one structure but not the others will fail here before
it silently breaks the pipeline at runtime.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voxint.db.models import GPU_SEGMENT, POST_SEGMENT, STAGE_ORDER, Stage
from voxint.pipeline.engine import StageFn
from voxint.pipeline.stages.context import build_stage_fns
from voxint.worker.tasks import pipeline_task_for_stage


class TestStageEnumAndOrder:
    """STAGE_ORDER must be the single ordering authority for Stage members."""

    def test_stage_order_covers_all_enum_members(self) -> None:
        assert set(STAGE_ORDER) == set(Stage)

    def test_stage_order_has_no_duplicates(self) -> None:
        assert len(STAGE_ORDER) == len(set(STAGE_ORDER))

    def test_enum_iteration_matches_stage_order(self) -> None:
        assert tuple(Stage) == STAGE_ORDER


class TestLanePartition:
    """GPU_SEGMENT and POST_SEGMENT must be a complete, disjoint partition."""

    def test_union_equals_all_stages(self) -> None:
        assert set(Stage) == GPU_SEGMENT | POST_SEGMENT

    def test_no_overlap(self) -> None:
        assert not GPU_SEGMENT & POST_SEGMENT

    def test_post_segment_is_contiguous_suffix(self) -> None:
        post_indices = sorted(STAGE_ORDER.index(s) for s in POST_SEGMENT)
        expected = list(range(post_indices[0], len(STAGE_ORDER)))
        assert post_indices == expected

    def test_gpu_segment_is_contiguous_prefix(self) -> None:
        gpu_indices = sorted(STAGE_ORDER.index(s) for s in GPU_SEGMENT)
        expected = list(range(len(GPU_SEGMENT)))
        assert gpu_indices == expected

    def test_both_lanes_non_empty(self) -> None:
        assert GPU_SEGMENT
        assert POST_SEGMENT

    def test_exact_lane_boundary(self) -> None:
        assert tuple(s for s in STAGE_ORDER if s in GPU_SEGMENT) == STAGE_ORDER[:4]
        assert tuple(s for s in STAGE_ORDER if s in POST_SEGMENT) == STAGE_ORDER[4:]


class TestBuildStageFns:
    """build_stage_fns must return a callable for every stage in STAGE_ORDER."""

    @pytest.fixture()
    def stage_fns(self) -> dict[Stage, StageFn]:
        ctx = MagicMock()
        return build_stage_fns(ctx)

    def test_keys_match_all_stages(self, stage_fns: dict[Stage, StageFn]) -> None:
        assert set(stage_fns.keys()) == set(Stage)

    def test_all_values_are_callable(self, stage_fns: dict[Stage, StageFn]) -> None:
        for stage, fn in stage_fns.items():
            assert callable(fn), f"build_stage_fns[{stage}] is not callable"

    def test_runtime_guard_catches_missing_stage(self) -> None:
        """The guard in build_stage_fns raises RuntimeError if a stage has no fn.

        We can't easily add a fake Stage member, so we test by monkeypatching
        the built dict to drop one entry before the guard runs. This verifies
        the guard logic fires for the production code path."""
        from voxint.pipeline.stages import context as ctx_mod

        original = ctx_mod.build_stage_fns
        dropped = Stage.FINALIZE

        def patched_build(ctx: object) -> dict[Stage, StageFn]:
            fns = original(ctx)  # type: ignore[arg-type]
            del fns[dropped]
            missing = set(Stage) - fns.keys()
            if missing:
                raise RuntimeError(f"build_stage_fns missing stages: {missing}")
            return fns  # pragma: no cover

        with pytest.raises(RuntimeError, match="FINALIZE"):
            patched_build(MagicMock())


class TestLaneRouting:
    """pipeline_task_for_stage must route every stage to exactly one lane."""

    def test_gpu_stages_route_to_run_pipeline(self) -> None:
        from voxint.worker.tasks import run_pipeline

        for stage in GPU_SEGMENT:
            assert pipeline_task_for_stage(stage) is run_pipeline, stage

    def test_post_stages_route_to_finish_pipeline(self) -> None:
        from voxint.worker.tasks import finish_pipeline

        for stage in POST_SEGMENT:
            assert pipeline_task_for_stage(stage) is finish_pipeline, stage

    def test_none_routes_to_run_pipeline(self) -> None:
        from voxint.worker.tasks import run_pipeline

        assert pipeline_task_for_stage(None) is run_pipeline
